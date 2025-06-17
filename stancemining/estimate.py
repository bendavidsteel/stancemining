from typing import Optional, Any, Dict

import gpytorch
from gpytorch.priors import Prior, NormalPrior
from gpytorch.constraints import Interval, Positive
from linear_operator.utils.errors import NotPSDError
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def get_inferred_normal(dataset, opinion_sequences, all_classifier_indices):
    estimator = StanceEstimation(dataset.all_classifier_profiles)

    user_stances = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns)))
    user_stance_vars = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns)))

    # setup the optimizer
    num_steps = 1000
    optim = _get_optimizer(num_steps)
    if opinion_sequences.shape[0] > 1000:
        device = torch.device("cpu") # currently too big for GPU
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for stance_idx, stance_column in enumerate(dataset.stance_columns):

        op_seqs, lengths, all_predictor_ids, max_length = _get_op_seqs(opinion_sequences, stance_idx, all_classifier_indices)
        if max_length == 0:
            continue

        stance_opinion_sequences, classifier_indices, mask = _get_op_seq_with_mask(opinion_sequences, max_length, op_seqs, lengths, all_predictor_ids)
        
        if stance_column not in estimator.predictor_confusion_probs or len(estimator.predictor_confusion_probs[stance_column]) == 0:
            continue
        estimator.set_stance(stance_column)
        stance_opinion_sequences, classifier_indices, mask = _data_to_torch_device(stance_opinion_sequences, classifier_indices, mask)

        estimator.predictor_confusion_probs[stance_column]['predict_probs'] = estimator.predictor_confusion_probs[stance_column]['predict_probs'].to(device)
        estimator.predictor_confusion_probs[stance_column]['true_probs'] = estimator.predictor_confusion_probs[stance_column]['true_probs'].to(device)

        prior = True
        _train_model(estimator.model, estimator.guide, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps)

        # grab the learned variational parameters
        if prior:
            user_stances[:, stance_idx] = pyro.param("user_stance_loc_q").cpu().detach().numpy().squeeze(-1)
            user_stance_vars[:, stance_idx] = pyro.param("user_stance_var_q").cpu().detach().numpy().squeeze(-1)
        else:
            user_stances[:, stance_idx] = pyro.param("user_stance").detach().numpy().squeeze(-1)
            user_stance_vars[:, stance_idx, 1] = pyro.param("user_stance_var").detach().numpy().squeeze(-1)
        # set parameters to nan where no data was available
        user_stances[lengths == 0, stance_idx, 0] = np.nan
        user_stance_vars[lengths == 0, stance_idx, 1] = np.nan

    return user_stances, user_stance_vars

class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, learn_inducing_locations=False, lengthscale_loc=1.0, lengthscale_scale=0.5, variational_dist='natural'):
        if variational_dist == 'cholesky':
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        elif variational_dist == 'natural':
            variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        else:
            raise NotImplementedError("Variational distribution type not supported.")
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(constant_constraint=gpytorch.constraints.Interval(-1, 1))
        lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=torch.log(torch.tensor(lengthscale_loc)), scale=lengthscale_scale)
        # TODO consider switching to rational quadratic kernel https://www.cs.toronto.edu/~duvenaud/cookbook/
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

def inv_probit(x, jitter=1e-3):
    """
    Inverse probit function (standard normal CDF) with jitter for numerical stability.
    
    Args:
        x: Input tensor
        jitter: Small constant to ensure outputs are strictly between 0 and 1
        
    Returns:
        Probabilities between jitter and 1-jitter
    """
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2 * jitter) + jitter

class BoundedOrdinalWithErrorLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(
            self, 
            bin_edges: torch.Tensor, 
            classifier_profiles,
            batch_shape: torch.Size = torch.Size([]),
            sigma_prior: Optional[Prior] = None,
            sigma_constraint: Optional[Interval] = None,
        ):
        super().__init__()
        self.predictor_confusion_probs = get_target_predictor_confusion_probs(classifier_profiles)
        if torch.cuda.is_available():
            self.predictor_confusion_probs['predict_probs'] = self.predictor_confusion_probs['predict_probs'].to('cuda')
            self.predictor_confusion_probs['true_probs'] = self.predictor_confusion_probs['true_probs'].to('cuda')
        
        self.num_bins = len(bin_edges) + 1
        self.register_parameter("bin_edges", torch.nn.Parameter(bin_edges, requires_grad=False))

        if sigma_constraint is None:
            sigma_constraint = Positive()

        self.raw_sigma = torch.nn.Parameter(torch.ones(*batch_shape, 1))
        if sigma_prior is not None:
            self.register_prior("sigma_prior", sigma_prior, lambda m: m.sigma, lambda m, v: m._set_sigma(v))

        self.register_constraint("raw_sigma", sigma_constraint)

    @property
    def sigma(self) -> torch.Tensor:
        return self.raw_sigma_constraint.transform(self.raw_sigma)

    @sigma.setter
    def sigma(self, value: torch.Tensor) -> None:
        self._set_sigma(value)

    def _set_sigma(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma)
        self.initialize(raw_sigma=self.raw_sigma_constraint.inverse_transform(value))

    def forward(self, function_samples: torch.Tensor, *args: Any, data: Dict[str, torch.Tensor] = {}, **kwargs: Any):
        assert 'classifier_ids' in kwargs, "Classifier IDs not provided in kwargs"

        classifier_ids = kwargs['classifier_ids']

        function_samples = torch.tanh(function_samples)

        # Compute scaled bin edges
        scaled_edges = self.bin_edges / self.sigma
        scaled_edges_left = torch.cat([scaled_edges, torch.tensor([torch.inf], device=scaled_edges.device)], dim=-1)
        scaled_edges_right = torch.cat([torch.tensor([-torch.inf], device=scaled_edges.device), scaled_edges])
        
        # Calculate cumulative probabilities using standard normal CDF (probit function)
        # These represent P(Y ≤ k | F)
        function_samples = function_samples.unsqueeze(-1)
        scaled_function_samples = function_samples / self.sigma
        scaled_edges_left = scaled_edges_left.reshape(1, 1, -1)
        scaled_edges_right = scaled_edges_right.reshape(1, 1, -1)
        probs = inv_probit(scaled_edges_left - scaled_function_samples) - inv_probit(scaled_edges_right - scaled_function_samples)
        
        # Apply confusion matrix
        predict_probs = torch.einsum('sxc,xco->sxo', probs, self.predictor_confusion_probs['predict_probs'][classifier_ids])

        return torch.distributions.Categorical(probs=predict_probs)

def get_ordinal_gp_model(train_x, train_y, classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, sigma_loc=1.0, sigma_scale=0.5):
    bin_edges = torch.tensor([-0.5, 0.5])
    # likelihood = OrdinalLikelihood(bin_edges)
    sigma_prior = NormalPrior(sigma_loc, sigma_scale)
    likelihood = BoundedOrdinalWithErrorLikelihood(bin_edges, classifier_profiles, sigma_prior=sigma_prior)
    
    max_inducing_points = 1000
    if train_x.size(0) > max_inducing_points:
        learn_inducing_locations = True
        perm = torch.randperm(train_x.size(0))
        idx = perm[:max_inducing_points]
        inducing_points = train_x[idx]
    else:
        learn_inducing_locations = False
        inducing_points = train_x

    model = GPClassificationModel(
        inducing_points, 
        learn_inducing_locations=learn_inducing_locations, 
        lengthscale_loc=lengthscale_loc, 
        lengthscale_scale=lengthscale_scale,
        variational_dist='natural'
    )
    return model, likelihood

def setup_ordinal_gp_model(timestamps, stance, classifier_ids, classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc=1.0, sigma_scale=0.1):
    timestamps = torch.tensor(timestamps, dtype=torch.float32)
    stance = torch.tensor(stance, dtype=torch.float32) + 1
    classifier_ids = torch.tensor(classifier_ids, dtype=torch.int)

    model, likelihood = get_ordinal_gp_model(timestamps, stance, classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, sigma_loc=sigma_loc, sigma_scale=sigma_scale)
    return model, likelihood, timestamps, stance, classifier_ids

def optimize_step(optimizers, model, mll, likelihood, batch_X, batch_y, batch_classifier_ids):
    for optimizer in optimizers:
        optimizer.zero_grad()
    with gpytorch.settings.variational_cholesky_jitter(1e-4):
        output = model(batch_X)
        loss = -mll(output, batch_y, classifier_ids=batch_classifier_ids)
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=1.0)
    for optimizer in optimizers:
        optimizer.step()
    return loss


def train_ordinal_likelihood_gp(model: GPClassificationModel, likelihood: BoundedOrdinalWithErrorLikelihood, train_X: torch.Tensor, train_y: torch.Tensor, classifier_ids: torch.Tensor, verbose=True):
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.size(0))
    return train_ordinal_likelihood_gp_with_mll(model, likelihood, mll, train_X, train_y, classifier_ids, verbose=verbose)

def train_ordinal_likelihood_gp_with_mll(
        model: GPClassificationModel, 
        likelihood: BoundedOrdinalWithErrorLikelihood, 
        mll,
        train_X: torch.Tensor, 
        train_y: torch.Tensor, 
        classifier_ids: torch.Tensor,
        verbose=True
    ):
    max_full_batch = 10000

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_X = train_X.cuda()
        train_y = train_y.cuda()
    model.train()
    likelihood.train()
    losses = []
    if isinstance(model.variational_strategy._variational_distribution, gpytorch.variational.CholeskyVariationalDistribution):
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            {'params': likelihood.parameters()},
        ], lr=0.05)
        optimizers = [optimizer]
    elif isinstance(model.variational_strategy._variational_distribution, gpytorch.variational.NaturalVariationalDistribution):
        variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1)

        hyperparameter_optimizer = torch.optim.Adam([
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)
        optimizers = [hyperparameter_optimizer, variational_ngd_optimizer]
    else:
        raise NotImplementedError("Unrecognized variational distribution.")

    training_iter = 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], int(training_iter / 10))
    best_loss = torch.tensor(float('inf'))
    num_since_best = 0
    num_iters_before_stopping = training_iter // 20
    min_loss_improvement = 0.0001

    with gpytorch.settings.cholesky_max_tries(10):
        if train_X.size(0) < max_full_batch:
            
            
            for k in range(training_iter):
                loss = optimize_step(optimizers, model, mll, likelihood, train_X, train_y, classifier_ids)
        
                if k % 50 == 0 and verbose:
                    print('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, loss.item()))
                
                scheduler.step(k)
                losses.append(loss.item())
                # checking since substantive decrease in loss
                if best_loss - loss.item() > min_loss_improvement:
                    best_loss = loss.item()
                    num_since_best = 0
                else:
                    num_since_best += 1
                if num_since_best > num_iters_before_stopping:
                            break
        else:
            train_dataset = TensorDataset(train_X, train_y, classifier_ids)
            train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)

            for k in range(training_iter):
                epoch_losses = []
                for batch_X, batch_y, batch_classifier_ids in tqdm(train_loader, desc='Epoch Batches'):
                    loss = optimize_step(optimizers, model, mll, likelihood, batch_X, batch_y, batch_classifier_ids)
                    epoch_losses.append(loss.item())

                epoch_loss = sum(epoch_losses) / len(epoch_losses)

                scheduler.step(k)

                if k % 50 == 0 and verbose:
                    print('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, epoch_loss))

                losses.append(epoch_loss)
                # checking since substantive decrease in loss
                if best_loss - epoch_loss > min_loss_improvement:
                    best_loss = epoch_loss
                    num_since_best = 0
                else:
                    num_since_best += 1
                if num_since_best > num_iters_before_stopping:
                    break

    return model, likelihood, losses

def get_likelihood_prediction(model, likelihood, test_x):
    if not isinstance(test_x, torch.Tensor):
        test_x = torch.tensor(test_x).float()
    classifier_ids = torch.zeros_like(test_x, dtype=torch.int, device=test_x.device)
    model.eval()
    likelihood.eval()
    test_x = test_x.cuda()
    with torch.no_grad():
        try:
            with gpytorch.settings.fast_pred_var():
                model_pred = model(test_x)
                observed_pred = likelihood(model_pred, classifier_ids=classifier_ids) # returns samples that are run through the likelihood
        except NotPSDError:
            model_pred = model(test_x)
            observed_pred = likelihood(model_pred, classifier_ids=classifier_ids) # returns samples that are run through the likelihood
    return observed_pred

def get_model_prediction(model, test_x):
    test_x = torch.tensor(test_x).float()

    model.eval()
    model = model.cuda()
    test_x = test_x.cuda()

    with torch.no_grad():
        try:
            with gpytorch.settings.fast_pred_var():
                model_pred = model(test_x)
                
                # Get upper and lower confidence bounds
                lower, upper = model_pred.confidence_region()
        except NotPSDError:
            model_pred = model(test_x)
                
            # Get upper and lower confidence bounds
            lower, upper = model_pred.confidence_region()

    return torch.tanh(model_pred.loc).cpu().numpy(), torch.tanh(lower).cpu().numpy(), torch.tanh(upper).cpu().numpy()

