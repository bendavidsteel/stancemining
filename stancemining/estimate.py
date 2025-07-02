import datetime
import json
import multiprocessing
import re
from typing import Optional, Any, Dict, List, Tuple, Iterable, Union, Callable

import gpytorch
from gpytorch.priors import Prior, NormalPrior
from gpytorch.constraints import Interval, Positive
from gpytorch.likelihoods.likelihood_list import _get_tuple_args_, LikelihoodList
from gpytorch.mlls import VariationalELBO, MarginalLogLikelihood
from gpytorch.utils.generic import length_safe_zip
import huggingface_hub
from linear_operator.utils.errors import NotPSDError
import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from stancemining.finetune import LABELS_2_ID
from stancemining.main import logger

MAX_FULL_GP_BATCH = 10000

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

@torch.jit.script
def inv_probit(x, jitter: float = 1e-3):
    """
    Inverse probit function (standard normal CDF) with jitter for numerical stability.
    
    Args:
        x: Input tensor
        jitter: Small constant to ensure outputs are strictly between 0 and 1
        
    Returns:
        Probabilities between jitter and 1-jitter
    """
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2 * jitter) + jitter

def get_predict_probs(true_cat, confusion_profile):
    predict_sum = sum(v for k, v in confusion_profile[true_cat].items())
    if predict_sum == 0:
        return torch.tensor([1/3, 1/3, 1/3])  # Uniform distribution if no predictions
    predict_probs = torch.tensor([
        confusion_profile[true_cat]["predicted_against"] / predict_sum,
        confusion_profile[true_cat]["predicted_neutral"] / predict_sum,
        confusion_profile[true_cat]["predicted_favor"] / predict_sum,
    ])
    return predict_probs

def get_true_probs(predicted_cat, confusion_profile):
    true_sum = sum(v[predicted_cat] for k, v in confusion_profile.items())
    if true_sum == 0:
        return torch.tensor([1/3, 1/3, 1/3])
    true_probs = torch.tensor([
        confusion_profile["true_against"][predicted_cat] / true_sum,
        confusion_profile["true_neutral"][predicted_cat] / true_sum,
        confusion_profile["true_favor"][predicted_cat] / true_sum,
    ])
    return true_probs

def get_target_predictor_confusion_probs(classifier_profiles):
    predict_probs = torch.zeros(len(classifier_profiles), 3, 3)
    true_probs = torch.zeros(len(classifier_profiles), 3, 3)

    assert len(classifier_profiles) == max(classifier_profiles.keys()) + 1
    for predictor_id in classifier_profiles:
        classifier_profile = classifier_profiles[predictor_id]

        try:
            confusion_profile = {
                "true_favor": classifier_profile["true_favor"],
                "true_against": classifier_profile["true_against"],
                "true_neutral": classifier_profile["true_neutral"],
            }
        except KeyError:
            continue

        for true_idx, true_cat in enumerate(["true_against", "true_neutral", "true_favor"]):
            predict_probs[predictor_id, true_idx, :] = get_predict_probs(true_cat, confusion_profile)

        for predicted_idx, predicted_cat in enumerate(["predicted_against", "predicted_neutral", "predicted_favor"]):
            true_probs[predictor_id, predicted_idx, :] = get_true_probs(predicted_cat, confusion_profile)

    return {
        'predict_probs': predict_probs,
        'true_probs': true_probs,
    }

@torch.jit.script
def get_ordinal_probs(function_samples, bin_edges, sigma, predictor_predict_probs):
    function_samples = torch.tanh(function_samples)

    # Compute scaled bin edges
    scaled_edges = bin_edges / sigma
    scaled_edges_left = torch.cat([scaled_edges, torch.tensor([torch.inf], device=scaled_edges.device)], dim=-1)
    scaled_edges_right = torch.cat([torch.tensor([-torch.inf], device=scaled_edges.device), scaled_edges])
    
    # Calculate cumulative probabilities using standard normal CDF (probit function)
    # These represent P(Y ≤ k | F)
    function_samples = function_samples.unsqueeze(-1)
    scaled_function_samples = function_samples / sigma
    scaled_edges_left = scaled_edges_left.reshape(1, 1, -1)
    scaled_edges_right = scaled_edges_right.reshape(1, 1, -1)
    probs = inv_probit(scaled_edges_left - scaled_function_samples) - inv_probit(scaled_edges_right - scaled_function_samples)
    
    # Apply confusion matrix
    predict_probs = torch.einsum('sxc,xco->sxo', probs, predictor_predict_probs)
    return predict_probs

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
        predictor_predict_probs = get_target_predictor_confusion_probs(classifier_profiles)['predict_probs']
        self.register_parameter('predictor_predict_probs', torch.nn.Parameter(predictor_predict_probs, requires_grad=False))
        
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
        predictor_predict_probs = self.predictor_predict_probs[classifier_ids]

        predict_probs = get_ordinal_probs(function_samples, self.bin_edges, self.sigma, predictor_predict_probs)

        return torch.distributions.Categorical(probs=predict_probs)


class SumVariationalELBO(MarginalLogLikelihood):
    """Sum of marginal log likelihoods, to be used with Multi-Output models.

    Args:
        likelihood: A MultiOutputLikelihood
        model: A MultiOutputModel
        mll_cls: The Marginal Log Likelihood class (default: ExactMarginalLogLikelihood)

    In case the model outputs are independent, this provives the MLL of the multi-output model.

    """

    def __init__(self, likelihood, model, mll_cls=VariationalELBO):
        super().__init__(model.likelihood, model)
        self.mlls = torch.nn.ModuleList([mll_cls(mdl.likelihood, mdl, mdl.train_targets.size(0)) for mdl in model.models])

    def forward(self, outputs, targets, **kwargs):
        """
        Args:
            outputs: (Iterable[MultivariateNormal]) - the outputs of the latent function
            targets: (Iterable[Tensor]) - the target values
            params: (Iterable[Iterable[Tensor]]) - the arguments to be passed through
                (e.g. parameters in case of heteroskedastic likelihoods)
        """
        if len(kwargs) == 0:
            sum_mll = sum(mll(output, target) for mll, output, target in length_safe_zip(self.mlls, outputs, targets))
        else:
            sum_mll = sum(
                mll(output, target, **{k: v for k, v in zip(kwargs.keys(), k_vals)})
                for mll, output, target, *k_vals in length_safe_zip(self.mlls, outputs, targets, *kwargs.values())
            )
        return sum_mll.div_(len(self.mlls))

class FullBatchNGD(torch.optim.Optimizer):
    r"""Implements a natural gradient descent step.
    It **can only** be used in conjunction with a :obj:`~gpytorch.variational._NaturalVariationalDistribution`.

    .. seealso::
        - :obj:`gpytorch.variational.NaturalVariationalDistribution`
        - :obj:`gpytorch.variational.TrilNaturalVariationalDistribution`
        - The `natural gradient descent tutorial
          <examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb>`_
          for use instructions.

    Example:
        >>> ngd_optimizer = torch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1)
        >>> ngd_optimizer.zero_grad()
        >>> mll(gp_model(input), target).backward()
        >>> ngd_optimizer.step()
    """

    def __init__(self, params: Iterable[Union[torch.nn.Parameter, dict]], lr: float = 0.1):
        super().__init__(params, defaults=dict(lr=lr))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        """
        Performs a single optimization step.

        (Note that the :attr:`closure` argument is not used by this optimizer; it is simply included to be
        compatible with the PyTorch optimizer API.)
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=(-group["lr"] * p.size(0)))

        return None

class ApproximateGPModelListModel(GPClassificationModel):
    def __init__(self, *args, train_inputs=None, train_targets=None, likelihood=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.likelihood = likelihood

class LikelihoodList(LikelihoodList):
    def forward(self, *args, **kwargs):
        return [
            likelihood.forward(*args_, **{k: v for k, v in zip(kwargs.keys(), k_vals)})
            for likelihood, args_, *k_vals in length_safe_zip(self.likelihoods, _get_tuple_args_(*args), *kwargs.values())
        ]

    def __call__(self, *args, **kwargs):
        return [
            likelihood(*args_, **{k: v for k, v in zip(kwargs.keys(), k_vals)})
            for likelihood, args_, *k_vals in length_safe_zip(self.likelihoods, _get_tuple_args_(*args), *kwargs.values())
        ]

def get_inducing_points(train_x):
    max_inducing_points = 1000
    if train_x.size(0) > max_inducing_points:
        learn_inducing_locations = True
        perm = torch.randperm(train_x.size(0))
        idx = perm[:max_inducing_points]
        inducing_points = train_x[idx]
    else:
        learn_inducing_locations = False
        inducing_points = train_x

    return inducing_points, learn_inducing_locations

def get_ordinal_gp_model(train_x, train_y, classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, sigma_loc=1.0, sigma_scale=0.5):
    bin_edges = torch.tensor([-0.5, 0.5])
    # likelihood = OrdinalLikelihood(bin_edges)
    sigma_prior = NormalPrior(sigma_loc, sigma_scale)
    likelihood = BoundedOrdinalWithErrorLikelihood(bin_edges, classifier_profiles, sigma_prior=sigma_prior)
    
    inducing_points, learn_inducing_locations = get_inducing_points(train_x)

    model = GPClassificationModel(
        inducing_points, 
        learn_inducing_locations=learn_inducing_locations, 
        lengthscale_loc=lengthscale_loc, 
        lengthscale_scale=lengthscale_scale,
        variational_dist='natural'
    )
    return model, likelihood

def inputs_np_to_torch(timestamps, stance, classifier_ids):
    timestamps = torch.as_tensor(timestamps, dtype=torch.float32)
    stance = torch.as_tensor(stance, dtype=torch.float32) + 1
    classifier_ids = torch.as_tensor(classifier_ids, dtype=torch.int)
    return timestamps, stance, classifier_ids

def setup_ordinal_gp_model(timestamps, stance, classifier_ids, classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc=1.0, sigma_scale=0.1):
    timestamps, stance, classifier_ids = inputs_np_to_torch(timestamps, stance, classifier_ids)

    model, likelihood = get_ordinal_gp_model(timestamps, stance, classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, sigma_loc=sigma_loc, sigma_scale=sigma_scale)

    return model, likelihood, timestamps, stance, classifier_ids

def get_batchable_ordinal_gp_model(train_x, train_y, classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, sigma_loc=1.0, sigma_scale=0.5):
    bin_edges = torch.tensor([-0.5, 0.5])
    # likelihood = OrdinalLikelihood(bin_edges)
    sigma_prior = NormalPrior(sigma_loc, sigma_scale)
    likelihood = BoundedOrdinalWithErrorLikelihood(bin_edges, classifier_profiles, sigma_prior=sigma_prior)
    
    inducing_points, learn_inducing_locations = get_inducing_points(train_x)

    model = ApproximateGPModelListModel(
        inducing_points, 
        learn_inducing_locations=learn_inducing_locations, 
        lengthscale_loc=lengthscale_loc, 
        lengthscale_scale=lengthscale_scale,
        variational_dist='natural',
        train_inputs=train_x,
        train_targets=train_y,
        likelihood=likelihood
    )
    return model

def setup_batchable_ordinal_gp_model(timestamps, stance, classifier_ids, classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc=1.0, sigma_scale=0.1):
    timestamps, stance, classifier_ids = inputs_np_to_torch(timestamps, stance, classifier_ids)

    model = get_batchable_ordinal_gp_model(timestamps, stance, classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, sigma_loc=sigma_loc, sigma_scale=sigma_scale)
    
    return model, classifier_ids

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

def get_optimizer(model, likelihood, num_data=None):
    adam_params = []
    if isinstance(model, gpytorch.models.IndependentModelList):
        variational_distribution = model.models[0].variational_strategy._variational_distribution
        # don't add likelihood parameters if they are already included in the model
    else:
        variational_distribution = model.variational_strategy._variational_distribution
        adam_params.append({'params': likelihood.parameters()})

    if isinstance(variational_distribution, gpytorch.variational.CholeskyVariationalDistribution):
        optimizer = torch.optim.Adam(
            adam_params + [{'params': model.parameters()}],  # Includes GaussianLikelihood parameters
            lr=0.05
        )
        optimizers = [optimizer]
    elif isinstance(variational_distribution, gpytorch.variational.NaturalVariationalDistribution):
        if isinstance(model, gpytorch.models.IndependentModelList):
            variational_ngd_optimizer = FullBatchNGD(model.variational_parameters(), lr=0.1)
        else:
            assert num_data is not None, "num_data must be provided for NGD optimizer"
            variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=num_data, lr=0.1)

        hyperparameter_optimizer = torch.optim.Adam(
            adam_params + [{'params': model.hyperparameters()}],
            lr=0.01
        )
        optimizers = [hyperparameter_optimizer, variational_ngd_optimizer]
    else:
        raise NotImplementedError("Unrecognized variational distribution.")
    
    return optimizers

def batch_train_ordinal_likelihood_gp(
        models: List[ApproximateGPModelListModel], 
        all_classifier_ids: torch.Tensor,
        verbose=True
    ):
    if torch.cuda.is_available():
        for model in models:
            model = model.cuda()
            model.likelihood = model.likelihood.cuda()
            model.train_inputs = model.train_inputs.cuda()
            model.train_targets = model.train_targets.cuda()

    model = gpytorch.models.IndependentModelList(*models)
    likelihood = LikelihoodList(*[m.likelihood for m in models])

    mll = SumVariationalELBO(likelihood, model)

    # TODO check that num_data makes sense here
    optimizers = get_optimizer(model, likelihood)

    model.train()
    likelihood.train()
    losses = []

    model.compile(fullgraph=True, mode='reduce-overhead')
    likelihood.compile(fullgraph=True, mode='reduce-overhead')
    mll.compile(fullgraph=True, mode='reduce-overhead')

    training_iter = 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], int(training_iter / 10))
    best_loss = torch.tensor(float('inf'))
    num_since_best = 0
    num_iters_before_stopping = training_iter // 20
    min_loss_improvement = 0.0001

    with gpytorch.settings.cholesky_max_tries(10):
        pbar = tqdm(total=training_iter, desc='Training GP Model') if verbose else None
        for k in range(training_iter):
            for optimizer in optimizers:
                optimizer.zero_grad()
            with gpytorch.settings.variational_cholesky_jitter(1e-4):
                output = model(*model.train_inputs)
                loss = -mll(output, model.train_targets, classifier_ids=all_classifier_ids)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=1.0)
            for optimizer in optimizers:
                optimizer.step()

            if pbar is not None:
                pbar.update(1)
                pbar.set_description('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, loss.item()))
            
            scheduler.step(k)
            losses.append(loss.item())
            # checking since substantive decrease in loss
            if best_loss - loss.item() > min_loss_improvement:
                best_loss = loss.item()
                num_since_best = 0
            else:
                num_since_best += 1
            if num_since_best > num_iters_before_stopping:
                if pbar is not None:
                    pbar.close()
                break
        

    return model


def train_ordinal_likelihood_gp(model: GPClassificationModel, likelihood: BoundedOrdinalWithErrorLikelihood, train_X: torch.Tensor, train_y: torch.Tensor, classifier_ids: torch.Tensor, verbose=True):
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.size(0))
    return train_ordinal_likelihood_gp_with_mll(model, likelihood, mll, train_X, train_y, classifier_ids, verbose=verbose)

def train_ordinal_likelihood_gp_with_mll(
        model: GPClassificationModel, 
        likelihood: BoundedOrdinalWithErrorLikelihood, 
        mll: MarginalLogLikelihood,
        train_X: torch.Tensor, 
        train_y: torch.Tensor, 
        classifier_ids: torch.Tensor,
        verbose=True
    ):

    model.train()
    likelihood.train()
    losses = []
    optimizers = get_optimizer(model, likelihood, num_data=train_y.size(0))

    model.compile(fullgraph=True, mode='reduce-overhead')
    likelihood.compile(fullgraph=True, mode='reduce-overhead')
    mll.compile(fullgraph=True, mode='reduce-overhead')

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iter = 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], int(training_iter / 10))
    best_loss = torch.tensor(float('inf'))
    num_since_best = 0
    num_iters_before_stopping = training_iter // 20
    min_loss_improvement = 0.0001

    with gpytorch.settings.cholesky_max_tries(10):
        if train_X.size(0) < MAX_FULL_GP_BATCH:
            if torch.cuda.is_available():
                train_X = train_X.cuda()
                train_y = train_y.cuda()

            pbar = tqdm(total=training_iter, desc='Training GP Model') if verbose else None
            for k in range(training_iter):
                loss = optimize_step(optimizers, model, mll, likelihood, train_X, train_y, classifier_ids)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_description('Iter %d/%d - Loss: %.3f' % (k + 1, training_iter, loss.item()))
                
                scheduler.step(k)
                losses.append(loss.item())
                # checking since substantive decrease in loss
                if best_loss - loss.item() > min_loss_improvement:
                    best_loss = loss.item()
                    num_since_best = 0
                else:
                    num_since_best += 1
                if num_since_best > num_iters_before_stopping:
                    if pbar is not None:
                        pbar.close()
                    break
        else:
            train_dataset = TensorDataset(train_X, train_y, classifier_ids)
            train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)

            pbar = tqdm(total=training_iter * len(train_loader), desc='Training GP Model') if verbose else None
            for k in range(training_iter):
                epoch_losses = []
                for i, (batch_X, batch_y, batch_classifier_ids) in enumerate(train_loader):
                    if torch.cuda.is_available():
                        batch_X = batch_X.cuda()
                        batch_y = batch_y.cuda()
                        batch_classifier_ids = batch_classifier_ids.cuda()
                    loss = optimize_step(optimizers, model, mll, likelihood, batch_X, batch_y, batch_classifier_ids)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_description('Iter %d/%d - Loss: %.3f' % (k * len(train_loader) + i + 1, training_iter * len(train_loader), loss.item()))
                    epoch_losses.append(loss.item())

                epoch_loss = sum(epoch_losses) / len(epoch_losses)

                scheduler.step(k)

                losses.append(epoch_loss)
                # checking since substantive decrease in loss
                if best_loss - epoch_loss > min_loss_improvement:
                    best_loss = epoch_loss
                    num_since_best = 0
                else:
                    num_since_best += 1
                if num_since_best > num_iters_before_stopping:
                    if pbar is not None:
                        pbar.close()
                    break

    return model, likelihood, losses

def get_likelihood_prediction(model, likelihood, test_x):
    if not isinstance(test_x, torch.Tensor):
        test_x = torch.as_tensor(test_x).float()
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

def get_model_prediction(model: GPClassificationModel, test_x):
    test_x = torch.as_tensor(test_x).float()

    model.eval()
    model = model.cuda()
    test_x = test_x.cuda()

    with torch.no_grad():
        try:
            with gpytorch.settings.fast_pred_var():
                model_pred = model(test_x)
        except NotPSDError:
            model_pred = model(test_x)
                
    # Get upper and lower confidence bounds
    lower, upper = model_pred.confidence_region()

    return torch.tanh(model_pred.loc).cpu().numpy(), torch.tanh(lower).cpu().numpy(), torch.tanh(upper).cpu().numpy()

def get_batch_model_predictions(model: gpytorch.models.IndependentModelList, test_xs):
    test_xs = [torch.as_tensor(test_x).float() for test_x in test_xs]

    model.eval()
    model = model.cuda()
    test_xs = [test_x.cuda() for test_x in test_xs]

    with torch.no_grad():
        try:
            with gpytorch.settings.fast_pred_var():
                model_preds = model(*test_xs)
        except NotPSDError:
            model_preds = model(*test_xs)

    # Get upper and lower confidence bounds
    preds = []
    lowers = []
    uppers = []
    for model_pred in model_preds:
        preds.append(torch.tanh(model_pred.loc).cpu().numpy())
        lowers.append(torch.tanh(model_pred.confidence_region()[0]).cpu().numpy())
        uppers.append(torch.tanh(model_pred.confidence_region()[1]).cpu().numpy())

    return preds, lowers, uppers

def get_classifier_profiles(confusion_matrix=None):
    if confusion_matrix is None:
        file_path = huggingface_hub.hf_hub_download(repo_id='bendavidsteel/SmolLM2-360M-Instruct-stance-detection', filename='metadata.json')
        with open(file_path, 'r') as f:
            metadata = json.load(f)

        confusion_matrix = metadata['test_metrics']['test/confusion_matrix']


    classifier_profiles = {
        0: {
            'true_against': {
                'predicted_against': confusion_matrix[LABELS_2_ID['against']][LABELS_2_ID['against']],
                'predicted_neutral': confusion_matrix[LABELS_2_ID['against']][LABELS_2_ID['neutral']],
                'predicted_favor': confusion_matrix[LABELS_2_ID['against']][LABELS_2_ID['favor']]
            },
            'true_neutral': {
                'predicted_against': confusion_matrix[LABELS_2_ID['neutral']][LABELS_2_ID['against']],
                'predicted_neutral': confusion_matrix[LABELS_2_ID['neutral']][LABELS_2_ID['neutral']],
                'predicted_favor': confusion_matrix[LABELS_2_ID['neutral']][LABELS_2_ID['favor']]
            },
            'true_favor': {
                'predicted_favor': confusion_matrix[LABELS_2_ID['favor']][LABELS_2_ID['favor']],
                'predicted_neutral': confusion_matrix[LABELS_2_ID['favor']][LABELS_2_ID['neutral']],
                'predicted_against': confusion_matrix[LABELS_2_ID['favor']][LABELS_2_ID['against']]
            }
        }
    }
    return classifier_profiles

def get_timestamps(df: pl.DataFrame, start_date: datetime.datetime, time_column, time_scale):
    
    numerator_match = re.search('^\d', time_scale)
    assert numerator_match is not None, "time_scale must begin with an integer number of units"
    unit_match = re.search('[a-z]+$', time_scale)
    available_time_units = ['h', 'd', 'w', 'mo', 'y']
    assert unit_match is not None, f"time_scale must end with a unit from {available_time_units}"

    numerator = numerator_match.group(0)
    unit = unit_match.group(0)

    try:
        numerator = int(numerator)
    except:
        raise ValueError('time_scale argument must start with an integer')
    
    assert unit in available_time_units, f"time_scale unit must be in {available_time_units}"
    num_hours = numerator
    if unit == 'h':
        num_hours *= 1
    elif unit == 'd':
        num_hours *= 24
    elif unit == 'w':
        num_hours *= 24 * 7
    elif unit == 'mo':
        num_hours *= 24 * 30
    elif unit == 'y':
        num_hours *= 24 * 365

    return df.select(((pl.col(time_column) - start_date).dt.total_hours() / num_hours).alias('timestamps'))['timestamps'].to_numpy()


def get_time_series_data(filtered_df: pl.DataFrame, time_column, time_scale):
    sorted_df = filtered_df.sort(time_column)
    start_date = filtered_df[time_column].min().date()
    end_date = filtered_df[time_column].max().date()
    timestamps = get_timestamps(sorted_df, start_date, time_column, time_scale)

    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date)
        current_date += datetime.timedelta(days=1)
    day_df = pl.DataFrame({time_column: days})

    # Calculate volume
    day_df = day_df.join(
            sorted_df.select(pl.col(time_column).dt.date())\
                .group_by(time_column)\
                .len()\
                .rename({'len': 'volume'}),
            on=time_column,
            how='left'
        )\
        .fill_null(0)\
        .group_by(pl.col(time_column).dt.truncate('1d'))\
        .agg(pl.col('volume').sum())\
        .sort(time_column)

    test_x = get_timestamps(day_df, start_date, time_column, time_scale)
    
    stance = sorted_df['Stance'].to_numpy()

    classifier_ids = np.zeros_like(timestamps, dtype=np.uint16)

    return timestamps, stance, classifier_ids, test_x, day_df

def get_gp_timeseries(
        timestamps, 
        stance, 
        classifier_ids, 
        classifier_profiles, 
        test_x, 
        lengthscale_loc = 2.0, # mode at ~7.5 months
        lengthscale_scale = 0.1,
        sigma_loc = 1.0,
        sigma_scale = 0.2,
        verbose = False
    ):
    
    model, likelihood, train_x, train_y, classifier_ids = setup_ordinal_gp_model(
        timestamps, 
        stance, 
        classifier_ids, 
        classifier_profiles, 
        lengthscale_loc,
        lengthscale_scale,
        sigma_loc=sigma_loc,
        sigma_scale=sigma_scale
    )
    model, likelihood, losses = train_ordinal_likelihood_gp(model, likelihood, train_x, train_y, classifier_ids, verbose=verbose)
    pred, lower, upper = get_model_prediction(model, test_x)

    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    likelihood_sigma = likelihood.sigma.item()
    
    return lengthscale, likelihood_sigma, losses, pred, lower, upper

def combine_trend_df(trend_df: pl.DataFrame, pred, lower, upper, target_name, filter_type, filter_value):
    trend_df = trend_df.with_columns([
        pl.Series(name='trend_mean', values=pred),
        pl.Series(name='trend_lower', values=lower),
        pl.Series(name='trend_upper', values=upper)
    ])
    
    # Join trend and volume data in a single operation
    trend_df = trend_df.with_columns([
        pl.lit(target_name).alias('target'),
        pl.lit(filter_type).alias('filter_type'),
        pl.lit(str(filter_value)).alias('filter_value')
    ])
    return trend_df

def calculate_trends_for_filtered_df_with_batching(
        target_df: pl.DataFrame, 
        target_name: str, 
        filter_type: str, 
        classifier_profiles,
        lengthscale_loc: float,
        lengthscale_scale: float,
        sigma_loc: float,
        sigma_scale: float,
        time_column: str,
        time_scale: str,
        verbose: bool
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:

    assert isinstance(target_df.schema[time_column], pl.Datetime), f"Column {time_column} must be of type DateTime"

    batch_interpolation_outputs = []
    all_trends_df = None

    unique_df = target_df.group_by(filter_type)\
        .len()
    
    batched_values = unique_df.sort('len')\
        .with_columns((pl.col('len').cum_sum() / MAX_FULL_GP_BATCH).floor().alias('group'))\
        .group_by('group')\
        .agg(pl.col(filter_type))[filter_type].to_list()
    
    if verbose:
        pbar = tqdm(total=unique_df.shape[0], desc='Training GPs')
    for batch in batched_values:
        try:
            batch_trend_df, batch_interpolation_outputs_batch = batch_calculate_trends_for_filtered_df(
                target_df, 
                target_name, 
                filter_type, 
                batch, 
                classifier_profiles,
                lengthscale_loc,
                lengthscale_scale,
                sigma_loc,
                sigma_scale,
                time_column,
                time_scale,
                False
            )
        except Exception as ex:
            logger.error(f"Failed to calculate trends for {target_name} {filter_type} {batch}: {ex}")
            continue
        if verbose:
            pbar.update(len(batch))
        if batch_trend_df is not None:
            if all_trends_df is None:
                all_trends_df = batch_trend_df
            else:
                all_trends_df = pl.concat([all_trends_df, batch_trend_df])
            batch_interpolation_outputs.extend(batch_interpolation_outputs_batch)

    return all_trends_df, batch_interpolation_outputs

def batch_calculate_trends_for_filtered_df(
        target_df: pl.DataFrame, 
        target_name: str, 
        filter_type: str, 
        unique_values: List[str], 
        classifier_profiles,
        lengthscale_loc: float,
        lengthscale_scale: float,
        sigma_loc: float,
        sigma_scale: float,
        time_column: str,
        time_scale: str,
        verbose: bool
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:

    batch_interpolation_outputs = []
    all_trends_df = None

    all_timestamps, all_stance, all_classifier_ids, all_test_x, all_day_dfs = [], [], [], [], []
    for unique_value in unique_values:
        filtered_df = target_df.filter(pl.col(filter_type) == unique_value)

        timestamps, stance, classifier_ids, test_x, day_df = get_time_series_data(filtered_df, time_column, time_scale)
        all_timestamps.append(timestamps)
        all_stance.append(stance)
        all_classifier_ids.append(classifier_ids)
        all_test_x.append(test_x)
        all_day_dfs.append(day_df)

    models = []
    batch_classifier_ids = []
    for i in range(len(all_timestamps)):
        timestamps = all_timestamps[i]
        stance = all_stance[i]
        classifier_ids = all_classifier_ids[i]

        model, classifier_ids = setup_batchable_ordinal_gp_model(
            timestamps, 
            stance, 
            classifier_ids, 
            classifier_profiles, 
            lengthscale_loc,
            lengthscale_scale,
            sigma_loc=sigma_loc,
            sigma_scale=sigma_scale
        )
        models.append(model)
        batch_classifier_ids.append(classifier_ids)

    model = batch_train_ordinal_likelihood_gp(models, batch_classifier_ids, verbose=verbose)
    preds, lowers, uppers = get_batch_model_predictions(model, all_test_x)

    

    for filter_value, day_df, model, pred, lower, upper in zip(unique_values, all_day_dfs, models, preds, lowers, uppers):
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        likelihood_sigma = model.likelihood.sigma.item()
        trend_df = combine_trend_df(day_df, pred, lower, upper, target_name, filter_type, filter_value)
        if all_trends_df is None:
            all_trends_df = trend_df
        else:
            all_trends_df = pl.concat([all_trends_df, trend_df])
        interpolation_outputs = {
            'lengthscale': lengthscale,
            'sigma': likelihood_sigma,
            'loss': None, # cannot retrieve individual losses from batch training
            'target_name': target_name,
            'filter_type': filter_type,
            'filter_value': filter_value
        }
        batch_interpolation_outputs.append(interpolation_outputs)
    return all_trends_df, batch_interpolation_outputs


def calculate_trends_for_filtered_df(
        filtered_df: pl.DataFrame, 
        target_name, 
        filter_type, 
        filter_value, 
        classifier_profiles,
        time_column,
        time_scale,
        interpolation_method='gp',
        lengthscale_loc = 2.0, # mode at ~7.5 months
        lengthscale_scale = 0.1,
        sigma_loc = 1.0,
        sigma_scale = 0.2,
        verbose=False
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Calculate trends for a filtered DataFrame with optimized operations"""
    # First sort by createtime - ensures consistent results
    start_date = filtered_df[time_column].min().date()
    end_date = filtered_df[time_column].max().date()

    if end_date - start_date < datetime.timedelta(days=1):
        # TODO implement static distribution calculation
        # return calculate_static_dist_for_filtered_df(filtered_df, target_name, filter_type, filter_value, trend_path, classifier_profiles)
        logger.warning(f"Skipping {target_name} {filter_type} {filter_value} - not enough data")
        return None, None

    timestamps, stance, classifier_ids, test_x, trend_df = get_time_series_data(filtered_df, time_column, time_scale)
    
    if interpolation_method == 'gp':
        try:
            lengthscale, likelihood_sigma, losses, pred, lower, upper = get_gp_timeseries(timestamps, stance, classifier_ids, classifier_profiles, test_x, lengthscale_loc, lengthscale_scale, sigma_loc, sigma_scale, verbose)
        except Exception as ex:
            logger.error(f"Failed for target {target_name} filter_type {filter_type} filter_value {filter_value} ex: {ex}")
            return None, None
        
        trend_df = combine_trend_df(trend_df, pred, lower, upper, target_name, filter_type, filter_value)

        # write data to this
        interpolation_outputs = {
            'lengthscale': lengthscale,
            'sigma': likelihood_sigma,
            'loss': losses[-1],
            'target_name': target_name,
            'filter_type': filter_type,
            'filter_value': filter_value
        }
    else:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        pred = lowess(stance, timestamps, xvals=test_x, is_sorted=True, return_sorted=False)

        trend_df = combine_trend_df(trend_df, pred, np.full_like(pred, np.nan), np.full_like(pred, np.nan), target_name, filter_type, filter_value)
        interpolation_outputs = {}

    return trend_df, interpolation_outputs

def compute_trends_for_target(
        df: pl.DataFrame, 
        target_name, 
        filter_columns: List[str], 
        time_column,
        interpolation_method='gp',
        classifier_profiles=None,
        min_filter_count=5,
        time_scale='1mo',
        verbose=False,
        lengthscale_loc = 2.0, # mode at ~7.5 months
        lengthscale_scale = 0.1,
        sigma_loc = 1.0,
        sigma_scale = 0.2
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
    """Precompute trend data for a specific target with optimized vectorized operations"""
    # Get target data in one operation
    target_df = df.filter(pl.col('Target') == target_name)
    
    logger.info(f"Processing target {target_name}: {target_df.shape[0]} points")
    
    # Calculate all trends
    
    all_interpolation_outputs = []
    all_trend_df = None    

    # For each filter type, calculate trends for each unique value
    for filter_column in filter_columns:
        # Get all unique values for this filter
        all_unique_value_df = target_df.group_by(filter_column).len().filter(pl.col('len') >= min_filter_count)

        if interpolation_method == 'gp':
            if all_unique_value_df.filter(pl.col('len') < MAX_FULL_GP_BATCH).shape[0] > 1:
                batch_unique_values = all_unique_value_df.filter(pl.col('len') < MAX_FULL_GP_BATCH)[filter_column].to_list()
                sequential_unique_values = all_unique_value_df.filter(pl.col('len') >= MAX_FULL_GP_BATCH)[filter_column].to_list()
            else:
                batch_unique_values = []
                sequential_unique_values = all_unique_value_df[filter_column].to_list()
        else:
            # For non-GP methods, we can process all unique values sequentially
            batch_unique_values = []
            sequential_unique_values = all_unique_value_df[filter_column].to_list()
            
        # Apply filtering and trend calculation for each value
        if len(batch_unique_values) > 0:
            # compute in parallel
            filtered_df = target_df.filter(pl.col(filter_column).is_in(batch_unique_values))
            batch_trend_df, batch_interpolation_outputs = calculate_trends_for_filtered_df_with_batching(
                filtered_df, 
                target_name, 
                filter_column, 
                classifier_profiles,
                lengthscale_loc,
                lengthscale_scale,
                sigma_loc,
                sigma_scale,
                time_column,
                time_scale,
                verbose
            )
            all_interpolation_outputs += batch_interpolation_outputs
            if batch_trend_df is not None:
                if all_trend_df is None:
                    all_trend_df = batch_trend_df
                else:
                    # Concatenate the new batch DataFrame to the existing one
                    all_trend_df = pl.concat([all_trend_df, batch_trend_df])
        
        if len(sequential_unique_values) > 0:
            for filter_value in sequential_unique_values:
                filtered_df = target_df.filter(pl.col(filter_column) == filter_value)
                try:
                    trend_df, interpolation_outputs = calculate_trends_for_filtered_df(
                        filtered_df, 
                        target_name, 
                        filter_column, 
                        filter_value, 
                        classifier_profiles,
                        time_column,
                        time_scale,
                        interpolation_method=interpolation_method,
                        lengthscale_loc=lengthscale_loc,
                        lengthscale_scale=lengthscale_scale,
                        sigma_loc=sigma_loc,
                        sigma_scale=sigma_scale,
                        verbose=verbose
                    )
                except Exception as ex:
                    logger.error(f"Failed to calculate trends for {target_name} {filter_column} {filter_value}: {ex}")
                    continue
                
                if interpolation_outputs is None:
                    continue

                if all_trend_df is None:
                    all_trend_df = trend_df
                else:
                    all_trend_df = pl.concat([all_trend_df, trend_df])
                all_interpolation_outputs.append(interpolation_outputs)

                logger.info(f"Processed {filter_column} {filter_value}: {len(filtered_df)} points")
        

    # First, the overall trend
    trend_df, interpolation_outputs = calculate_trends_for_filtered_df(
        target_df, 
        target_name, 
        'all', 
        'all', 
        classifier_profiles,
        time_column,
        time_scale,
        lengthscale_loc=lengthscale_loc,
        lengthscale_scale=lengthscale_scale,
        sigma_loc=sigma_loc,
        sigma_scale=sigma_scale,
        verbose=verbose
    )
    if interpolation_outputs is not None:
        all_interpolation_outputs.append(interpolation_outputs)
    if trend_df is not None:
        if all_trend_df is None:
            all_trend_df = trend_df
        else:
            all_trend_df = pl.concat([all_trend_df, trend_df])

    return all_trend_df, all_interpolation_outputs

def get_stance_trends(
        document_df: pl.DataFrame, 
        time_column: str = None, 
        filter_columns: List[str] = [], 
        min_count: int = 5, 
        time_scale: str = '1mo',
        interpolation_method: str = 'gp',
        verbose: bool = False
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
    all_trend_gps_data = []
    all_trend_df = None

    targets_df = document_df.explode(['Targets', 'Stances']).rename({'Targets': 'Target', 'Stances': 'Stance'})

    target_names = targets_df.group_by('Target')\
        .len()\
        .filter(pl.col('len') > min_count)\
        .drop_nulls('Target')\
        .sort('len', descending=True)['Target'].to_list()

    if interpolation_method == 'gp':
        classifier_profiles = get_classifier_profiles()
    else:
        classifier_profiles = None

    for target_name in target_names:
        logger.info(f"Processing primary target: {target_name}")
            
        # Process the target with grouping
        target_trend_df, interpolation_outputs = compute_trends_for_target(
            targets_df, 
            target_name, 
            filter_columns, 
            time_column,
            interpolation_method=interpolation_method,
            classifier_profiles=classifier_profiles,
            min_filter_count=min_count,
            time_scale=time_scale,
            verbose=verbose
        )
        all_trend_gps_data += interpolation_outputs
        if target_trend_df is not None:
            if all_trend_df is None:
                all_trend_df = target_trend_df
            else:
                all_trend_df = pl.concat([all_trend_df, target_trend_df])


    gp_df = pl.DataFrame(all_trend_gps_data)
    return all_trend_df, gp_df

