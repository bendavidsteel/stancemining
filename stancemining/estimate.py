import datetime
import json
import re
from typing import Optional, Any, Dict, List, Tuple, Iterable, Union, Callable

import huggingface_hub
import numpy as np
import polars as pl

from tqdm import tqdm

from stancemining.finetune import STANCE_LABELS_2_ID
from stancemining.main import logger

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

try:
    import gpytorch
    from gpytorch.priors import Prior, NormalPrior
    from gpytorch.constraints import Interval, Positive
    from gpytorch.likelihoods.likelihood_list import _get_tuple_args_, LikelihoodList
    from gpytorch.mlls import VariationalELBO, MarginalLogLikelihood
    from gpytorch.utils.generic import length_safe_zip
    import pyro
    import pyro.distributions
    import pyro.infer
    import pyro.poutine
    from linear_operator.utils.errors import NotPSDError
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

MAX_FULL_GP_BATCH = 10000

if GP_AVAILABLE:
    class StanceEstimation:
        def __init__(self, all_classifier_profiles):
            self.predictor_confusion_probs = _get_target_predictor_confusion_probs(all_classifier_profiles)
            
        def model(self, opinion_sequences, predictor_ids, mask, prior=False):
            
            with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
                if prior:
                    user_stance_loc = pyro.param("user_stance_loc", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=pyro.distributions.constraints.interval(-1, 1))
                    user_stance_loc_var = pyro.param("user_stance_loc_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=pyro.distributions.constraints.positive)
                    user_stance = pyro.sample("user_stance", pyro.distributions.Normal(user_stance_loc, user_stance_loc_var))
                    user_stance_var = pyro.sample(
                        "user_stance_var", 
                        pyro.distributions.LogNormal(
                            torch.full(((opinion_sequences.shape[0], 1)), 0.1).to(mask.device), 
                            torch.full(((opinion_sequences.shape[0], 1)), 0.2).to(mask.device)
                        )
                    )
                    
                else:
                    user_stance_var = pyro.param("user_stance_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=pyro.distributions.constraints.positive)
                    # # sample stance from the uniform prior
                    user_stance = pyro.param("user_stance", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=pyro.distributions.constraints.interval(-1, 1))

                # loop over the observed data
                with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                    with pyro.poutine.mask(mask=mask):
                        # User creates comments with latent stance
                        # Standard deviation should be half the distance between categories
                        # comment_var = (1/2) ** 2
                        
                        comment_stances = pyro.sample("latent_comment_stance", pyro.distributions.Normal(user_stance, user_stance_var).expand(opinion_sequences.shape))

                        # Quantize comment stance into 3 categories
                        comment_stance_cats = torch.zeros_like(comment_stances, dtype=torch.int).to(opinion_sequences.device)
                        comment_stance_cats[comment_stances > 1/2] = 2
                        comment_stance_cats[comment_stances < -1/2] = 0
                        comment_stance_cats[(comment_stances >= -1/2) & (comment_stances <= 1/2)] = 1

                        # Get prediction probabilities based on the confusion matrix
                        # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                        predict_probs = self.predictor_confusion_probs['predict_probs'][predictor_ids, comment_stance_cats, :]

                        pyro.sample("predicted_comment_stance", pyro.distributions.Categorical(probs=predict_probs), obs=opinion_sequences)

        def guide(self, opinion_sequences, predictor_ids, mask, prior=False):
            # comment_stance_var_loc = pyro.param("comment_stance_var_loc", torch.tensor(0.1))
            # comment_stance_var_scale = pyro.param("comment_stance_var_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
            # comment_stance_var = pyro.sample("comment_stance_var", dist.LogNormal(comment_stance_var_loc, comment_stance_var_scale), infer={'is_auxiliary': True})
            # # sample stance from the uniform prior
            
            # loop over the observed data
            with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
                if prior:
                    user_stance_q = pyro.param("user_stance_loc_q", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=pyro.distributions.constraints.interval(-1, 1))
                    user_stance_var_q = pyro.param("user_stance_var_q", torch.tensor(0.001).unsqueeze(0).tile((opinion_sequences.shape[0],1)).to(mask.device), constraint=pyro.distributions.constraints.positive)
                    user_stance = pyro.sample("user_stance", pyro.distributions.Delta(user_stance_q))
                    user_stance_var = pyro.sample("user_stance_var", pyro.distributions.Delta(user_stance_var_q))
                with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                    with pyro.poutine.mask(mask=mask):

                        # Get true probabilities from the confusion matrix
                        true_probs = self.predictor_confusion_probs['true_probs'][predictor_ids, opinion_sequences, :]

                        # Sample latent comment stance categories
                        comment_stance_cats = pyro.sample("latent_comment_stance_category", pyro.distributions.Categorical(probs=true_probs), infer={'is_auxiliary': True})

                        # Determine latent locations based on categories
                        latent_locs = torch.zeros_like(comment_stance_cats, dtype=torch.float).to(opinion_sequences.device)
                        latent_locs[comment_stance_cats == 1] = 0
                        latent_locs[comment_stance_cats == 2] = 1
                        latent_locs[comment_stance_cats == 0] = -1

                        # Sample latent comment stances
                        # comment_stances = pyro.sample("latent_comment_stance", dist.Normal(latent_locs, comment_var))
                        comment_stances = pyro.sample("latent_comment_stance", pyro.distributions.Delta(latent_locs))

    def _train_pyro_model(model_func, guide_func, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps):
        pyro.clear_param_store()
        # do gradient steps
        svi = pyro.infer.SVI(model_func, guide_func, optim, loss=pyro.infer.Trace_ELBO())
        losses = []
        for step in range(num_steps):
            loss = svi.step(stance_opinion_sequences, classifier_indices, mask, prior=prior)
            losses.append(loss)

    def _get_op_seqs(opinion_sequences, all_classifier_indices):
        op_seqs = []
        lengths = []
        all_predictor_ids = []
        for user_idx in range(opinion_sequences.shape[0]):
            seq = []
            length = 0
            predictor_ids = []
            for i in range(opinion_sequences.shape[1]):
                if not np.isnan(opinion_sequences[user_idx, i]):
                    seq.append(opinion_sequences[user_idx, i])
                    length += 1
                    predictor_ids.append(all_classifier_indices[user_idx, i].astype(int))

            op_seqs.append(np.array(seq))
            lengths.append(length)
            all_predictor_ids.append(np.array(predictor_ids))

        max_length = max(lengths)

        return op_seqs, lengths, all_predictor_ids, max_length

    def _get_op_seq_with_mask(opinion_sequences, max_length, op_seqs, lengths, all_predictor_ids):
        stance_opinion_sequences = np.zeros((opinion_sequences.shape[0], max_length))
        classifier_indices = np.zeros((opinion_sequences.shape[0], max_length))
        mask = np.zeros((opinion_sequences.shape[0], max_length))
        for i in range(opinion_sequences.shape[0]):
            stance_opinion_sequences[i, :lengths[i]] = op_seqs[i]
            classifier_indices[i, :lengths[i]] = all_predictor_ids[i]
            mask[i, :lengths[i]] = 1

        return stance_opinion_sequences, classifier_indices, mask

    def _data_to_torch_device(stance_opinion_sequences, classifier_indices, mask, device):
        stance_opinion_sequences = torch.tensor(stance_opinion_sequences).int() + 1
        classifier_indices = torch.tensor(classifier_indices).int()
        mask = torch.tensor(mask).bool()

        stance_opinion_sequences = stance_opinion_sequences.to(device)
        classifier_indices = classifier_indices.to(device)
        mask = mask.to(device)

        return stance_opinion_sequences, classifier_indices, mask

    def _get_pyro_optimizer(num_steps):
        
        gamma = 0.1  # final learning rate will be gamma * initial_lr
        initial_lr = 0.01
        lrd = gamma ** (1 / num_steps)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
        return optim

    def get_stance_normal(stance_df):
        classifier_profiles = _get_classifier_profiles()
        estimator = StanceEstimation(classifier_profiles)

        num_steps = 1000
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optim = _get_pyro_optimizer(num_steps)
        user_stances = np.zeros(1)
        user_stance_vars = np.zeros(1)

        stance_sequence = stance_df['Stance'].to_numpy().reshape(1, -1)
        all_classifier_indices = np.zeros_like(stance_sequence)

        op_seqs, lengths, all_predictor_ids, max_length = _get_op_seqs(stance_sequence, all_classifier_indices)

        stance_opinion_sequences, classifier_indices, mask = _get_op_seq_with_mask(stance_sequence, max_length, op_seqs, lengths, all_predictor_ids)
        
        stance_opinion_sequences, classifier_indices, mask = _data_to_torch_device(stance_opinion_sequences, classifier_indices, mask, device)

        estimator.predictor_confusion_probs['predict_probs'] = estimator.predictor_confusion_probs['predict_probs'].to(device)
        estimator.predictor_confusion_probs['true_probs'] = estimator.predictor_confusion_probs['true_probs'].to(device)

        prior = True
        _train_pyro_model(estimator.model, estimator.guide, optim, stance_opinion_sequences, classifier_indices, mask, prior, num_steps)

        # grab the learned variational parameters
        if prior:
            user_stances[:] = pyro.param("user_stance_loc_q").cpu().detach().numpy().squeeze(-1)
            user_stance_vars[:] = pyro.param("user_stance_var_q").cpu().detach().numpy().squeeze(-1)
        else:
            user_stances[:] = pyro.param("user_stance").detach().numpy().squeeze(-1)
            user_stance_vars[:, 1] = pyro.param("user_stance_var").detach().numpy().squeeze(-1)
        
        return user_stances.item(), user_stance_vars.item()

    def infer_stance_normal_for_target(target_df: pl.DataFrame, min_count: int = 5, filter_cols: List[str] = [], verbose: bool = False) -> pl.DataFrame:
        """Get the stance normal for a specific target DataFrame.

        Args:
            target_df (pl.DataFrame): DataFrame containing stance data for a specific target.
            min_count (int): Minimum number of samples required to calculate stance.
            filter_cols (List[str]): List of columns to filter by unique values.
            verbose (bool): If True, print progress messages.

        Returns:
            pl.DataFrame: DataFrame containing the stance normal for the target.
        """
        stance_values = []

        for filter_col in filter_cols:
            for unique_value in tqdm(target_df[filter_col].unique(), disable=not verbose, desc=f"Getting stance for unique values in '{filter_col}'"):
                value_df = target_df.filter(pl.col(filter_col) == unique_value)
                if value_df.shape[0] < min_count:
                    continue
                value_stance_loc, value_stance_var = get_stance_normal(value_df)
                stance_values.append({
                    'Target': target_df['Target'].unique()[0],
                    'filter_type': filter_col,
                    'filter_value': unique_value,
                    'stance_loc': value_stance_loc,
                    'stance_var': value_stance_var
                })

        all_stance_loc, all_stance_var = get_stance_normal(target_df)
        stance_values.append({
            'Target': target_df['Target'].unique()[0],
            'filter_type': 'all',
            'filter_value': 'all',
            'stance_loc': all_stance_loc,
            'stance_var': all_stance_var
        })

        return pl.DataFrame(stance_values)

    def infer_stance_normal_for_all_targets(document_df: pl.DataFrame, filter_cols: List[str] = [], min_count: int = 5, verbose: bool = False) -> pl.DataFrame:
        """Get the stance normal for all targets in the document DataFrame.

        Args:
            document_df (pl.DataFrame): DataFrame containing document data with stance information.
            filter_cols (List[str]): List of columns to filter by unique values.
            min_count (int): Minimum number of samples required to calculate stance.
            verbose (bool): If True, print progress messages.

        Returns:
            pl.DataFrame: DataFrame containing the stance normal for all targets.
        """
        targets_df, target_names = _document_to_targets(document_df, min_count)

        all_value_stance_df = None
        for target_name in target_names:
            target_df = targets_df.filter(pl.col('Target') == target_name)
            if target_df.shape[0] < min_count:
                logger.warning(f"Skipping target '{target_name}' with {target_df.shape[0]} samples, less than min_count {min_count}.")
                continue
            if verbose:
                logger.info(f"Calculating stance for target '{target_name}' with {target_df.shape[0]} samples.")
            target_value_stance_df = infer_stance_normal_for_target(target_df, min_count=min_count, filter_cols=filter_cols, verbose=verbose)

            if all_value_stance_df is None:
                all_value_stance_df = target_value_stance_df
            else:
                all_value_stance_df = pl.concat([all_value_stance_df, target_value_stance_df])

        return all_value_stance_df

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
    def _inv_probit(x, jitter: float = 1e-3):
        """
        Inverse probit function (standard normal CDF) with jitter for numerical stability.
        
        Args:
            x: Input tensor
            jitter: Small constant to ensure outputs are strictly between 0 and 1
            
        Returns:
            Probabilities between jitter and 1-jitter
        """
        return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2 * jitter) + jitter

    def _get_predict_probs(true_cat, confusion_profile):
        predict_sum = sum(v for k, v in confusion_profile[true_cat].items())
        if predict_sum == 0:
            return torch.tensor([1/3, 1/3, 1/3])  # Uniform distribution if no predictions
        predict_probs = torch.tensor([
            confusion_profile[true_cat]["predicted_against"] / predict_sum,
            confusion_profile[true_cat]["predicted_neutral"] / predict_sum,
            confusion_profile[true_cat]["predicted_favor"] / predict_sum,
        ])
        return predict_probs

    def _get_true_probs(predicted_cat, confusion_profile):
        true_sum = sum(v[predicted_cat] for k, v in confusion_profile.items())
        if true_sum == 0:
            return torch.tensor([1/3, 1/3, 1/3])
        true_probs = torch.tensor([
            confusion_profile["true_against"][predicted_cat] / true_sum,
            confusion_profile["true_neutral"][predicted_cat] / true_sum,
            confusion_profile["true_favor"][predicted_cat] / true_sum,
        ])
        return true_probs

    def _get_target_predictor_confusion_probs(classifier_profiles):
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
                predict_probs[predictor_id, true_idx, :] = _get_predict_probs(true_cat, confusion_profile)

            for predicted_idx, predicted_cat in enumerate(["predicted_against", "predicted_neutral", "predicted_favor"]):
                true_probs[predictor_id, predicted_idx, :] = _get_true_probs(predicted_cat, confusion_profile)

        return {
            'predict_probs': predict_probs,
            'true_probs': true_probs,
        }

    @torch.jit.script
    def _get_ordinal_probs(function_samples, bin_edges, sigma, predictor_predict_probs):
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
        probs = _inv_probit(scaled_edges_left - scaled_function_samples) - _inv_probit(scaled_edges_right - scaled_function_samples)
        
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
            predictor_predict_probs = _get_target_predictor_confusion_probs(classifier_profiles)['predict_probs']
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

            predict_probs = _get_ordinal_probs(function_samples, self.bin_edges, self.sigma, predictor_predict_probs)

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

    def _get_inducing_points(train_x):
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

    def _get_ordinal_gp_model(train_x, train_y, classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, sigma_loc=1.0, sigma_scale=0.5):
        bin_edges = torch.tensor([-0.5, 0.5])
        # likelihood = OrdinalLikelihood(bin_edges)
        sigma_prior = NormalPrior(sigma_loc, sigma_scale)
        likelihood = BoundedOrdinalWithErrorLikelihood(bin_edges, classifier_profiles, sigma_prior=sigma_prior)
        
        inducing_points, learn_inducing_locations = _get_inducing_points(train_x)

        model = GPClassificationModel(
            inducing_points, 
            learn_inducing_locations=learn_inducing_locations, 
            lengthscale_loc=lengthscale_loc, 
            lengthscale_scale=lengthscale_scale,
            variational_dist='natural'
        )
        return model, likelihood

    def _inputs_np_to_torch(timestamps, stance, classifier_ids):
        timestamps = torch.as_tensor(timestamps, dtype=torch.float32)
        stance = torch.as_tensor(stance, dtype=torch.float32) + 1
        classifier_ids = torch.as_tensor(classifier_ids, dtype=torch.int)
        return timestamps, stance, classifier_ids

    def _setup_ordinal_gp_model(timestamps, stance, classifier_ids, classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc=1.0, sigma_scale=0.1):
        timestamps, stance, classifier_ids = _inputs_np_to_torch(timestamps, stance, classifier_ids)

        model, likelihood = _get_ordinal_gp_model(timestamps, stance, classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, sigma_loc=sigma_loc, sigma_scale=sigma_scale)

        return model, likelihood, timestamps, stance, classifier_ids

    def _get_batchable_ordinal_gp_model(train_x, train_y, classifier_profiles, lengthscale_loc=1.0, lengthscale_scale=0.5, sigma_loc=1.0, sigma_scale=0.5):
        bin_edges = torch.tensor([-0.5, 0.5])
        # likelihood = OrdinalLikelihood(bin_edges)
        sigma_prior = NormalPrior(sigma_loc, sigma_scale)
        likelihood = BoundedOrdinalWithErrorLikelihood(bin_edges, classifier_profiles, sigma_prior=sigma_prior)
        
        inducing_points, learn_inducing_locations = _get_inducing_points(train_x)

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

    def _setup_batchable_ordinal_gp_model(timestamps, stance, classifier_ids, classifier_profiles, lengthscale_loc, lengthscale_scale, sigma_loc=1.0, sigma_scale=0.1):
        timestamps, stance, classifier_ids = _inputs_np_to_torch(timestamps, stance, classifier_ids)

        model = _get_batchable_ordinal_gp_model(timestamps, stance, classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, sigma_loc=sigma_loc, sigma_scale=sigma_scale)
        
        return model, classifier_ids

    def _optimize_step(optimizers, model, mll, likelihood, batch_X, batch_y, batch_classifier_ids):
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

    def _get_optimizer(model, likelihood, num_data=None):
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

    def _batch_train_ordinal_likelihood_gp(
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
        optimizers = _get_optimizer(model, likelihood)

        model.train()
        likelihood.train()
        losses = []

        model.compile(fullgraph=True, mode='reduce-overhead')
        likelihood.compile(fullgraph=True, mode='reduce-overhead')
        mll.compile(fullgraph=True, mode='reduce-overhead')

        training_iter = 5000
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[0], int(training_iter / 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=training_iter)
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


    def _train_ordinal_likelihood_gp(model: GPClassificationModel, likelihood: BoundedOrdinalWithErrorLikelihood, train_X: torch.Tensor, train_y: torch.Tensor, classifier_ids: torch.Tensor, verbose=True):
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.size(0))
        return _train_ordinal_likelihood_gp_with_mll(model, likelihood, mll, train_X, train_y, classifier_ids, verbose=verbose)

    def _train_ordinal_likelihood_gp_with_mll(
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
        optimizers = _get_optimizer(model, likelihood, num_data=train_y.size(0))

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
                    loss = _optimize_step(optimizers, model, mll, likelihood, train_X, train_y, classifier_ids)

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
                        loss = _optimize_step(optimizers, model, mll, likelihood, batch_X, batch_y, batch_classifier_ids)
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

    def _get_likelihood_prediction(model, likelihood, test_x):
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

    def _get_model_prediction(model: GPClassificationModel, test_x):
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

    def _get_batch_model_predictions(model: gpytorch.models.IndependentModelList, test_xs):
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

    def _get_classifier_profiles(
            model_name: str = 'bendavidsteel/SmolLM2-360M-Instruct-stance-detection', 
            confusion_matrix=None
        ):
        if confusion_matrix is None:
            file_path = huggingface_hub.hf_hub_download(repo_id=model_name, filename='metadata.json')
            with open(file_path, 'r') as f:
                metadata = json.load(f)

            confusion_matrix = metadata['test_metrics']['test/confusion_matrix']


        classifier_profiles = {
            0: {
                'true_against': {
                    'predicted_against': confusion_matrix[STANCE_LABELS_2_ID['against']][STANCE_LABELS_2_ID['against']],
                    'predicted_neutral': confusion_matrix[STANCE_LABELS_2_ID['against']][STANCE_LABELS_2_ID['neutral']],
                    'predicted_favor': confusion_matrix[STANCE_LABELS_2_ID['against']][STANCE_LABELS_2_ID['favor']]
                },
                'true_neutral': {
                    'predicted_against': confusion_matrix[STANCE_LABELS_2_ID['neutral']][STANCE_LABELS_2_ID['against']],
                    'predicted_neutral': confusion_matrix[STANCE_LABELS_2_ID['neutral']][STANCE_LABELS_2_ID['neutral']],
                    'predicted_favor': confusion_matrix[STANCE_LABELS_2_ID['neutral']][STANCE_LABELS_2_ID['favor']]
                },
                'true_favor': {
                    'predicted_favor': confusion_matrix[STANCE_LABELS_2_ID['favor']][STANCE_LABELS_2_ID['favor']],
                    'predicted_neutral': confusion_matrix[STANCE_LABELS_2_ID['favor']][STANCE_LABELS_2_ID['neutral']],
                    'predicted_against': confusion_matrix[STANCE_LABELS_2_ID['favor']][STANCE_LABELS_2_ID['against']]
                }
            }
        }
        return classifier_profiles


    def _get_gp_timeseries(
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
        
        model, likelihood, train_x, train_y, classifier_ids = _setup_ordinal_gp_model(
            timestamps, 
            stance, 
            classifier_ids, 
            classifier_profiles, 
            lengthscale_loc,
            lengthscale_scale,
            sigma_loc=sigma_loc,
            sigma_scale=sigma_scale
        )
        model, likelihood, losses = _train_ordinal_likelihood_gp(model, likelihood, train_x, train_y, classifier_ids, verbose=verbose)
        pred, lower, upper = _get_model_prediction(model, test_x)

        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        likelihood_sigma = likelihood.sigma.item()
        
        return lengthscale, likelihood_sigma, losses, pred, lower, upper

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _est_loc_linear_numba(bw, endog, exog, data_predict):
        """
        Numba-optimized local linear estimation for k_vars=1.
        Uses analytical 2x2 pseudo-inverse instead of np.linalg.pinv.
        """
        n_bootstrap = endog.shape[0]
        nobs = endog.shape[1]
        N_predict = data_predict.shape[0]
        h = bw[0]
        h2 = h * h * 2.0
        norm_factor = 1.0 / (np.sqrt(2 * np.pi) * h * nobs)

        means = np.zeros((n_bootstrap, N_predict))

        for b in prange(n_bootstrap):
            for p in range(N_predict):
                x_p = data_predict[p, 0]

                # Compute kernel weights and build M matrix and V vector
                M00 = 0.0
                M01 = 0.0
                M11 = 0.0
                V0 = 0.0
                V1 = 0.0

                for i in range(nobs):
                    xi = exog[b, i, 0]
                    diff = x_p - xi
                    ker = np.exp(-diff * diff / h2) * norm_factor

                    y_i = endog[b, i, 0]
                    ker_y = ker * y_i

                    M00 += ker
                    M01 += ker * (-diff)
                    M11 += ker * diff * diff
                    V0 += ker_y
                    V1 += ker_y * (-diff)

                # Solve 2x2 system using pseudo-inverse formula
                # M = [[M00, M01], [M01, M11]]
                # For symmetric 2x2: pinv = adj(M) / det(M) when det != 0
                det = M00 * M11 - M01 * M01

                if abs(det) > 1e-10:
                    # mean = (M11 * V0 - M01 * V1) / det
                    means[b, p] = (M11 * V0 - M01 * V1) / det
                else:
                    # Fallback for singular matrix
                    if abs(M00) > 1e-10:
                        means[b, p] = V0 / M00
                    else:
                        means[b, p] = 0.0

        return means


def gaussian(h, Xi, x):
    """
    Vectorized gaussian kernel.
    Xi: shape (n_bootstrap, nobs, k_vars)
    x: shape (N_predict, k_vars)
    Returns: shape (n_bootstrap, N_predict, nobs)
    """
    # Reshape for broadcasting: (N_predict, 1, k_vars) - (1, nobs, k_vars)
    diff = x[np.newaxis, :, np.newaxis, :] - Xi[:, np.newaxis, ...]
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-np.sum(diff**2, axis=3) / (h**2 * 2.))

def _est_loc_linear(bw, endog, exog, data_predict):
    """
    Fully vectorized local linear estimation for multiple prediction points.
    data_predict: shape (N_predict, k_vars)
    Returns: mean (n_bootstrap, N_predict,), mfx (n_bootstrap, N_predict, k_vars)
    """
    n_bootstrap, nobs, k_vars = exog.shape
    N_predict = data_predict.shape[0]
    
    # Compute kernels for all prediction points at once
    # ker shape: (N_predict, nobs)
    ker = gaussian(bw[0], exog, data_predict) / (bw[0] * float(nobs))
    
    M12 = exog[:, np.newaxis, :, :] - data_predict[np.newaxis, :, np.newaxis, :] # shape (n_bootstrap, N_predict, nobs, k_vars)
    ker_weighted = ker[..., np.newaxis] # shape (n_bootstrap, N_predict, nobs, 1)
    
    # M22: (N_predict, k_vars, k_vars)
    # For each i: M12[i].T @ (M12[i] * ker[i])
    M22 = np.einsum('bpnk,bpnj->bpkj', M12 * ker_weighted, M12) # shape (n_bootstrap, N_predict, k_vars, k_vars)
    M12_sum = (M12 * ker_weighted).sum(axis=-2) # shape (n_bootstrap, N_predict, k_vars)
    
    # Build M matrix: (N_predict, k_vars+1, k_vars+1)
    M = np.zeros((n_bootstrap, N_predict, k_vars + 1, k_vars + 1))
    M[..., 0, 0] = ker.sum(axis=-1)
    M[..., 0, 1:] = M12_sum
    M[..., 1:, 0] = M12_sum
    M[..., 1:, 1:] = M22

    # ker_endog: (N_predict, nobs, 1)
    ker_endog = ker_weighted * endog[:, np.newaxis, :, :] # shape (n_bootstrap, N_predict, nobs, 1)
    
    # Build V vector: (n_bootstrap, N_predict, k_vars+1, 1)
    V = np.zeros((n_bootstrap, N_predict, k_vars + 1, 1))
    V[..., 0, 0] = ker_endog.sum(axis=(-2, -1))
    V[..., 1:, 0] = (M12 * ker_endog).sum(axis=-2)

    # Solve all linear systems at once
    # (N_predict, k_vars+1, k_vars+1) @ (N_predict, k_vars+1, 1)
    mean_mfx = np.linalg.pinv(M) @ V

    means = mean_mfx[..., 0, 0]
    mfx_all = mean_mfx[..., 1:, 0]

    return means, mfx_all


def gaussian_gpu(h, Xi, x):
    """
    Vectorized gaussian kernel.
    Xi: shape (n_bootstrap, nobs, k_vars)
    x: shape (N_predict, k_vars)
    Returns: shape (n_bootstrap, N_predict, nobs)
    """
    # Reshape for broadcasting: (N_predict, 1, k_vars) - (1, nobs, k_vars)
    diff = x[torch.newaxis, :, torch.newaxis, :] - Xi[:, torch.newaxis, ...]
    return (1. / np.sqrt(2 * torch.pi)) * torch.exp(-torch.sum(diff**2, axis=3) / (h**2 * 2.))

def _est_loc_linear_gpu(bw, endog, exog, data_predict):
    """
    Fully vectorized local linear estimation for multiple prediction points.
    data_predict: shape (N_predict, k_vars)
    Returns: mean (n_bootstrap, N_predict,), mfx (n_bootstrap, N_predict, k_vars)
    """
    n_bootstrap, nobs, k_vars = exog.shape
    N_predict = data_predict.shape[0]
    
    # Compute kernels for all prediction points at once
    # ker shape: (N_predict, nobs)
    ker = gaussian_gpu(bw[0], exog, data_predict) / (bw[0] * float(nobs))
    
    M12 = exog[:, torch.newaxis, :, :] - data_predict[torch.newaxis, :, torch.newaxis, :] # shape (n_bootstrap, N_predict, nobs, k_vars)
    ker_weighted = ker[..., torch.newaxis] # shape (n_bootstrap, N_predict, nobs, 1)
    
    # M22: (N_predict, k_vars, k_vars)
    # For each i: M12[i].T @ (M12[i] * ker[i])
    M22 = torch.einsum('bpnk,bpnj->bpkj', M12 * ker_weighted, M12) # shape (n_bootstrap, N_predict, k_vars, k_vars)
    M12_sum = (M12 * ker_weighted).sum(axis=-2) # shape (n_bootstrap, N_predict, k_vars)
    
    # Build M matrix: (N_predict, k_vars+1, k_vars+1)
    M = torch.zeros((n_bootstrap, N_predict, k_vars + 1, k_vars + 1), device=exog.device)
    M[..., 0, 0] = ker.sum(axis=-1)
    M[..., 0, 1:] = M12_sum
    M[..., 1:, 0] = M12_sum
    M[..., 1:, 1:] = M22

    # ker_endog: (N_predict, nobs, 1)
    ker_endog = ker_weighted * endog[:, np.newaxis, :, :] # shape (n_bootstrap, N_predict, nobs, 1)
    
    # Build V vector: (n_bootstrap, N_predict, k_vars+1, 1)
    V = torch.zeros((n_bootstrap, N_predict, k_vars + 1, 1), device=exog.device)
    V[..., 0, 0] = ker_endog.sum(axis=(-2, -1))
    V[..., 1:, 0] = (M12 * ker_endog).sum(axis=-2)

    # Solve all linear systems at once
    # (N_predict, k_vars+1, k_vars+1) @ (N_predict, k_vars+1, 1)
    mean_mfx = torch.linalg.pinv(M) @ V

    means = mean_mfx[..., 0, 0]
    mfx_all = mean_mfx[..., 1:, 0]

    return means, mfx_all

def kernel_reg_fit(endog, exog, data_predict, bw):
    endog = endog[..., np.newaxis]  # shape (n_bootstrap, nobs, 1)
    exog = exog[..., np.newaxis]  # shape (n_bootstrap, nobs, 1)
    data_predict = data_predict[:, np.newaxis]  # shape (N_predict, 1)
    bw = np.asarray(bw)

    if NUMBA_AVAILABLE:
        mean = _est_loc_linear_numba(bw, endog, exog, data_predict)
    else:
        mean, _ = _est_loc_linear(bw, endog, exog, data_predict)

    return mean

@torch.compile(mode='default')
def kernel_reg_fit_gpu(endog, exog, data_predict, bw):
    endog = endog[..., torch.newaxis]  # shape (n_bootstrap, nobs, 1)
    exog = exog[..., torch.newaxis]  # shape (n_bootstrap, nobs, 1)
    data_predict = data_predict[:, torch.newaxis]  # shape (N_predict, 1)
    bw = torch.as_tensor(bw, device=endog.device)

    mean, _ = _est_loc_linear_gpu(
        bw,
        endog,
        exog,
        data_predict
    )

    return mean


def bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap=100):
    n_samples = len(stance)
    indices = np.random.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

    # Resample with replacement
    boot_endog = stance[indices]
    boot_exog = timestamps[indices]

    # Fit kernel regression on bootstrap sample
    all_preds = kernel_reg_fit(boot_endog, boot_exog, test_x, [bandwidth])

    all_preds = np.clip(all_preds, -1, 1)
    return all_preds

def bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap=100):
    n_samples = len(stance)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = torch.randint(0, n_samples, (n_bootstrap, n_samples), device=device)
    stance = torch.as_tensor(stance, device=device)
    timestamps = torch.as_tensor(timestamps, device=device)
    test_x = torch.as_tensor(test_x, device=device)

    # Resample with replacement
    boot_endog = stance[indices]
    boot_exog = timestamps[indices]

    # Fit kernel regression on bootstrap sample
    all_preds = kernel_reg_fit_gpu(boot_endog, boot_exog, test_x, [bandwidth])

    all_preds = torch.clip(all_preds, -1, 1)
    return all_preds


# =============================================================================
# Bayesian Kernel Regression Methods
# =============================================================================

if GP_AVAILABLE:
    class ExactGPModel(gpytorch.models.ExactGP):
        """
        Exact GP with ZeroMean prior - predictions revert to 0 in sparse data regions.

        This provides proper Bayesian inference with a prior centered at 0, meaning:
        - One data point of 1 won't push prediction all the way to 1
        - Many data points of 1 will push prediction closer to 1
        - In regions with no data, predictions revert to 0 (the prior mean)

        The lengthscale parameter controls how correlated values at distance X are.
        """
        def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            if lengthscale_prior is not None:
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
                )
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def exact_gp_fit(
            train_x: np.ndarray,
            train_y: np.ndarray,
            test_x: np.ndarray,
            lengthscale_loc: float = 2.0,
            lengthscale_scale: float = 0.1,
            noise: float = 0.5,
            learn_hyperparams: bool = True,
            n_iter: int = 100,
            verbose: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fit an exact GP with zero-mean prior and predict at test points.

        Args:
            train_x: Training inputs, shape (n_train,)
            train_y: Training targets (stance values in [-1, 1]), shape (n_train,)
            test_x: Test inputs, shape (n_test,)
            lengthscale_loc: Location parameter for log-normal lengthscale prior
            lengthscale_scale: Scale parameter for log-normal lengthscale prior
            noise: Initial/fixed noise variance
            learn_hyperparams: If True, optimize hyperparameters; if False, use fixed values
            n_iter: Number of optimization iterations (if learning hyperparams)
            verbose: Print training progress

        Returns:
            mean: Posterior mean at test points, shape (n_test,)
            lower: Lower 95% confidence bound, shape (n_test,)
            upper: Upper 95% confidence bound, shape (n_test,)
            hyperparams: Dict with learned/used hyperparameters
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
        train_y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
        test_x = torch.as_tensor(test_x, dtype=torch.float32, device=device)

        # Create log-normal prior for lengthscale (same as ordinal GP)
        lengthscale_prior = gpytorch.priors.LogNormalPrior(
            loc=torch.log(torch.tensor(lengthscale_loc)),
            scale=lengthscale_scale
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise

        model = ExactGPModel(train_x, train_y, likelihood, lengthscale_prior=lengthscale_prior)

        model = model.to(device)
        likelihood = likelihood.to(device)

        if learn_hyperparams:
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(n_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                if verbose and (i + 1) % 20 == 0:
                    logger.info(f"Iter {i+1}/{n_iter} - Loss: {loss.item():.3f}")

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(test_x))
            mean = pred.mean
            lower, upper = pred.confidence_region()

        hyperparams = {
            'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
            'outputscale': model.covar_module.outputscale.item(),
            'noise': likelihood.noise.item()
        }

        mean = torch.clamp(mean, -1, 1).cpu().numpy()
        lower = torch.clamp(lower, -1, 1).cpu().numpy()
        upper = torch.clamp(upper, -1, 1).cpu().numpy()

        return mean, lower, upper, hyperparams


def bayesian_kernel_ridge_fit(
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        use_gpu: bool = True
    ) -> np.ndarray:
    """
    Bayesian Kernel Ridge Regression with implicit zero-mean prior.

    This is mathematically equivalent to the posterior mean of a GP with:
    - Zero mean prior
    - RBF kernel with given lengthscale
    - Noise variance = alpha * outputscale

    The alpha parameter controls the prior strength:
    - Higher alpha = stronger regularization = predictions closer to 0
    - Lower alpha = weaker regularization = predictions closer to data

    Args:
        train_x: Training inputs, shape (n_train,)
        train_y: Training targets, shape (n_train,)
        test_x: Test inputs, shape (n_test,)
        lengthscale: RBF kernel lengthscale (controls correlation distance)
        alpha: Regularization parameter (prior strength toward 0)
        use_gpu: Use GPU if available

    Returns:
        Predictions at test points, shape (n_test,)
    """
    if use_gpu and GPU_AVAILABLE:
        return _bayesian_kernel_ridge_fit_gpu(train_x, train_y, test_x, lengthscale, alpha)
    else:
        return _bayesian_kernel_ridge_fit_cpu(train_x, train_y, test_x, lengthscale, alpha)


def _bayesian_kernel_ridge_fit_cpu(train_x, train_y, test_x, lengthscale, alpha):
    """CPU implementation of Bayesian Kernel Ridge Regression."""
    train_x = np.asarray(train_x).reshape(-1, 1)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x).reshape(-1, 1)

    gamma = 1.0 / (2 * lengthscale ** 2)

    # Compute kernel matrices
    # K_train: (n_train, n_train)
    diff_train = train_x - train_x.T
    K_train = np.exp(-gamma * diff_train ** 2)

    # K_test: (n_test, n_train)
    diff_test = test_x - train_x.T
    K_test = np.exp(-gamma * diff_test ** 2)

    # Solve (K + alpha*I) @ w = y for w, then predict K_test @ w
    n = len(train_y)
    K_reg = K_train + alpha * np.eye(n)

    # Use solve instead of inverse for numerical stability
    weights = np.linalg.solve(K_reg, train_y)
    pred = K_test @ weights

    return np.clip(pred, -1, 1)


def _bayesian_kernel_ridge_fit_gpu(train_x, train_y, test_x, lengthscale, alpha):
    """GPU implementation of Bayesian Kernel Ridge Regression."""
    device = torch.device('cuda')

    train_x = torch.as_tensor(train_x, dtype=torch.float32, device=device).reshape(-1, 1)
    train_y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    test_x = torch.as_tensor(test_x, dtype=torch.float32, device=device).reshape(-1, 1)

    gamma = 1.0 / (2 * lengthscale ** 2)

    # Compute kernel matrices using broadcasting
    diff_train = train_x - train_x.T
    K_train = torch.exp(-gamma * diff_train ** 2)

    diff_test = test_x - train_x.T
    K_test = torch.exp(-gamma * diff_test ** 2)

    n = len(train_y)
    K_reg = K_train + alpha * torch.eye(n, device=device)

    weights = torch.linalg.solve(K_reg, train_y)
    pred = K_test @ weights

    return torch.clamp(pred, -1, 1).cpu().numpy()


def _bootstrap_bayesian_krr_gpu_single_batch(
        stance: torch.Tensor,
        timestamps: torch.Tensor,
        test_x: torch.Tensor,
        indices: torch.Tensor,
        lengthscale: float,
        alpha: float
    ) -> torch.Tensor:
    """Process a single batch of bootstraps on GPU."""
    n_samples = stance.shape[0]
    gamma = 1.0 / (2 * lengthscale ** 2)

    # Resample
    boot_stance = stance[indices]  # (batch_size, n_samples)
    boot_timestamps = timestamps[indices]  # (batch_size, n_samples)

    # K_train: (batch_size, n_samples, n_samples)
    boot_x = boot_timestamps.unsqueeze(-1)
    diff_train = boot_x - boot_x.transpose(-1, -2)
    K_train = torch.exp(-gamma * diff_train ** 2)

    # K_test: (batch_size, n_test, n_samples)
    test_x_expanded = test_x.unsqueeze(0).unsqueeze(-1)
    boot_x_for_test = boot_timestamps.unsqueeze(1)
    diff_test = test_x_expanded - boot_x_for_test
    K_test = torch.exp(-gamma * diff_test ** 2)

    # Regularize and solve
    eye = torch.eye(n_samples, device=stance.device).unsqueeze(0)
    K_reg = K_train + alpha * eye

    weights = torch.linalg.solve(K_reg, boot_stance)

    # Predict
    pred = torch.bmm(K_test, weights.unsqueeze(-1)).squeeze(-1)
    return torch.clamp(pred, -1, 1)


def bootstrap_bayesian_krr_gpu_batched(
        stance: np.ndarray,
        timestamps: np.ndarray,
        test_x: np.ndarray,
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        n_bootstrap: int = 100,
        max_gpu_mem_gb: float = 8.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched GPU implementation of bootstrap Bayesian KRR.

    Automatically batches bootstraps to fit in GPU memory.
    Falls back to CPU for very large sample sizes.
    """
    if not GPU_AVAILABLE:
        return bootstrap_bayesian_krr_numba(stance, timestamps, test_x, lengthscale, alpha, n_bootstrap)

    n_samples = len(stance)
    n_test = len(test_x)

    # Estimate memory per bootstrap: K_train + K_test + overhead
    # K_train: n_samples^2, K_test: n_test * n_samples, plus ~2x for solve workspace
    bytes_per_bootstrap = (n_samples * n_samples + n_test * n_samples) * 4 * 3  # float32, 3x overhead
    max_bytes = max_gpu_mem_gb * 1e9

    # Fall back to CPU if single bootstrap doesn't fit
    if bytes_per_bootstrap > max_bytes:
        return bootstrap_bayesian_krr_numba(stance, timestamps, test_x, lengthscale, alpha, n_bootstrap)

    # Calculate batch size that fits in memory
    batch_size = max(1, int(max_bytes // bytes_per_bootstrap))
    batch_size = min(batch_size, n_bootstrap)

    device = torch.device('cuda')
    stance_t = torch.as_tensor(stance, dtype=torch.float32, device=device)
    timestamps_t = torch.as_tensor(timestamps, dtype=torch.float32, device=device)
    test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)

    all_preds = []

    for i in range(0, n_bootstrap, batch_size):
        current_batch_size = min(batch_size, n_bootstrap - i)
        indices = torch.randint(0, n_samples, (current_batch_size, n_samples), device=device)

        pred = _bootstrap_bayesian_krr_gpu_single_batch(
            stance_t, timestamps_t, test_x_t, indices, lengthscale, alpha
        )
        all_preds.append(pred.cpu())

        # Clear GPU cache between batches
        if i + batch_size < n_bootstrap:
            torch.cuda.empty_cache()

    pred = torch.cat(all_preds, dim=0)

    mean = torch.mean(pred, dim=0).cpu().numpy()
    lower = torch.quantile(pred, 0.05, dim=0).cpu().numpy()
    upper = torch.quantile(pred, 0.95, dim=0).cpu().numpy()

    return mean, lower, upper


def bootstrap_bayesian_krr_cpu_batched(
        stance: np.ndarray,
        timestamps: np.ndarray,
        test_x: np.ndarray,
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched CPU implementation of bootstrap Bayesian KRR using numpy.

    Vectorizes across all bootstrap samples for better performance than looping.
    """
    n_samples = len(stance)

    # Generate all bootstrap indices at once
    indices = np.random.randint(0, n_samples, size=(n_bootstrap, n_samples))

    # Resample
    boot_stance = stance[indices]  # (n_bootstrap, n_samples)
    boot_timestamps = timestamps[indices]  # (n_bootstrap, n_samples)

    gamma = 1.0 / (2 * lengthscale ** 2)

    # Compute kernel matrices for all bootstraps
    # boot_timestamps: (n_bootstrap, n_samples) -> (n_bootstrap, n_samples, 1)
    boot_x = boot_timestamps[:, :, np.newaxis]

    # K_train: (n_bootstrap, n_samples, n_samples)
    diff_train = boot_x - boot_x.transpose(0, 2, 1)
    K_train = np.exp(-gamma * diff_train ** 2)

    # K_test: (n_bootstrap, n_test, n_samples)
    # test_x: (n_test,) -> (1, n_test, 1) for broadcasting
    # boot_timestamps: (n_bootstrap, n_samples) -> (n_bootstrap, 1, n_samples)
    test_x_expanded = test_x[np.newaxis, :, np.newaxis]  # (1, n_test, 1)
    boot_x_for_test = boot_timestamps[:, np.newaxis, :]  # (n_bootstrap, 1, n_samples)
    diff_test = test_x_expanded - boot_x_for_test  # (n_bootstrap, n_test, n_samples)
    K_test = np.exp(-gamma * diff_test ** 2)

    # Regularize and solve
    eye = np.eye(n_samples)[np.newaxis, :, :]  # (1, n_samples, n_samples)
    K_reg = K_train + alpha * eye

    # Solve (K_reg @ weights = boot_stance) for each bootstrap
    # np.linalg.solve broadcasts over leading dimensions
    weights = np.linalg.solve(K_reg, boot_stance)  # (n_bootstrap, n_samples)

    # Predict: (n_bootstrap, n_test, n_samples) @ (n_bootstrap, n_samples, 1)
    pred = np.einsum('btn,bn->bt', K_test, weights)  # (n_bootstrap, n_test)
    pred = np.clip(pred, -1, 1)

    mean = np.mean(pred, axis=0)
    lower = np.percentile(pred, 5, axis=0)
    upper = np.percentile(pred, 95, axis=0)

    return mean, lower, upper


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _bayesian_krr_bootstrap_numba(
            stance: np.ndarray,
            timestamps: np.ndarray,
            test_x: np.ndarray,
            indices: np.ndarray,
            lengthscale: float,
            alpha: float
        ) -> np.ndarray:
        """
        Numba-optimized bootstrap Bayesian KRR.

        Uses JIT compilation for fast inner loops without OpenMP (avoids OpenBLAS conflict).
        """
        n_bootstrap, n_samples = indices.shape
        n_test = len(test_x)
        gamma = 1.0 / (2 * lengthscale ** 2)

        all_preds = np.zeros((n_bootstrap, n_test))

        for b in range(n_bootstrap):
            # Resample for this bootstrap
            boot_stance = stance[indices[b]]
            boot_timestamps = timestamps[indices[b]]

            # Compute K_train: (n_samples, n_samples)
            K_train = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    diff = boot_timestamps[i] - boot_timestamps[j]
                    K_train[i, j] = np.exp(-gamma * diff * diff)

            # Add regularization
            for i in range(n_samples):
                K_train[i, i] += alpha

            # Solve K_train @ weights = boot_stance using Cholesky
            # Since K_train + alpha*I is positive definite, use Cholesky
            # Manual Cholesky solve for numba compatibility
            L = np.linalg.cholesky(K_train)
            # Solve L @ y = boot_stance
            y = np.zeros(n_samples)
            for i in range(n_samples):
                s = boot_stance[i]
                for j in range(i):
                    s -= L[i, j] * y[j]
                y[i] = s / L[i, i]
            # Solve L.T @ weights = y
            weights = np.zeros(n_samples)
            for i in range(n_samples - 1, -1, -1):
                s = y[i]
                for j in range(i + 1, n_samples):
                    s -= L[j, i] * weights[j]
                weights[i] = s / L[i, i]

            # Compute predictions at test points
            for t in range(n_test):
                pred = 0.0
                for i in range(n_samples):
                    diff = test_x[t] - boot_timestamps[i]
                    k = np.exp(-gamma * diff * diff)
                    pred += k * weights[i]
                # Clip to [-1, 1]
                all_preds[b, t] = max(-1.0, min(1.0, pred))

        return all_preds


def bootstrap_bayesian_krr_numba(
        stance: np.ndarray,
        timestamps: np.ndarray,
        test_x: np.ndarray,
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated bootstrap Bayesian KRR.

    Falls back to batched numpy if numba not available.
    """
    if not NUMBA_AVAILABLE:
        return bootstrap_bayesian_krr_cpu_batched(
            stance, timestamps, test_x, lengthscale, alpha, n_bootstrap
        )

    n_samples = len(stance)
    indices = np.random.randint(0, n_samples, size=(n_bootstrap, n_samples))

    all_preds = _bayesian_krr_bootstrap_numba(
        stance.astype(np.float64),
        timestamps.astype(np.float64),
        test_x.astype(np.float64),
        indices,
        float(lengthscale),
        float(alpha)
    )

    mean = np.mean(all_preds, axis=0)
    lower = np.percentile(all_preds, 5, axis=0)
    upper = np.percentile(all_preds, 95, axis=0)

    return mean, lower, upper


def _get_timestamps(df: pl.DataFrame, start_date: datetime.datetime, time_column, time_scale):
    
    numerator_match = re.search('^\d', time_scale)
    assert numerator_match is not None, "time_scale must begin with an integer number of units"
    unit_match = re.search('[a-z]+$', time_scale)
    available_time_units = ['h', 'd', 'w', 'mo', 'y']
    assert unit_match is not None, f"time_scale must end with a unit from {available_time_units}"

    numerator = numerator_match.group(0)
    unit = unit_match.group(0)

    try:
        numerator = int(numerator)
    except Exception:
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


def _get_time_series_data(filtered_df: pl.DataFrame, stance_target_type: str, time_column, time_scale):
    filtered_df = filtered_df.sort(time_column)
    start_date = filtered_df[time_column].min().date()
    end_date = filtered_df[time_column].max().date()
    timestamps = _get_timestamps(filtered_df, start_date, time_column, time_scale)

    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date)
        current_date += datetime.timedelta(days=1)
    day_df = pl.DataFrame({time_column: days})

    # Calculate volume
    day_df = day_df.join(
            filtered_df.select(pl.col(time_column).dt.date())\
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

    test_x = _get_timestamps(day_df, start_date, time_column, time_scale)
    
    if stance_target_type == 'noun-phrases':
        mapping = {
            'AGAINST': -1.,
            'NEUTRAL': 0.,
            'FAVOR': 1.
        }
    elif stance_target_type == 'claims':
        mapping = {
            'SUPPORTING': 1.,
            'LEANING SUPPORTING': 0.5,
            'IRRELEVANT': 0.,
            'DISCUSSING': 0.,
            'QUERYING': 0.,
            'LEANING REFUTING': -0.5,
            'REFUTING': -1.
        }
    else:
        raise ValueError("stance_target_type must be either 'noun-phrases' or 'claims'")
    
    filtered_df = filtered_df.with_columns(pl.col('Stance').replace_strict(mapping).cast(pl.Float32))
    stance = filtered_df['Stance'].to_numpy()

    classifier_ids = np.zeros_like(timestamps, dtype=np.uint16)

    return timestamps, stance, classifier_ids, test_x, day_df


def _combine_trend_df(trend_df: pl.DataFrame, pred, lower, upper, target_name, filter_type, filter_value):
    trend_df = trend_df.with_columns([
        pl.Series(name='trend_mean', values=pred, dtype=pl.Float32),
        pl.Series(name='trend_lower', values=lower, dtype=pl.Float32),
        pl.Series(name='trend_upper', values=upper, dtype=pl.Float32)
    ])
    
    # Join trend and volume data in a single operation
    trend_df = trend_df.with_columns([
        pl.lit(target_name).alias('target'),
        pl.lit(filter_type).alias('filter_type'),
        pl.lit(str(filter_value)).alias('filter_value')
    ])
    return trend_df

def _calculate_trends_for_filtered_df_with_batching(
        target_df: pl.DataFrame, 
        target_name: str, 
        stance_target_type: str,
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
            batch_trend_df, batch_interpolation_outputs_batch = _batch_calculate_trends_for_filtered_df(
                target_df, 
                target_name, 
                stance_target_type,
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

def _batch_calculate_trends_for_filtered_df(
        target_df: pl.DataFrame, 
        target_name: str, 
        stance_target_type: str,
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

        timestamps, stance, classifier_ids, test_x, day_df = _get_time_series_data(filtered_df, time_column, time_scale)
        all_timestamps.append(timestamps)
        all_stance.append(stance)
        all_classifier_ids.append(classifier_ids)
        all_test_x.append(test_x)
        all_day_dfs.append(day_df)

    assert stance_target_type == 'noun-phrases', "GP only supports noun-phrase targets"

    models = []
    batch_classifier_ids = []
    for i in range(len(all_timestamps)):
        timestamps = all_timestamps[i]
        stance = all_stance[i]
        classifier_ids = all_classifier_ids[i]

        model, classifier_ids = _setup_batchable_ordinal_gp_model(
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

    model = _batch_train_ordinal_likelihood_gp(models, batch_classifier_ids, verbose=verbose)
    preds, lowers, uppers = _get_batch_model_predictions(model, all_test_x)

    

    for filter_value, day_df, model, pred, lower, upper in zip(unique_values, all_day_dfs, models, preds, lowers, uppers):
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        likelihood_sigma = model.likelihood.sigma.item()
        trend_df = _combine_trend_df(day_df, pred, lower, upper, target_name, filter_type, filter_value)
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


def _calculate_trends_for_filtered_df(
        filtered_df: pl.DataFrame, 
        target_name, 
        stance_target_type,
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
    """Calculate trends for a filtered DataFrame with optimized operations."""
    # First sort by createtime - ensures consistent results
    start_date = filtered_df[time_column].min().date()
    end_date = filtered_df[time_column].max().date()

    if end_date - start_date < datetime.timedelta(days=1):
        # TODO implement static distribution calculation
        # return calculate_static_dist_for_filtered_df(filtered_df, target_name, filter_type, filter_value, trend_path, classifier_profiles)
        logger.warning(f"Skipping {target_name} {filter_type} {filter_value} - not enough data")
        return None, None

    timestamps, stance, classifier_ids, test_x, trend_df = _get_time_series_data(filtered_df, stance_target_type, time_column, time_scale)
    
    if interpolation_method == 'gp':
        assert stance_target_type == 'noun-phrases', "GP only supports noun-phrase targets"
        try:
            lengthscale, likelihood_sigma, losses, pred, lower, upper = _get_gp_timeseries(timestamps, stance, classifier_ids, classifier_profiles, test_x, lengthscale_loc, lengthscale_scale, sigma_loc, sigma_scale, verbose)
        except Exception as ex:
            logger.error(f"Failed for target {target_name} filter_type {filter_type} filter_value {filter_value} ex: {ex}")
            return None, None
        
        trend_df = _combine_trend_df(trend_df, pred, lower, upper, target_name, filter_type, filter_value)

        # write data to this
        interpolation_outputs = {
            'lengthscale': lengthscale,
            'sigma': likelihood_sigma,
            'loss': losses[-1],
            'target_name': target_name,
            'filter_type': filter_type,
            'filter_value': filter_value
        }
    elif interpolation_method == 'lowess':
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError:
            raise ImportError("Please install statsmodels to use LOWESS interpolation method: pip install statsmodels")
        
        it = 3
        pred = lowess(stance, timestamps, it=it, xvals=test_x, is_sorted=True, return_sorted=False)
        
        while np.all(np.isnan(pred)) and it < 30:
            it += 2
            pred = lowess(stance, timestamps, it=it, xvals=test_x, is_sorted=True, return_sorted=False)
        if np.all(np.isnan(pred)):
            logger.warning(f"Skipping {target_name} {filter_type} {filter_value} - LOWESS failed")
            return None, None
        
        # Interpolate NaN values
        pred[np.isnan(pred)] = np.interp(np.flatnonzero(np.isnan(pred)), np.flatnonzero(~np.isnan(pred)), pred[~np.isnan(pred)])

        pred = np.clip(pred, -1, 1)

        trend_df = _combine_trend_df(trend_df, pred, np.full_like(pred, np.nan), np.full_like(pred, np.nan), target_name, filter_type, filter_value)
        interpolation_outputs = {}

    elif interpolation_method == 'kernelreg':
        n_bootstrap = 100
        # Use same lengthscale as GP: mode of log-normal = exp(loc - scale²)
        bandwidth = np.exp(lengthscale_loc - lengthscale_scale**2)
        n_train_samples = len(stance)
        n_test_samples = len(test_x)

        max_array_size = 10**8 if GPU_AVAILABLE else 10**9
        if n_train_samples * n_test_samples >= max_array_size // n_bootstrap:
            # batch out bootstrapping to avoid memory issues
            batch_size = int(max(n_bootstrap // (n_train_samples * n_test_samples // (max_array_size // n_bootstrap)), 1))
            all_preds = []
            for i in range(0, n_bootstrap, batch_size):
                current_batch_size = min(batch_size, n_bootstrap - i)
                if GPU_AVAILABLE:
                    batch_preds = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, current_batch_size)
                else:
                    batch_preds = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, current_batch_size)
                all_preds.append(batch_preds)

            if GPU_AVAILABLE:
                all_preds = torch.cat(all_preds, dim=0)
            else:
                all_preds = np.vstack(all_preds)
        else:
            if GPU_AVAILABLE:
                all_preds = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap)
            else:
                all_preds = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap)

        # Compute statistics
        if GPU_AVAILABLE:
            pred_mean = torch.mean(all_preds, dim=0).cpu().numpy()
            pred_lower = torch.quantile(all_preds, 0.05, dim=0).cpu().numpy()
            pred_upper = torch.quantile(all_preds, 0.95, dim=0).cpu().numpy()
        else:
            pred_mean = np.mean(all_preds, axis=0)
            pred_lower = np.percentile(all_preds, 5, axis=0)
            pred_upper = np.percentile(all_preds, 95, axis=0)

        trend_df = _combine_trend_df(trend_df, pred_mean, pred_lower, pred_upper, target_name, filter_type, filter_value)
        interpolation_outputs = {}

    elif interpolation_method == 'bayesian_krr':
        n_bootstrap = 100
        # Use mode of log-normal prior: exp(loc - scale²)
        lengthscale = np.exp(lengthscale_loc - lengthscale_scale**2)
        alpha = 1.0  # Prior strength toward 0

        if GPU_AVAILABLE:
            pred_mean, pred_lower, pred_upper = bootstrap_bayesian_krr_gpu_batched(
                stance, timestamps, test_x,
                lengthscale=lengthscale, alpha=alpha, n_bootstrap=n_bootstrap
            )
        else:
            # Use numba-optimized version for CPU (falls back to batched numpy if numba unavailable)
            pred_mean, pred_lower, pred_upper = bootstrap_bayesian_krr_numba(
                stance, timestamps, test_x,
                lengthscale=lengthscale, alpha=alpha, n_bootstrap=n_bootstrap
            )

        trend_df = _combine_trend_df(trend_df, pred_mean, pred_lower, pred_upper, target_name, filter_type, filter_value)
        interpolation_outputs = {'lengthscale': lengthscale, 'alpha': alpha}

    elif interpolation_method == 'exact_gp':
        mean, lower, upper, hyperparams = exact_gp_fit(
            timestamps, stance, test_x,
            lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale,
            noise=0.5, learn_hyperparams=True, n_iter=50, verbose=verbose
        )

        trend_df = _combine_trend_df(trend_df, mean, lower, upper, target_name, filter_type, filter_value)
        interpolation_outputs = hyperparams

    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")

    return trend_df, interpolation_outputs

def infer_stance_trends_for_target(
        df: pl.DataFrame, 
        target_name, 
        filter_columns: List[str], 
        time_column: str = 'createtime',
        stance_target_type: str = 'noun-phrases',
        interpolation_method='gp',
        classifier_profiles=None,
        min_filter_count=5,
        time_scale='1mo',
        verbose=False,
        lengthscale_loc = 2.0, # mode at ~7.5 months
        lengthscale_scale = 0.1,
        sigma_loc = 1.0,
        sigma_scale = 0.2,
        get_overall_trend=True
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
    """Compute trend data for a specific target.
    
    Args:
        df (pl.DataFrame): DataFrame containing the data with 'Target' and 'Stance' columns.
        target_name (str): The target name to filter by.
        filter_columns (List[str]): List of columns to filter by for trend calculation (i.e. 'Source', 'Author', etc.).
        time_column (str): Column name for the time data, defaults to 'createtime'.
        interpolation_method (str): The method to use for interpolation, 'gp' for Gaussian Process and 'lowess' for LOWESS. Defaults to 'gp'.
            Gaussian Process is better for noise and modelling error, but is slower.
            LOWESS is faster, but does not model error, and does not allow setting a prior to properly model noise.
        classifier_profiles (Dict): Dictionary containing classifier profiles for ordinal GP.
        min_filter_count (int): Minimum count of occurrences for a filter value to be considered.
        time_scale (str): Time scale for the trends, e.g., '1mo', '1w'.
            Can be any combination of an integer and a time unit from ['h', 'd', 'w', 'mo', 'y'].
            If time scale is changed from '1mo', the lengthscale prior should be adjusted accordingly.
        verbose (bool): Whether to print progress information.
        lengthscale_loc (float): Location parameter for the lengthscale prior.
        lengthscale_scale (float): Scale parameter for the lengthscale prior.
        sigma_loc (float): Location parameter for the sigma prior.
        sigma_scale (float): Scale parameter for the sigma prior.

    Returns:
        Tuple[pl.DataFrame, List[Dict[str, Any]]]: A tuple containing the trend DataFrame and a list of interpolation outputs.
    """
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
            batch_trend_df, batch_interpolation_outputs = _calculate_trends_for_filtered_df_with_batching(
                filtered_df, 
                target_name, 
                stance_target_type,
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
                    trend_df, interpolation_outputs = _calculate_trends_for_filtered_df(
                        filtered_df, 
                        target_name, 
                        stance_target_type,
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
        

    if get_overall_trend:
        # First, the overall trend
        trend_df, interpolation_outputs = _calculate_trends_for_filtered_df(
            target_df, 
            target_name, 
            stance_target_type,
            'all', 
            'all', 
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
        if interpolation_outputs is not None:
            all_interpolation_outputs.append(interpolation_outputs)
        if trend_df is not None:
            if all_trend_df is None:
                all_trend_df = trend_df
            else:
                all_trend_df = pl.concat([all_trend_df, trend_df])

    return all_trend_df, all_interpolation_outputs

def _document_to_targets(document_df: pl.DataFrame, min_count):
    targets_df = document_df.explode(['Targets', 'Stances']).rename({'Targets': 'Target', 'Stances': 'Stance'})

    target_names = targets_df.group_by('Target')\
        .len()\
        .filter(pl.col('len') > min_count)\
        .drop_nulls('Target')\
        .sort('len', descending=True)['Target'].to_list()
    
    return targets_df, target_names

def infer_stance_trends_for_all_targets(
        document_df: pl.DataFrame, 
        time_column: str = 'createtime', 
        stance_target_type: str = 'noun-phrases',
        filter_columns: List[str] = [], 
        min_count: int = 5, 
        time_scale: str = '1mo',
        interpolation_method: str = 'gp',
        verbose: bool = False
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Compute trends for all targets in the document DataFrame.
    
    Args:
        document_df (pl.DataFrame): DataFrame containing the document data with 'Targets' and 'Stances' columns.
        time_column (str): Column name for the time data. Defaults to 'createtime'.
        filter_columns (List[str]): List of columns to filter by for trend calculation (i.e. 'Source', 'Author', etc.).
        min_count (int): Minimum count of occurrences for a filter value to be considered.
        time_scale (str): Time scale for the trends, e.g., '1mo', '1w'.
            Can be any combination of an integer and a time unit from ['h', 'd', 'w', 'mo', 'y'].
            If time scale is changed from '1mo', the lengthscale prior should be adjusted accordingly.
        interpolation_method (str): The method to use for interpolation, 'gp' for Gaussian Process, 'lowess' for LOWESS, and 'kernelreg' for Kernel Regression. Defaults to 'gp'.
            Gaussian Process is better for noise and modelling error, but is slower.
            LOWESS is faster, but does not model error, and does not allow setting a prior to properly model noise.
        verbose (bool): Whether to print progress information.
        
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - A DataFrame with trend data for each target and filter value.
            - A DataFrame with interpolation method outputs, useful for Gaussian Process outputs.
    """
    all_trend_gps_data = []
    all_trend_df = None

    targets_df, target_names = _document_to_targets(document_df, min_count)

    if interpolation_method == 'gp':
        classifier_profiles = _get_classifier_profiles()
    else:
        classifier_profiles = None

    for target_name in target_names:
        logger.info(f"Processing primary target: {target_name}")
            
        # Process the target with grouping
        target_trend_df, interpolation_outputs = infer_stance_trends_for_target(
            targets_df, 
            target_name, 
            filter_columns, 
            time_column,
            stance_target_type=stance_target_type,
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

