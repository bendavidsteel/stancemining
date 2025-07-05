import datetime
import time
from typing import List, Dict, Any
import os

from linear_operator.utils.errors import NotPSDError
import numpy as np
import polars as pl
import torch
import gpytorch
from tqdm import tqdm
import matplotlib.pyplot as plt

from stancemining.estimate import (
    _setup_ordinal_gp_model, 
    _train_ordinal_likelihood_gp,
    _get_classifier_profiles,
    _get_timestamps,
    _get_model_prediction
)

def generate_synthetic_data(
    n_samples: int = 100,
    n_time_points: int = 50,
    noise_scale: float = 0.5,
    random_walk_scale: float = 0.05,
    seed: int = 42,
    classifier_profiles=None
) -> tuple:
    """Generate synthetic time series data for GP training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate time points (in days)
    days = np.linspace(0, 365, n_time_points)
    
    # Generate latent stance trajectory using random walk
    latent_stance = [np.random.uniform(-1, 1)]
    for _ in range(n_time_points - 1):
        next_stance = np.clip(
            np.random.normal(latent_stance[-1], scale=random_walk_scale), 
            -1, 1
        )
        latent_stance.append(next_stance)
    latent_stance = np.array(latent_stance)
    
    # Add observation noise and quantize to ordinal scale
    noise = np.random.normal(scale=noise_scale, size=latent_stance.shape)
    noisy_stance = latent_stance + noise
    
    # Sample random subset of observations
    n_obs = min(n_samples, n_time_points)
    obs_indices = np.sort(np.random.choice(n_time_points, n_obs, replace=False))
    
    timestamps = days[obs_indices]
    true_stances = np.round(np.clip(noisy_stance[obs_indices], -1, 1)).astype(int)
    
    # Simulate noisy observations using classifier profiles if provided
    if classifier_profiles is not None:
        observe_probs = {
            -1: np.array([classifier_profiles[0]['true_against'][k] for k in ['predicted_against', 'predicted_neutral', 'predicted_favor']]),
            0: np.array([classifier_profiles[0]['true_neutral'][k] for k in ['predicted_against', 'predicted_neutral', 'predicted_favor']]),
            1: np.array([classifier_profiles[0]['true_favor'][k] for k in ['predicted_against', 'predicted_neutral', 'predicted_favor']])
        }
        observe_probs = {k: v / np.sum(v) for k, v in observe_probs.items()}
        
        observed_stances = []
        for stance in true_stances:
            observed_stance = np.random.choice([-1, 0, 1], p=observe_probs[stance])
            observed_stances.append(observed_stance)
        observed_stances = np.array(observed_stances)
    else:
        observed_stances = true_stances
    
    classifier_ids = np.zeros_like(observed_stances, dtype=int)
    
    return days, latent_stance, timestamps, observed_stances, classifier_ids

# Custom training function with timing and scheduler
def train_gp(model, likelihood, train_x, train_y, classifier_ids, optimizers, training_iter, scheduler=None):
    model.train()
    likelihood.train()
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x_gpu = train_x.cuda()
        train_y_gpu = train_y.cuda()
        classifier_ids_gpu = classifier_ids.cuda()
    else:
        train_x_gpu = train_x
        train_y_gpu = train_y
        classifier_ids_gpu = classifier_ids
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.size(0))
    
    losses = []
    for k in range(training_iter):
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        with gpytorch.settings.variational_cholesky_jitter(1e-4):
            output = model(train_x_gpu)
            loss = -mll(output, train_y_gpu, classifier_ids=classifier_ids_gpu)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=1.0)
        
        for optimizer in optimizers:
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        losses.append(loss.item())
    
    return losses

def custom_get_optimizer(model, likelihood, ngd_lr, lr, scheduler_type, max_epochs, num_data=None):
    adam_params = []
    variational_distribution = model.variational_strategy._variational_distribution
    adam_params.append({'params': likelihood.parameters()})
    
    if isinstance(variational_distribution, gpytorch.variational.NaturalVariationalDistribution):
        variational_ngd_optimizer = gpytorch.optim.NGD(
            model.variational_parameters(), 
            num_data=num_data, 
            lr=ngd_lr  # Use custom NGD learning rate
        )
    
        hyperparameter_optimizer = torch.optim.Adam(
            adam_params + [{'params': model.hyperparameters()}],
            lr=lr  # Use custom learning rate
        )
    
        # Apply different schedulers
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                hyperparameter_optimizer, T_max=max_epochs
            )
        elif scheduler_type == 'cosine_warm':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                hyperparameter_optimizer, T_0=max_epochs // 5
            )
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                hyperparameter_optimizer, gamma=0.995
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                hyperparameter_optimizer, step_size=max_epochs // 5, gamma=0.5
            )
        else:  # 'none'
            scheduler = None
    
        optimizers = [hyperparameter_optimizer, variational_ngd_optimizer]
        return optimizers, scheduler
    else:
        raise NotImplementedError("Only Natural Variational Distribution supported")

                        

def benchmark_gp_training(
    learning_rates: List[float],
    ngd_learning_rates: List[float],
    scheduler_types: List[str],
    data_sizes: List[int],
    noise_scales: List[float],
    rw_scales: List[float]
) -> pl.DataFrame:
    """Benchmark GP training time with different learning rates and schedulers."""
    
    results = []
    classifier_profiles = _get_classifier_profiles()
    
    # Create progress bar for all combinations
    total_combinations = len(learning_rates) * len(ngd_learning_rates) * len(scheduler_types) * len(data_sizes) * len(noise_scales) * len(rw_scales)
    pbar = tqdm(total=total_combinations, desc="Benchmarking GP Training")
    
    max_epochs = 500
    for data_size in data_sizes:
        for lr in learning_rates:
            for ngd_lr in ngd_learning_rates:
                for scheduler_type in scheduler_types:
                    for noise_scale in noise_scales:
                        for rw_scale in rw_scales:
                            # Generate synthetic data
                            test_x, latent_user_stance, timestamps, observed_stances, classifier_ids = generate_synthetic_data(
                                n_samples=data_size,
                                noise_scale=noise_scale,
                                random_walk_scale=rw_scale,
                                seed=42,  # Different seed for each trial
                                classifier_profiles=classifier_profiles
                            )
                        
                            # Setup GP model
                            model, likelihood, train_x, train_y, classifier_ids = _setup_ordinal_gp_model(
                                timestamps, 
                                observed_stances, 
                                classifier_ids, 
                                classifier_profiles,
                                lengthscale_loc=2.0,
                                lengthscale_scale=0.1
                            )
                        
                            
                            optimizers, scheduler = custom_get_optimizer(model, likelihood, ngd_lr, lr, scheduler_type, max_epochs, train_y.size(0))
                            losses = train_gp(
                                model, likelihood, train_x, train_y, classifier_ids, optimizers, max_epochs, scheduler=scheduler
                            )

                            # Get final model prediction
                            model.eval()
                            inferred_user_stance, _, _ = _get_model_prediction(model, test_x)
                            mse = np.mean((inferred_user_stance - latent_user_stance) ** 2)
                        
                            min_loss = min(losses)
                            final_loss = losses[-1]

                            # determine epoch where loss was 90% of the way to minimum loss
                            ninety_percent_loss = min_loss + 0.1 * (losses[0] - min_loss)
                            epoch_ninety_percent_loss = np.min(
                                np.where(np.array(losses) <= ninety_percent_loss)
                            )
                            # Store results
                            result = {
                                'data_size': data_size,
                                'learning_rate': lr,
                                'ngd_learning_rate': ngd_lr,
                                'scheduler_type': scheduler_type,
                                'noise_scale': noise_scale,
                                'rw_scale': rw_scale,
                                'final_loss': final_loss,
                                'min_loss': min_loss,
                                'epoch_ninety_percent_loss': epoch_ninety_percent_loss,
                                'n_data_points': len(train_x),
                                'loss_trajectory': losses,
                                'mse': mse  
                            }
                            results.append(result)
                        
                            pbar.update(1)
    
    pbar.close()
    return pl.DataFrame(results)

def analyze_results(results_df: pl.DataFrame) -> None:
    """Analyze and print benchmark results."""
    
    print("\n" + "="*60)
    print("GP TRAINING BENCHMARK RESULTS")
    print("="*60)
    
    # Group by configuration and calculate statistics
    summary = results_df.group_by(['learning_rate', 'ngd_learning_rate', 'scheduler_type'])\
        .agg([
            pl.col('min_loss').mean().alias('avg_min_loss'),
            pl.col('min_loss').std().alias('std_min_loss'),
            pl.col('epoch_ninety_percent_loss').mean().alias('avg_epoch_ninety_percent_loss'),
            pl.col('epoch_ninety_percent_loss').std().alias('std_epoch_ninety_percent_loss'),
            pl.col('mse').mean().alias('avg_mse'),
            pl.col('mse').std().alias('std_mse')
        ])
    
    # plot map of percentage of epochs to reach 90% of minimum loss vs average MSE
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary['avg_epoch_ninety_percent_loss'], 
        summary['avg_min_loss'], 
        s=100, 
        alpha=0.7
    )

    ax.xaxis.set_inverted(True)
    ax.yaxis.set_inverted(True)
    ax.set_yscale('log')
    ax.set_xlabel('Percentage of epochs to reach 90% of minimum loss')
    ax.set_ylabel('Average MSE')
    fig.savefig('figs/percentage_epochs_vs_avg_mse.png', dpi=300, bbox_inches='tight')
    
    # Best overall configurations
    print("\n" + "-"*60)
    print("TOP 5 CONFIGURATIONS (by number of epochs to 90% loss reduction):")
    print("-" * 60)

    top_configs = summary.sort('avg_epoch_ninety_percent_loss').head(5)
    for i, row in enumerate(top_configs.to_dicts(), 1):
        print(f"{i}. LR: {row['learning_rate']:.4f}, NGD-LR: {row['ngd_learning_rate']:.4f}, "
              f"Scheduler: {row['scheduler_type']:12s} -> "
              f"Percentage of epochs to 90% loss reduction: {row['avg_epoch_ninety_percent_loss']:.2f}±{row['std_epoch_ninety_percent_loss']:.2f}s, "
              f"Loss: {row['avg_min_loss']:.4f}±{row['std_min_loss']:.4f}")

    # best by min loss
    print("\n" + "-"*60)
    print("TOP 5 CONFIGURATIONS BY MIN LOSS:")
    print("-" * 60)
    best_loss_configs = summary.sort('avg_min_loss').head(5)
    for i, row in enumerate(best_loss_configs.to_dicts(), 1):
        print(f"{i}. LR: {row['learning_rate']:.4f}, NGD-LR: {row['ngd_learning_rate']:.4f}, "
              f"Scheduler: {row['scheduler_type']:12s} -> "
              f"Min Loss: {row['avg_min_loss']:.4f}±{row['std_min_loss']:.4f}, "
              f"Percentage of epochs to 90% loss reduction: {row['avg_epoch_ninety_percent_loss']:.2f}±{row['std_epoch_ninety_percent_loss']:.2f}s")

    # Scheduler comparison
    print("\n" + "-"*60)
    print("SCHEDULER COMPARISON (averaged across all configurations):")
    print("-" * 60)
    
    scheduler_summary = results_df.group_by('scheduler_type')\
        .agg([
            pl.col('epoch_ninety_percent_loss').mean().alias('avg_epoch_ninety_percent_loss'),
            pl.col('epoch_ninety_percent_loss').std().alias('std_epoch_ninety_percent_loss'),
            pl.col('min_loss').std().alias('std_min_loss'),
            pl.col('min_loss').mean().alias('avg_min_loss'),
        ])\
        .sort('avg_min_loss')

    for row in scheduler_summary.to_dicts():
        print(f"{row['scheduler_type']:15s}: Min Loss: {row['avg_min_loss']:.4f}, "
              f"Percentage of epochs to 90% loss reduction: {row['avg_epoch_ninety_percent_loss']:.2f}±{row['std_epoch_ninety_percent_loss']:.2f}s")

    # for best scheduler, print best learning rate and NGD learning rates
    best_scheduler = scheduler_summary.sort('avg_min_loss').head(1).to_dicts()[0]
    best_scheduler_type = best_scheduler['scheduler_type']
    best_configs = summary.filter(pl.col('scheduler_type') == best_scheduler_type).head(5)
    print("\n" + "-"*60)
    print(f"BEST SCHEDULER: {best_scheduler_type}")
    print("-" * 60)
    for i, row in enumerate(best_configs.to_dicts(), 1):
        print(f"{i}. LR: {row['learning_rate']:.4f}, NGD-LR: {row['ngd_learning_rate']:.4f} -> "
              f"Min Loss: {row['avg_min_loss']:.4f}±{row['std_min_loss']:.4f}, "
              f"Percentage of epochs to 90% loss reduction: {row['avg_epoch_ninety_percent_loss']:.2f}±{row['std_epoch_ninety_percent_loss']:.2f}s")


    # Create plots
    plot_loss_curves(results_df)
    plot_loss_curves_by_lr(results_df)

def plot_loss_curves(results_df: pl.DataFrame) -> None:
    """Plot loss curves over epochs for each scheduler with separate subplots for each LR combination."""
    
    # Create figs directory and subdirectory if they don't exist
    os.makedirs('figs/loss_curves_by_scheduler', exist_ok=True)
    
    # Convert to pandas for easier plotting
    df = results_df.to_pandas()
    
    # Filter out infinite values and empty trajectories
    df = df[np.isfinite(df['final_loss']) & (df['loss_trajectory'].apply(len) > 0)]
    
    # Get unique schedulers and learning rates
    schedulers = sorted(df['scheduler_type'].unique())
    learning_rates = sorted(df['learning_rate'].unique())
    ngd_learning_rates = sorted(df['ngd_learning_rate'].unique())
    lr_ngd_combinations = [(lr, ngd_lr) for lr in learning_rates for ngd_lr in ngd_learning_rates]
    
    # Create separate figure for each scheduler
    for scheduler in schedulers:
        scheduler_data = df[df['scheduler_type'] == scheduler]
        
        # Create subplots for each LR/NGD combination
        n_combinations = len(lr_ngd_combinations)
        n_cols = min(3, n_combinations)  # Max 3 columns
        n_rows = (n_combinations + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        if n_combinations > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        for i, (lr, ngd_lr) in enumerate(lr_ngd_combinations):
            ax = axes_flat[i]
            lr_data = scheduler_data[(scheduler_data['learning_rate'] == lr) & (scheduler_data['ngd_learning_rate'] == ngd_lr)]
            
            if len(lr_data) > 0:
                # Collect all loss trajectories for this lr/ngd_lr/scheduler combo
                trajectories = [traj for traj in lr_data['loss_trajectory'] if len(traj) > 0]
                
                if trajectories:
                    # Convert to numpy array for easier computation
                    max_len = max(len(traj) for traj in trajectories)
                    
                    # Calculate mean and std across trials
                    mean_loss = []
                    std_loss = []
                    
                    for epoch in range(max_len):
                        epoch_losses = [traj[epoch] for traj in trajectories if epoch < len(traj)]
                        if epoch_losses:
                            mean_loss.append(np.mean(epoch_losses))
                            std_loss.append(np.std(epoch_losses))
                        else:
                            mean_loss.append(np.nan)
                            std_loss.append(np.nan)
                    
                    epochs = range(len(mean_loss))
                    mean_loss = np.array(mean_loss)
                    std_loss = np.array(std_loss)
                    
                    # Plot with confidence intervals
                    ax.plot(epochs, mean_loss, 'b-', linewidth=2)
                    ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                                   color='blue', alpha=0.2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'LR: {lr}, NGD-LR: {ngd_lr}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_combinations, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'{scheduler.title()} Scheduler - Loss Curves', fontsize=16, y=0.98)
        
        # Save individual figure for each scheduler
        filename = f'loss_curves_{scheduler}_scheduler.png'
        filepath = os.path.join('figs/loss_curves_by_scheduler', filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Loss curve plots saved to: figs/loss_curves_by_scheduler/ ({len(schedulers)} figures)")

def plot_loss_curves_by_lr(results_df: pl.DataFrame) -> None:
    """Plot loss curves over epochs for each learning rate with different schedulers."""
    
    # Create figs directory and subdirectory if they don't exist
    os.makedirs('figs/loss_curves_by_lr', exist_ok=True)
    
    # Convert to pandas for easier plotting
    df = results_df.to_pandas()
    
    # Filter out infinite values and empty trajectories
    df = df[np.isfinite(df['final_loss']) & (df['loss_trajectory'].apply(len) > 0)]
    
    # Get unique schedulers and learning rates
    schedulers = sorted(df['scheduler_type'].unique())
    learning_rates = sorted(df['learning_rate'].unique())
    ngd_learning_rates = sorted(df['ngd_learning_rate'].unique())
    
    # Create separate figure for each LR/NGD combination
    lr_ngd_combinations = [(lr, ngd_lr) for lr in learning_rates for ngd_lr in ngd_learning_rates]
    colors = plt.cm.viridis(np.linspace(0, 1, len(schedulers)))
    
    for lr, ngd_lr in lr_ngd_combinations:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        lr_data = df[(df['learning_rate'] == lr) & (df['ngd_learning_rate'] == ngd_lr)]
        
        for j, scheduler in enumerate(schedulers):
            scheduler_data = lr_data[lr_data['scheduler_type'] == scheduler]
            
            if len(scheduler_data) > 0:
                # Collect all loss trajectories for this scheduler/lr combo
                trajectories = [traj for traj in scheduler_data['loss_trajectory'] if len(traj) > 0]
                
                if trajectories:
                    # Convert to numpy array for easier computation
                    max_len = max(len(traj) for traj in trajectories)
                    
                    # Calculate mean and std across trials
                    mean_loss = []
                    std_loss = []
                    
                    for epoch in range(max_len):
                        epoch_losses = [traj[epoch] for traj in trajectories if epoch < len(traj)]
                        if epoch_losses:
                            mean_loss.append(np.mean(epoch_losses))
                            std_loss.append(np.std(epoch_losses))
                        else:
                            mean_loss.append(np.nan)
                            std_loss.append(np.nan)
                    
                    epochs = range(len(mean_loss))
                    mean_loss = np.array(mean_loss)
                    std_loss = np.array(std_loss)
                    
                    # Plot with confidence intervals
                    ax.plot(epochs, mean_loss, color=colors[j], label=scheduler, linewidth=2)
                    ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                                   color=colors[j], alpha=0.2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Curves: LR={lr}, NGD-LR={ngd_lr}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save individual figure
        filename = f'loss_curves_lr_{lr}_ngd_{ngd_lr}.png'
        filepath = os.path.join('figs/loss_curves_by_lr', filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Loss curve plots saved to: figs/loss_curves_by_lr/ ({len(lr_ngd_combinations)} figures)")

def main():
    """Main benchmarking function."""
    print("Starting GP Training Time Benchmark")
    print("This will test different learning rates and schedulers on synthetic data")
    
    # Configuration
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    ngd_learning_rates = [0.05, 0.1, 0.2, 0.5]
    scheduler_types = ['none', 'cosine', 'cosine_warm', 'exponential', 'step']
    data_sizes = [20, 100, 200]
    noise_scales = [0.2, 0.5]
    rw_scales = [0.1, 0.5]
    
    print(f"\nConfiguration:")
    print(f"  Learning rates: {learning_rates}")
    print(f"  NGD learning rates: {ngd_learning_rates}")
    print(f"  Schedulers: {scheduler_types}")
    print(f"  Data sizes: {data_sizes}")
    print(f"  Noise scales: {noise_scales}")
    print(f"  Random walk scales: {rw_scales}")
    print(f"  Total runs: {len(learning_rates) * len(ngd_learning_rates) * len(scheduler_types) * len(data_sizes) * len(noise_scales) * len(rw_scales)}")

    # Run benchmark
    results_df = benchmark_gp_training(
        learning_rates=learning_rates,
        ngd_learning_rates=ngd_learning_rates,
        scheduler_types=scheduler_types,
        data_sizes=data_sizes,
        noise_scales=noise_scales,
        rw_scales=rw_scales
    )
    
    # Analyze results
    analyze_results(results_df)
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")

if __name__ == "__main__":
    main()