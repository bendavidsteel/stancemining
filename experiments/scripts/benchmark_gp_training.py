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
    setup_ordinal_gp_model, 
    train_ordinal_likelihood_gp,
    get_classifier_profiles,
    get_timestamps
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
    
    return timestamps, observed_stances, classifier_ids

# Custom training function with timing and scheduler
def timed_train_gp(model, likelihood, train_x, train_y, classifier_ids, optimizers, scheduler=None):
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
    
    start_time = time.time()
    training_iter = 10  # Fixed number of iterations for fair comparison
    
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
    
    end_time = time.time()
    training_time = end_time - start_time
    final_loss = losses[-1]
    
    return training_time, final_loss, losses

def benchmark_gp_training(
    learning_rates: List[float],
    ngd_learning_rates: List[float],
    scheduler_types: List[str],
    data_sizes: List[int],
    n_trials: int = 3
) -> pl.DataFrame:
    """Benchmark GP training time with different learning rates and schedulers."""
    
    results = []
    classifier_profiles = get_classifier_profiles()
    
    # Create progress bar for all combinations
    total_combinations = len(learning_rates) * len(ngd_learning_rates) * len(scheduler_types) * len(data_sizes) * n_trials
    pbar = tqdm(total=total_combinations, desc="Benchmarking GP Training")
    
    for data_size in data_sizes:
        for lr in learning_rates:
            for ngd_lr in ngd_learning_rates:
                for scheduler_type in scheduler_types:
                    for trial in range(n_trials):
                        # Generate synthetic data
                        timestamps, observed_stances, classifier_ids = generate_synthetic_data(
                            n_samples=data_size,
                            seed=42 + trial,  # Different seed for each trial
                            classifier_profiles=classifier_profiles
                        )
                    
                        # Setup GP model
                        model, likelihood, train_x, train_y, classifier_ids = setup_ordinal_gp_model(
                            timestamps, 
                            observed_stances, 
                            classifier_ids, 
                            classifier_profiles,
                            lengthscale_loc=2.0,
                            lengthscale_scale=0.1
                        )
                    
                        # Modify the optimizer in the training function
                        original_get_optimizer = None
                        
                        def custom_get_optimizer(model, likelihood, num_data=None):
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
                                        hyperparameter_optimizer, T_max=1000
                                    )
                                elif scheduler_type == 'cosine_warm':
                                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                        hyperparameter_optimizer, T_0=100
                                    )
                                elif scheduler_type == 'exponential':
                                    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                                        hyperparameter_optimizer, gamma=0.995
                                    )
                                elif scheduler_type == 'step':
                                    scheduler = torch.optim.lr_scheduler.StepLR(
                                        hyperparameter_optimizer, step_size=250, gamma=0.5
                                    )
                                else:  # 'none'
                                    scheduler = None
                            
                                optimizers = [hyperparameter_optimizer, variational_ngd_optimizer]
                                return optimizers, scheduler
                            else:
                                raise NotImplementedError("Only Natural Variational Distribution supported")
                    
                        # Monkey patch the training function to use custom optimizer
                        import stancemining.estimate as est_module
                        original_get_optimizer = est_module.get_optimizer
                        
                        
                        
                        try:
                            optimizers, scheduler = custom_get_optimizer(model, likelihood, train_y.size(0))
                            training_time, final_loss, losses = timed_train_gp(
                                model, likelihood, train_x, train_y, classifier_ids, optimizers, scheduler=scheduler
                            )
                        
                            # Calculate convergence metrics
                            if len(losses) >= 100:
                                early_loss = np.mean(losses[:100])
                                late_loss = np.mean(losses[-100:])
                                convergence_rate = (early_loss - late_loss) / early_loss
                            else:
                                convergence_rate = 0.0
                        
                            # Store results
                            result = {
                                'data_size': data_size,
                                'learning_rate': lr,
                                'ngd_learning_rate': ngd_lr,
                                'scheduler_type': scheduler_type,
                                'trial': trial,
                                'training_time': training_time,
                                'final_loss': final_loss,
                                'convergence_rate': convergence_rate,
                                'n_parameters': sum(p.numel() for p in model.parameters()),
                                'n_data_points': len(train_x),
                                'loss_trajectory': losses
                            }
                            results.append(result)
                        
                        except NotPSDError as e:
                            print(f"Failed for lr={lr}, ngd_lr={ngd_lr}, scheduler={scheduler_type}, size={data_size}, trial={trial}: {e}")
                            result = {
                                'data_size': data_size,
                                'learning_rate': lr,
                                'ngd_learning_rate': ngd_lr,
                                'scheduler_type': scheduler_type,
                                'trial': trial,
                                'training_time': float('inf'),
                                'final_loss': float('inf'),
                                'convergence_rate': 0.0,
                                'n_parameters': 0,
                                'n_data_points': len(train_x) if 'train_x' in locals() else 0,
                                'loss_trajectory': []
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
    summary = results_df.group_by(['data_size', 'learning_rate', 'ngd_learning_rate', 'scheduler_type'])\
        .agg([
            pl.col('training_time').mean().alias('avg_time'),
            pl.col('training_time').std().alias('std_time'),
            pl.col('final_loss').mean().alias('avg_loss'),
            pl.col('final_loss').std().alias('std_loss'),
            pl.col('convergence_rate').mean().alias('avg_convergence')
        ])\
        .sort(['data_size', 'avg_time'])
    
    print("\nBest configurations by data size (sorted by average training time):")
    print("-" * 60)
    
    for data_size in sorted(results_df['data_size'].unique()):
        print(f"\nData Size: {data_size}")
        size_results = summary.filter(pl.col('data_size') == data_size)\
            .head(3)  # Top 3 fastest
        
        for row in size_results.to_dicts():
            print(f"  LR: {row['learning_rate']:.4f}, NGD-LR: {row['ngd_learning_rate']:.4f}, Scheduler: {row['scheduler_type']:12s} -> "
                  f"Time: {row['avg_time']:.2f}±{row['std_time']:.2f}s, "
                  f"Loss: {row['avg_loss']:.4f}±{row['std_loss']:.4f}, "
                  f"Convergence: {row['avg_convergence']:.3f}")
    
    # Best overall configurations
    print("\n" + "-"*60)
    print("TOP 5 OVERALL CONFIGURATIONS (by training time):")
    print("-" * 60)
    
    top_configs = summary.sort('avg_time').head(5)
    for i, row in enumerate(top_configs.to_dicts(), 1):
        print(f"{i}. Data: {row['data_size']:3d}, LR: {row['learning_rate']:.4f}, NGD-LR: {row['ngd_learning_rate']:.4f}, "
              f"Scheduler: {row['scheduler_type']:12s} -> "
              f"Time: {row['avg_time']:.2f}±{row['std_time']:.2f}s, "
              f"Loss: {row['avg_loss']:.4f}±{row['std_loss']:.4f}")
    
    # Scheduler comparison
    print("\n" + "-"*60)
    print("SCHEDULER COMPARISON (averaged across all configurations):")
    print("-" * 60)
    
    scheduler_summary = results_df.group_by('scheduler_type')\
        .agg([
            pl.col('training_time').mean().alias('avg_time'),
            pl.col('final_loss').mean().alias('avg_loss'),
            pl.col('convergence_rate').mean().alias('avg_convergence')
        ])\
        .sort('avg_time')
    
    for row in scheduler_summary.to_dicts():
        print(f"{row['scheduler_type']:15s}: Time: {row['avg_time']:.2f}s, "
              f"Loss: {row['avg_loss']:.4f}, Convergence: {row['avg_convergence']:.3f}")
    
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
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    ngd_learning_rates = [0.05, 0.1, 0.2, 0.5]
    scheduler_types = ['none', 'cosine', 'cosine_warm', 'exponential', 'step']
    data_sizes = [20, 100, 200]
    n_trials = 3
    
    print(f"\nConfiguration:")
    print(f"  Learning rates: {learning_rates}")
    print(f"  NGD learning rates: {ngd_learning_rates}")
    print(f"  Schedulers: {scheduler_types}")
    print(f"  Data sizes: {data_sizes}")
    print(f"  Trials per config: {n_trials}")
    print(f"  Total runs: {len(learning_rates) * len(ngd_learning_rates) * len(scheduler_types) * len(data_sizes) * n_trials}")
    
    # Run benchmark
    results_df = benchmark_gp_training(
        learning_rates=learning_rates,
        ngd_learning_rates=ngd_learning_rates,
        scheduler_types=scheduler_types,
        data_sizes=data_sizes,
        n_trials=n_trials
    )
    
    # Analyze results
    analyze_results(results_df)
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")

if __name__ == "__main__":
    main()