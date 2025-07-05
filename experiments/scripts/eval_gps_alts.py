import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
from statsmodels.nonparametric.smoothers_lowess import lowess
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model

from stancemining.estimate import _get_gp_timeseries, _get_classifier_profiles, _get_timestamps

def get_mse(predicted, actual):
    """Calculate Mean Squared Error between predicted and actual values."""
    return np.mean((np.array(predicted) - np.array(actual)) ** 2)

def rolling_avg(timestamps, values, window_size=30):
    """Calculate rolling average with a specified window size."""
    if len(timestamps) < window_size:
        return np.array([np.nan] * len(timestamps))
    
    # TODO revisit implementation
    rolling_means = []
    for i in range(len(timestamps)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(timestamps), i + window_size // 2 + 1)
        mean_value = np.mean(values[start_idx:end_idx])
        rolling_means.append(mean_value)
    
    return np.array(rolling_means)

def spline_interpolate(timestamps, observations, test_x, n_knots=4, degree=3, alpha=0.1):
    """
    Use smoothing splines to interpolate stance values.
    
    Args:
        timestamps: observation times
        observations: observed stance values  
        test_x: times to predict at
        n_knots: number of knots for the spline
        degree: degree of the spline
        alpha: regularization strength
    """
    # Create smoothing spline
    model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.SplineTransformer(n_knots=n_knots, degree=degree), sklearn.linear_model.Ridge(alpha=alpha))
    model.fit(timestamps.reshape(-1, 1), observations)
    test_y = model.predict(test_x.reshape(-1, 1))
    return test_y

def get_lowess(observed_stances, base_timestamps, test_x):
    lowess_fit = lowess(observed_stances, base_timestamps, xvals=test_x, is_sorted=True, return_sorted=True)
    # replace nans with their neighbors
    # TODO check that this works
    lowess_fit[np.isnan(lowess_fit)] = np.interp(np.flatnonzero(np.isnan(lowess_fit)), np.flatnonzero(~np.isnan(lowess_fit)), lowess_fit[~np.isnan(lowess_fit)])
    return lowess_fit

def eval_gp(n_samples=2, noise_scale=0.1, random_walk_scale=0.01, num_days=365, plot=False):
    # create synthetic data using random walk
    days = np.arange(num_days)
    latent_user_stance = [np.random.uniform(-1, 1, size=(n_samples,))]
    for _ in range(len(days) - 1):
        next_user_stance = np.clip(np.random.normal(latent_user_stance[-1], scale=random_walk_scale), -1, 1)
        latent_user_stance.append(next_user_stance)
    latent_user_stance = np.stack(latent_user_stance, axis=-1)

    # add noise
    noise = np.random.normal(scale=noise_scale, size=latent_user_stance.shape)
    noisy_latent_user_stance = latent_user_stance + noise

    # sample time stamps and quantize stance
    classifier_profile = _get_classifier_profiles()
    observe_probs = {
        -1: np.array([classifier_profile[0]['true_against'][k] for k in ['predicted_against', 'predicted_neutral', 'predicted_favor']]),
        0: np.array([classifier_profile[0]['true_neutral'][k] for k in ['predicted_against', 'predicted_neutral', 'predicted_favor']]),
        1: np.array([classifier_profile[0]['true_favor'][k] for k in ['predicted_against', 'predicted_neutral', 'predicted_favor']])
    }
    observe_probs = {k: v / np.sum(v) for k, v in observe_probs.items()}

    time_column = 'datetime'
    time_scale = '1mo'
    start_date = datetime.datetime(2024, 1, 1)

    test_df = pl.DataFrame({
        time_column: [start_date + datetime.timedelta(days=int(t)) for t in days],
    })
    test_x = _get_timestamps(test_df, start_date, time_column, time_scale)

    # Initialize timing and results lists
    lowess_times = []
    gp_times = []
    kalman_times = []
    spline_times = []
    
    lowess_mses = []
    gp_mses = []
    kalman_mses = []
    spline_mses = []

    lowess_fits = []
    gp_fits = []
    kalman_fits = []
    spline_fits = []
    gp_lowers = []
    gp_uppers = []

    all_base_timestamps = []
    all_base_datetimes = []
    all_observed_stances = []
    all_true_stances = []

    for i in range(n_samples):
        n_stance_samples = np.random.randint(5, 30)
        idxs = np.random.choice(np.arange(num_days), size=n_stance_samples, replace=False)
        idxs = np.sort(idxs)
        timestamps = days[idxs]
        sorted_df = pl.DataFrame({
            time_column: [start_date + datetime.timedelta(days=int(i)) for i in timestamps],
        })
        base_timestamps = _get_timestamps(sorted_df, start_date, time_column, time_scale)
        user_true_stances = np.round(np.clip(noisy_latent_user_stance[i, idxs], -1, 1))

        all_true_stances.append(user_true_stances)

        # observe stances
        observed_stances = []
        for stance in user_true_stances:
            observed_stance = np.random.choice([-1, 0, 1], p=observe_probs[stance])
            observed_stances.append(observed_stance)
        observed_stances = np.array(observed_stances)

        all_base_timestamps.append(base_timestamps)
        all_base_datetimes.append(sorted_df[time_column].to_list())
        all_observed_stances.append(observed_stances)

        # Smoothing Splines
        start_time = datetime.datetime.now()
        spline_fit = spline_interpolate(base_timestamps, observed_stances, test_x)
        end_time = datetime.datetime.now()
        spline_time = (end_time - start_time).total_seconds()

        # LOWESS
        start_time = datetime.datetime.now()
        lowess_fit = get_lowess(observed_stances, base_timestamps, test_x)
        end_time = datetime.datetime.now()
        lowess_time = (end_time - start_time).total_seconds()

        # GP
        classifier_ids = np.zeros_like(observed_stances, dtype=int)
        start_time = datetime.datetime.now()
        lengthscale, likelihood_sigma, losses, gp_mean, gp_lower, gp_upper = _get_gp_timeseries(base_timestamps, observed_stances, classifier_ids, classifier_profile, test_x)
        # gp_mean, gp_lower, gp_upper = np.zeros_like(test_x), np.zeros_like(test_x), np.zeros_like(test_x)
        end_time = datetime.datetime.now()
        gp_time = (end_time - start_time).total_seconds()

        # Kalman Filter
        # start_time = datetime.datetime.now()
        # kalman_fit = kalman_interpolate(base_timestamps, observed_stances, test_x, 
        #                                process_noise_var=0.01**2, obs_noise_var=0.5**2)
        # end_time = datetime.datetime.now()
        # kalman_time = (end_time - start_time).total_seconds()

        # Store fits
        lowess_fits.append(lowess_fit)
        gp_fits.append(gp_mean)
        # kalman_fits.append(kalman_fit)
        spline_fits.append(spline_fit)
        gp_lowers.append(gp_lower)
        gp_uppers.append(gp_upper)

        # Calculate MSEs
        lowess_mse = get_mse(lowess_fit, latent_user_stance[i])
        gp_mse = get_mse(gp_mean, latent_user_stance[i])
        # kalman_mse = get_mse(kalman_fit, latent_user_stance[i])
        spline_mse = get_mse(spline_fit, latent_user_stance[i])

        # Store times and MSEs
        lowess_times.append(lowess_time)
        gp_times.append(gp_time)
        # kalman_times.append(kalman_time)
        spline_times.append(spline_time)
        
        lowess_mses.append(lowess_mse)
        gp_mses.append(gp_mse)
        # kalman_mses.append(kalman_mse)
        spline_mses.append(spline_mse)

    if plot:
        test_df = test_df.with_columns(pl.Series(name='timestamp', values=test_x))
        test_datetimes = test_df[time_column].to_list()

        # Plotting
        num_cols = min(3, n_samples)
        fig, axes = plt.subplots(num_cols, 1, figsize=(8, 2 * num_cols), sharex=True)
        if num_cols == 1:
            axes = [axes]
            
        for i in range(num_cols):
            axes[i].plot(test_datetimes, latent_user_stance[i], label='Latent User Stance', color='blue', linewidth=2)
            axes[i].fill_between(test_datetimes, latent_user_stance[i] - noise_scale, latent_user_stance[i] + noise_scale, color='blue', alpha=0.1)
            axes[i].scatter(all_base_datetimes[i], all_true_stances[i], label='True Stances', color='black', marker='x', s=50)
            axes[i].scatter(all_base_datetimes[i], all_observed_stances[i], label='Observed Stances', color='red', s=50)
            
            # Plot all methods
            axes[i].plot(test_datetimes, lowess_fits[i], label='LOWESS', color='green', alpha=0.8)
            axes[i].plot(test_datetimes, gp_fits[i], label='GP', color='orange', alpha=0.8)
            # axes[i].plot(test_datetimes, kalman_fits[i], label='Kalman Filter', color='purple', alpha=0.8)
            axes[i].plot(test_datetimes, spline_fits[i], label='Spline', color='brown', alpha=0.8)

            axes[i].fill_between(test_datetimes, gp_lowers[i], gp_uppers[i], color='orange', alpha=0.1)
            axes[i].set_title(f'Sample {i + 1}')
            # axes[i].legend()
            axes[i].set_ylabel('Stance')
            
            axes[i].xaxis.set_major_locator(mdates.YearLocator(month=1, day=1))
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Set minor ticks to months 1, 4, 7, 10 (but not January since that's covered by major)
            axes[i].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
            axes[i].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

            # Optional: Style the minor tick labels if needed
            axes[i].tick_params(which='minor', labelsize='small')

        axes[-1].set_xlabel('Time')
        # position legend slightly above the last subplot
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.2), fontsize='small')
        fig.tight_layout()
        fig.savefig('./figs/eval_gp.png', dpi=150, bbox_inches='tight')

    return {
        'lowess_times': lowess_times,
        'gp_times': gp_times,
        'spline_times': spline_times,
        'lowess_mses': lowess_mses,
        'gp_mses': gp_mses,
        'spline_mses': spline_mses,
    }

def run_parameter_sweep():
    """Run eval_gp across parameter grid and create separate plots."""
    np.random.seed(42)
    
    # Parameter ranges
    noise_scales = np.linspace(0.05, 0.5, 10)
    random_walk_scales = np.linspace(0.005, 0.05, 10)
    
    # Storage for results
    noise_results = {'noise_scales': [], 'lowess_mses': [], 'gp_mses': [], 'spline_mses': [],
                    'lowess_times': [], 'gp_times': [], 'spline_times': []}
    rw_results = {'rw_scales': [], 'lowess_mses': [], 'gp_mses': [], 'spline_mses': [],
                 'lowess_times': [], 'gp_times': [], 'spline_times': []}
    
    print("Running noise scale sweep...")
    # Noise scale sweep (fix random walk scale)
    for noise_scale in noise_scales:
        results = eval_gp(n_samples=5, noise_scale=noise_scale, random_walk_scale=0.01)
        
        noise_results['noise_scales'].append(noise_scale)
        noise_results['lowess_mses'].append(results['lowess_mses'])
        noise_results['gp_mses'].append(results['gp_mses'])
        noise_results['spline_mses'].append(results['spline_mses'])
        noise_results['lowess_times'].append(results['lowess_times'])
        noise_results['gp_times'].append(results['gp_times'])
        noise_results['spline_times'].append(results['spline_times'])
        
        print(f"Completed noise scale {noise_scale:.3f}")
    
    print("Running random walk scale sweep...")
    # Random walk scale sweep (fix noise scale)
    for rw_scale in random_walk_scales:
        results = eval_gp(n_samples=5, noise_scale=0.1, random_walk_scale=rw_scale)
        
        rw_results['rw_scales'].append(rw_scale)
        rw_results['lowess_mses'].append(results['lowess_mses'])
        rw_results['gp_mses'].append(results['gp_mses'])
        rw_results['spline_mses'].append(results['spline_mses'])
        rw_results['lowess_times'].append(results['lowess_times'])
        rw_results['gp_times'].append(results['gp_times'])
        rw_results['spline_times'].append(results['spline_times'])
        
        print(f"Completed random walk scale {rw_scale:.4f}")
    
    # Helper function to calculate mean and confidence intervals
    def calc_stats(data_list):
        means = [np.mean(data) for data in data_list]
        stds = [np.std(data) for data in data_list]
        return np.array(means), np.array(stds)
    
    # Plot 1: MSE vs Noise Scale
    plt.figure(figsize=(6, 2))
    
    lowess_means, lowess_stds = calc_stats(noise_results['lowess_mses'])
    gp_means, gp_stds = calc_stats(noise_results['gp_mses'])
    spline_means, spline_stds = calc_stats(noise_results['spline_mses'])
    
    plt.errorbar(noise_results['noise_scales'], lowess_means, yerr=lowess_stds, 
               label='LOWESS', marker='o', capsize=5)
    plt.errorbar(noise_results['noise_scales'], gp_means, yerr=gp_stds, 
               label='GP', marker='s', capsize=5)
    plt.errorbar(noise_results['noise_scales'], spline_means, yerr=spline_stds, 
               label='Spline', marker='^', capsize=5)
    
    plt.yscale('log')
    plt.xlabel('Noise Scale')
    plt.ylabel('MSE')
    # plt.title('MSE vs Noise Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figs/mse_vs_noise_scale.png', dpi=150, bbox_inches='tight')
    
    # Plot 2: Runtime Bar Chart (aggregated across all parameters)
    plt.figure(figsize=(4, 2))
    
    # Aggregate all runtime data across both parameter sweeps
    all_lowess_times = []
    all_gp_times = []
    all_spline_times = []
    
    # Add times from noise scale sweep
    for times_list in noise_results['lowess_times']:
        all_lowess_times.extend(times_list)
    for times_list in noise_results['gp_times']:
        all_gp_times.extend(times_list)
    for times_list in noise_results['spline_times']:
        all_spline_times.extend(times_list)
    
    # Add times from random walk scale sweep
    for times_list in rw_results['lowess_times']:
        all_lowess_times.extend(times_list)
    for times_list in rw_results['gp_times']:
        all_gp_times.extend(times_list)
    for times_list in rw_results['spline_times']:
        all_spline_times.extend(times_list)
    
    # Calculate aggregated statistics
    methods = ['LOWESS', 'GP', 'Spline']
    all_times = [all_lowess_times, all_gp_times, all_spline_times]
    quintiles = [np.quantile(times, [0.25, 0.5, 0.75]) for times in all_times]
    
    # Create bar chart
    x_pos = np.arange(len(methods))
    bars = plt.bar(x_pos, [q[1] for q in quintiles], yerr=[[q[1] - q[0] for q in quintiles], [q[2] - q[1] for q in quintiles]], capsize=5, 
                   color=['green', 'orange', 'brown'], alpha=0.7)
    plt.yscale('log')
    
    plt.xlabel('Method')
    plt.ylabel('Time (seconds)')
    # plt.title('Runtime Comparison (Aggregated)')
    plt.xticks(x_pos, methods)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('./figs/runtime_comparison.png', dpi=150, bbox_inches='tight')
    
    # Plot 3: MSE vs Random Walk Scale
    plt.figure(figsize=(6, 2))
    
    lowess_means, lowess_stds = calc_stats(rw_results['lowess_mses'])
    gp_means, gp_stds = calc_stats(rw_results['gp_mses'])
    spline_means, spline_stds = calc_stats(rw_results['spline_mses'])
    
    plt.errorbar(rw_results['rw_scales'], lowess_means, yerr=lowess_stds, 
               label='LOWESS', marker='o', capsize=5)
    plt.errorbar(rw_results['rw_scales'], gp_means, yerr=gp_stds, 
               label='GP', marker='s', capsize=5)
    plt.errorbar(rw_results['rw_scales'], spline_means, yerr=spline_stds, 
               label='Spline', marker='^', capsize=5)
    
    plt.yscale('log')
    plt.xlabel('Random Walk Scale')
    plt.ylabel('MSE')
    # plt.title('MSE vs Random Walk Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figs/mse_vs_rw_scale.png', dpi=150, bbox_inches='tight')
    
    
    print("Parameter sweep completed! Results saved to ./figs/")

    all_lowess_mses = np.concatenate(noise_results['lowess_mses'] + rw_results['lowess_mses'])
    all_gp_mses = np.concatenate(noise_results['gp_mses'] + rw_results['gp_mses'])
    all_spline_mses = np.concatenate(noise_results['spline_mses'] + rw_results['spline_mses'])

    # Calculate aggregated statistics
    methods = ['LOWESS', 'GP', 'Spline']
    all_mses = [all_lowess_mses, all_gp_mses, all_spline_mses]
    quintiles = [np.quantile(mses, [0.25, 0.5, 0.75]) for mses in all_mses]

    plt.figure(figsize=(4, 2))

    # Create bar chart
    x_pos = np.arange(len(methods))
    bars = plt.bar(x_pos, [q[1] for q in quintiles], yerr=[[q[1] - q[0] for q in quintiles], [q[2] - q[1] for q in quintiles]], capsize=5, 
                   color=['green', 'orange', 'brown'], alpha=0.7)
    plt.yscale('log')
    
    plt.xlabel('Method')
    plt.ylabel('MSE')
    # plt.title('Runtime Comparison (Aggregated)')
    plt.xticks(x_pos, methods)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('./figs/mse_comparison.png', dpi=150, bbox_inches='tight')

    print(f"LOWESS mean MSE: {np.mean(all_lowess_mses):.4f} ± {np.std(all_lowess_mses):.4f}")
    print(f"GP mean MSE: {np.mean(all_gp_mses):.4f} ± {np.std(all_gp_mses):.4f}")
    print(f"Spline mean MSE: {np.mean(all_spline_mses):.4f} ± {np.std(all_spline_mses):.4f}")

def main():
    np.random.seed(42)  # For reproducibility
    run_parameter_sweep()

if __name__ == '__main__':
    main()