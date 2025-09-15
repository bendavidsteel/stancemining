import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

def run_parameter_sweep():
    """Run eval_gp across parameter grid and create separate plots."""
    np.random.seed(42)
    
    # Parameter ranges
    noise_scales = np.linspace(0.05, 0.5, 10)
    random_walk_scales = np.linspace(0.005, 0.05, 10)
    
    
    rw_results = pl.read_parquet('./data/rw_sweep_results.parquet.zstd').to_dict(as_series=False)
    noise_results = pl.read_parquet('./data/noise_sweep_results.parquet.zstd').to_dict(as_series=False)

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

    print(f"LOWESS training time: {np.mean(all_lowess_times):.4f} ± {np.std(all_lowess_times):.4f} seconds")
    print(f"GP training time: {np.mean(all_gp_times):.4f} ± {np.std(all_gp_times):.4f} seconds")
    print(f"Spline training time: {np.mean(all_spline_times):.4f} ± {np.std(all_spline_times):.4f} seconds")

    # plot runtime and mse on scatter plot
    plt.figure(figsize=(2, 2))
    median_lowess_time = np.median(all_lowess_times)
    median_gp_time = np.median(all_gp_times)
    median_spline_time = np.median(all_spline_times)
    median_lowess_mse = np.median(all_lowess_mses)
    median_gp_mse = np.median(all_gp_mses)
    median_spline_mse = np.median(all_spline_mses)

    capsize = 4
    scatter_markersize = 32
    markersize = 8

    plt.scatter([median_lowess_time], [median_lowess_mse], label='LOWESS', marker='o', color='green', s=scatter_markersize)
    plt.scatter([median_gp_time], [median_gp_mse], label='GP', marker='s', color='orange', s=scatter_markersize)
    plt.scatter([median_spline_time], [median_spline_mse], label='Spline', marker='^', color='brown', s=scatter_markersize)
    plt.errorbar([np.quantile(all_lowess_times, 0.5)], [np.quantile(all_lowess_mses, 0.5)], 
                 xerr=[[np.quantile(all_lowess_times, 0.5) - np.quantile(all_lowess_times, 0.25)], 
                       [np.quantile(all_lowess_times, 0.75) - np.quantile(all_lowess_times, 0.5)]],
                 yerr=[[np.quantile(all_lowess_mses, 0.5) - np.quantile(all_lowess_mses, 0.25)], 
                       [np.quantile(all_lowess_mses, 0.75) - np.quantile(all_lowess_mses, 0.5)]],
                 marker='o', color='green', capsize=capsize, markersize=markersize)
    plt.errorbar([np.quantile(all_gp_times, 0.5)], [np.quantile(all_gp_mses, 0.5)], 
                 xerr=[[np.quantile(all_gp_times, 0.5) - np.quantile(all_gp_times, 0.25)], 
                       [np.quantile(all_gp_times, 0.75) - np.quantile(all_gp_times, 0.5)]],
                 yerr=[[np.quantile(all_gp_mses, 0.5) - np.quantile(all_gp_mses, 0.25)], 
                    [np.quantile(all_gp_mses, 0.75) - np.quantile(all_gp_mses, 0.5)]],
                 marker='s', color='orange', capsize=capsize, markersize=markersize)
    plt.errorbar([np.quantile(all_spline_times, 0.5)], [np.quantile(all_spline_mses, 0.5)], 
                 xerr=[[np.quantile(all_spline_times, 0.5) - np.quantile(all_spline_times, 0.25)], 
                       [np.quantile(all_spline_times, 0.75) - np.quantile(all_spline_times, 0.5)]],
                 yerr=[[np.quantile(all_spline_mses, 0.5) - np.quantile(all_spline_mses, 0.25)], 
                       [np.quantile(all_spline_mses, 0.75) - np.quantile(all_spline_mses, 0.5)]],
                 marker='^', color='brown', capsize=capsize, markersize=markersize)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Median Time (seconds)')
    plt.ylabel('Median MSE')
    plt.legend()
    plt.savefig('./figs/mse_vs_time.png', dpi=150, bbox_inches='tight')

def main():
    np.random.seed(42)  # For reproducibility
    run_parameter_sweep()

if __name__ == '__main__':
    main()