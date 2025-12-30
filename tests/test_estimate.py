import time

import numpy as np
import pytest

from stancemining.estimate import (
    bootstrap_kernelreg,
    bootstrap_kernelreg_gpu,
    GPU_AVAILABLE,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_test_data(n_samples=1000, n_test_points=100, seed=42):
    """Generate synthetic test data for kernel regression."""
    np.random.seed(seed)

    # Generate timestamps (sorted)
    timestamps = np.sort(np.random.uniform(0, 100, n_samples)).astype(np.float32)

    # Generate stance values with some underlying trend + noise
    true_trend = np.sin(timestamps * 0.1) * 0.5
    stance = (true_trend + np.random.normal(0, 0.3, n_samples)).astype(np.float32)
    stance = np.clip(stance, -1, 1)

    # Generate test points
    test_x = np.linspace(0, 100, n_test_points).astype(np.float32)

    return stance, timestamps, test_x


class TestBootstrapKernelregTiming:
    """Timing tests for bootstrap_kernelreg functions."""

    @pytest.fixture
    def small_data(self):
        """Small dataset for quick tests."""
        return generate_test_data(n_samples=500, n_test_points=50)

    @pytest.fixture
    def medium_data(self):
        """Medium dataset for performance tests."""
        return generate_test_data(n_samples=2000, n_test_points=100)

    @pytest.fixture
    def large_data(self):
        """Large dataset for stress tests."""
        return generate_test_data(n_samples=10000, n_test_points=200)

    def test_bootstrap_kernelreg_cpu_small(self, small_data):
        """Test CPU kernel regression with small dataset."""
        stance, timestamps, test_x = small_data
        bandwidth = 5.0
        n_bootstrap = 50

        start = time.perf_counter()
        result = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap)
        elapsed = time.perf_counter() - start

        print(f"\nCPU small (n={len(stance)}, bootstrap={n_bootstrap}): {elapsed:.4f}s")

        assert result.shape == (n_bootstrap, len(test_x))
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_bootstrap_kernelreg_cpu_medium(self, medium_data):
        """Test CPU kernel regression with medium dataset."""
        stance, timestamps, test_x = medium_data
        bandwidth = 5.0
        n_bootstrap = 100

        start = time.perf_counter()
        result = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap)
        elapsed = time.perf_counter() - start

        print(f"\nCPU medium (n={len(stance)}, bootstrap={n_bootstrap}): {elapsed:.4f}s")

        assert result.shape == (n_bootstrap, len(test_x))
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_bootstrap_kernelreg_cpu_large(self, large_data):
        """Test CPU kernel regression with large dataset."""
        stance, timestamps, test_x = large_data
        bandwidth = 5.0
        n_bootstrap = 100

        start = time.perf_counter()
        result = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap)
        elapsed = time.perf_counter() - start

        print(f"\nCPU large (n={len(stance)}, bootstrap={n_bootstrap}): {elapsed:.4f}s")

        assert result.shape == (n_bootstrap, len(test_x))
        assert np.all(result >= -1) and np.all(result <= 1)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_bootstrap_kernelreg_gpu_small(self, small_data):
        """Test GPU kernel regression with small dataset."""
        stance, timestamps, test_x = small_data
        bandwidth = 5.0
        n_bootstrap = 50

        # Warmup run
        _ = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, 10)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU small (n={len(stance)}, bootstrap={n_bootstrap}): {elapsed:.4f}s")

        result_np = result.cpu().numpy() if TORCH_AVAILABLE else result
        assert result_np.shape == (n_bootstrap, len(test_x))
        assert np.all(result_np >= -1) and np.all(result_np <= 1)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_bootstrap_kernelreg_gpu_medium(self, medium_data):
        """Test GPU kernel regression with medium dataset."""
        stance, timestamps, test_x = medium_data
        bandwidth = 5.0
        n_bootstrap = 100

        # Warmup run
        _ = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, 10)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU medium (n={len(stance)}, bootstrap={n_bootstrap}): {elapsed:.4f}s")

        result_np = result.cpu().numpy() if TORCH_AVAILABLE else result
        assert result_np.shape == (n_bootstrap, len(test_x))
        assert np.all(result_np >= -1) and np.all(result_np <= 1)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_bootstrap_kernelreg_gpu_large(self, large_data):
        """Test GPU kernel regression with large dataset."""
        stance, timestamps, test_x = large_data
        bandwidth = 5.0
        n_bootstrap = 100

        # Warmup run
        _ = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, 10)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU large (n={len(stance)}, bootstrap={n_bootstrap}): {elapsed:.4f}s")

        result_np = result.cpu().numpy() if TORCH_AVAILABLE else result
        assert result_np.shape == (n_bootstrap, len(test_x))
        assert np.all(result_np >= -1) and np.all(result_np <= 1)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_cpu_vs_gpu_comparison(self, medium_data):
        """Compare CPU and GPU implementations for correctness and timing."""
        stance, timestamps, test_x = medium_data
        bandwidth = 5.0

        # Use large n_bootstrap for convergence testing
        n_bootstrap_correctness = 1000
        n_bootstrap_timing = 100

        # --- Correctness testing with large bootstrap ---
        print(f"\n--- Correctness Test (n_bootstrap={n_bootstrap_correctness}) ---")

        result_cpu = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap_correctness)
        result_gpu = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap_correctness)
        torch.cuda.synchronize()
        result_gpu_np = result_gpu.cpu().numpy()

        # Test mean convergence
        cpu_mean = np.mean(result_cpu, axis=0)
        gpu_mean = np.mean(result_gpu_np, axis=0)
        mean_diff = np.abs(cpu_mean - gpu_mean)
        print(f"Mean: max diff = {mean_diff.max():.6f}, avg diff = {mean_diff.mean():.6f}")
        assert mean_diff.max() < 0.02, f"Mean max diff {mean_diff.max():.6f} exceeds threshold"

        # Test std convergence
        cpu_std = np.std(result_cpu, axis=0)
        gpu_std = np.std(result_gpu_np, axis=0)
        std_diff = np.abs(cpu_std - gpu_std)
        print(f"Std:  max diff = {std_diff.max():.6f}, avg diff = {std_diff.mean():.6f}")
        assert std_diff.max() < 0.02, f"Std max diff {std_diff.max():.6f} exceeds threshold"

        # Test percentile convergence
        cpu_p5 = np.percentile(result_cpu, 5, axis=0)
        cpu_p95 = np.percentile(result_cpu, 95, axis=0)
        gpu_p5 = np.percentile(result_gpu_np, 5, axis=0)
        gpu_p95 = np.percentile(result_gpu_np, 95, axis=0)
        p5_diff = np.abs(cpu_p5 - gpu_p5).max()
        p95_diff = np.abs(cpu_p95 - gpu_p95).max()
        print(f"5th percentile max diff: {p5_diff:.6f}")
        print(f"95th percentile max diff: {p95_diff:.6f}")
        assert p5_diff < 0.03, f"5th percentile diff {p5_diff:.6f} exceeds threshold"
        assert p95_diff < 0.03, f"95th percentile diff {p95_diff:.6f} exceeds threshold"

        # --- Timing comparison ---
        print(f"\n--- Timing Test (n_bootstrap={n_bootstrap_timing}) ---")

        start_cpu = time.perf_counter()
        result_cpu = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap_timing)
        elapsed_cpu = time.perf_counter() - start_cpu

        # Warmup GPU
        _ = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, 10)
        torch.cuda.synchronize()

        start_gpu = time.perf_counter()
        result_gpu = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap_timing)
        torch.cuda.synchronize()
        elapsed_gpu = time.perf_counter() - start_gpu

        print(f"CPU time: {elapsed_cpu:.4f}s")
        print(f"GPU time: {elapsed_gpu:.4f}s")
        print(f"Speedup: {elapsed_cpu / elapsed_gpu:.2f}x")

        assert result_cpu.shape == result_gpu.cpu().numpy().shape


class TestBootstrapKernelregScaling:
    """Test how the functions scale with different parameters."""

    @pytest.mark.parametrize("n_bootstrap", [10, 50, 100, 200])
    def test_cpu_bootstrap_scaling(self, n_bootstrap):
        """Test CPU scaling with number of bootstrap samples."""
        stance, timestamps, test_x = generate_test_data(n_samples=1000, n_test_points=50)
        bandwidth = 5.0

        start = time.perf_counter()
        result = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap)
        elapsed = time.perf_counter() - start

        print(f"\nCPU bootstrap={n_bootstrap}: {elapsed:.4f}s")
        assert result.shape[0] == n_bootstrap

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.parametrize("n_bootstrap", [10, 50, 100, 200])
    def test_gpu_bootstrap_scaling(self, n_bootstrap):
        """Test GPU scaling with number of bootstrap samples."""
        stance, timestamps, test_x = generate_test_data(n_samples=1000, n_test_points=50)
        bandwidth = 5.0

        # Warmup
        _ = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, 10)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU bootstrap={n_bootstrap}: {elapsed:.4f}s")
        assert result.shape[0] == n_bootstrap

    @pytest.mark.parametrize("n_samples", [500, 1000, 2000, 5000])
    def test_cpu_sample_scaling(self, n_samples):
        """Test CPU scaling with number of input samples."""
        stance, timestamps, test_x = generate_test_data(n_samples=n_samples, n_test_points=50)
        bandwidth = 5.0
        n_bootstrap = 50

        start = time.perf_counter()
        result = bootstrap_kernelreg(stance, timestamps, test_x, bandwidth, n_bootstrap)
        elapsed = time.perf_counter() - start

        print(f"\nCPU n_samples={n_samples}: {elapsed:.4f}s")
        assert result.shape == (n_bootstrap, len(test_x))

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.parametrize("n_samples", [500, 1000, 2000, 5000])
    def test_gpu_sample_scaling(self, n_samples):
        """Test GPU scaling with number of input samples."""
        stance, timestamps, test_x = generate_test_data(n_samples=n_samples, n_test_points=50)
        bandwidth = 5.0
        n_bootstrap = 50

        # Warmup
        _ = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, 10)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = bootstrap_kernelreg_gpu(stance, timestamps, test_x, bandwidth, n_bootstrap)
        if TORCH_AVAILABLE:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU n_samples={n_samples}: {elapsed:.4f}s")
        assert result.shape == (n_bootstrap, len(test_x))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
