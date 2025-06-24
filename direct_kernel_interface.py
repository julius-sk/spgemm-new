#!/usr/bin/env python3
"""
Direct CUDA Kernel Interface for MaxK-GNN - FIXED VERSION
Uses compiled bindings to call spmm_maxk.cu and spmm_maxk_backward.cu kernels directly
"""

import torch
import numpy as np
import time
import os
from pathlib import Path
from graph_loader import GraphDataLoader

# Try to import the direct kernel bindings
try:
    import maxk_cuda_kernels  # This will be our compiled extension
    DIRECT_KERNELS_AVAILABLE = True
    print("✅ Direct CUDA kernels loaded successfully")
except ImportError:
    DIRECT_KERNELS_AVAILABLE = False
    print("⚠️ Direct CUDA kernels not available")
    print("   Build with: python setup_direct_kernels.py build_ext --inplace")

class DirectMaxKKernels:
    """
    Direct interface to MaxK-GNN CUDA kernels
    Calls the actual spmm_maxk.cu and spmm_maxk_backward.cu functions
    """
    
    def __init__(self, graph_name=""):
        self.graph_name = graph_name
        self.warp4_metadata = None
        self.num_warps = 0
        
    def load_warp4_metadata(self, graph_name=None, num_warps=12, warp_max_nz=64):
        """Load warp4 metadata required by the kernels"""
        if graph_name is None:
            graph_name = self.graph_name
            
        if not DIRECT_KERNELS_AVAILABLE:
            print("⚠️ Direct kernels not available for metadata loading")
            return False
            
        try:
            # Use the C++ function to load metadata
            self.warp4_metadata = maxk_cuda_kernels.load_warp4_metadata(
                graph_name, num_warps, warp_max_nz
            )
            self.num_warps = self.warp4_metadata.size(0) // 4
            print(f"✅ Loaded warp4 metadata: {self.num_warps} warps")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load warp4 metadata: {e}")
            print(f"   Ensure generate_meta.py has been run for {graph_name}")
            return False
    
    def generate_maxk_sparse_data(self, input_features, dim_k):
        """
        Generate MaxK sparse representation (replicates main.cu logic)
        Does TopK selection to create sparse input
        """
        v_num, dim_origin = input_features.shape
        
        # Apply TopK to get sparse representation
        topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
        
        # Create sparse data and selector (matching main.cu format)
        sparse_data = topk_values  # Shape: (v_num, dim_k)
        sparse_selector = topk_indices.to(torch.uint8)  # Shape: (v_num, dim_k)
        
        return sparse_data, sparse_selector
    
    def run_forward_kernel(self, graph_data, input_features, dim_k, timing=True):
        """
        Run the direct MaxK forward kernel (spmm_maxk.cu)
        
        Args:
            graph_data: Dict with 'indices' and 'values' tensors
            input_features: Dense input tensor (v_num x dim_origin)  
            dim_k: Sparse dimension (k value)
            timing: Whether to measure execution time
            
        Returns:
            (output_tensor, execution_time_ms)
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded. Call load_warp4_metadata() first")
        
        # Generate sparse representation using TopK
        sparse_data, sparse_selector = self.generate_maxk_sparse_data(input_features, dim_k)
        
        if timing:
            # Use the built-in timing function
            times = maxk_cuda_kernels.benchmark_spmm_maxk(
                self.warp4_metadata,
                graph_data['indices'],
                graph_data['values'],
                sparse_data,
                sparse_selector,
                self.num_warps,
                dim_k,
                num_runs=4
            )
            avg_time = np.mean(times)
            
            # Get the result from the last run
            output = maxk_cuda_kernels.spmm_maxk_forward(
                self.warp4_metadata,
                graph_data['indices'], 
                graph_data['values'],
                sparse_data,
                sparse_selector,
                self.num_warps,
                dim_k
            )
            
            return output, avg_time
        else:
            # Single run without timing
            output = maxk_cuda_kernels.spmm_maxk_forward(
                self.warp4_metadata,
                graph_data['indices'],
                graph_data['values'],
                sparse_data,
                sparse_selector,
                self.num_warps,
                dim_k
            )
            return output, 0.0
    
    def run_backward_kernel(self, graph_data, grad_output, dim_k, timing=True):
        """
        Run the direct MaxK backward kernel (spmm_maxk_backward.cu)
        
        Args:
            graph_data: Dict with 'indices' and 'values' tensors
            grad_output: Gradient tensor from next layer
            dim_k: Sparse dimension (k value)
            timing: Whether to measure execution time
            
        Returns:
            (grad_input_tensor, execution_time_ms)
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded")
        
        # Generate sparse selector based on grad_output
        _, sparse_selector = self.generate_maxk_sparse_data(grad_output, dim_k)
        
        if timing:
            # Measure timing manually for backward kernel
            timer = maxk_cuda_kernels.CudaTimer()
            times = []
            
            # Warmup + timing runs
            for i in range(8):  # 4 warmup + 4 timing
                timer.start()
                grad_input = maxk_cuda_kernels.spmm_maxk_backward(
                    self.warp4_metadata,
                    graph_data['indices'],
                    graph_data['values'],
                    grad_output,
                    sparse_selector,
                    self.num_warps,
                    dim_k
                )
                elapsed = timer.stop()
                
                if i >= 4:  # Only count timing runs
                    times.append(elapsed)
            
            avg_time = np.mean(times)
            return grad_input, avg_time
        else:
            # Single run
            grad_input = maxk_cuda_kernels.spmm_maxk_backward(
                self.warp4_metadata,
                graph_data['indices'],
                graph_data['values'],
                grad_output,
                sparse_selector,
                self.num_warps,
                dim_k
            )
            return grad_input, 0.0
    
def validate_against_cusparse(self, graph_data, input_features, dim_k, tolerance=0.001):
    """
    FIXED: Handle dimension mismatch between sparse and full outputs
    """
    if not DIRECT_KERNELS_AVAILABLE:
        print("⚠️ Cannot validate - direct kernels not available")
        return False
    
    print(f"🔍 Validating MaxK kernel vs cuSPARSE for k={dim_k}")
    
    # Step 1: Create TopK sparse input
    topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
    sparse_input_full = torch.zeros_like(input_features)
    sparse_input_full.scatter_(1, topk_indices, topk_values)
    
    print(f"📊 Input shapes:")
    print(f"   input_features: {input_features.shape}")
    print(f"   topk_values: {topk_values.shape}")
    print(f"   topk_indices: {topk_indices.shape}")
    print(f"   sparse_input_full: {sparse_input_full.shape}")
    
    # Step 2: Run MaxK kernel - returns SPARSE output (v_num, dim_k)
    maxk_sparse_output, _ = self.run_forward_kernel(graph_data, input_features, dim_k, timing=False)
    
    print(f"📊 MaxK kernel output:")
    print(f"   maxk_sparse_output shape: {maxk_sparse_output.shape}")
    print(f"   maxk_sparse_output[0,:5]: {maxk_sparse_output[0,:5]}")
    print(f"   maxk_sparse_output range: [{maxk_sparse_output.min():.6f}, {maxk_sparse_output.max():.6f}]")
    print(f"   maxk_sparse_output non-zero: {torch.count_nonzero(maxk_sparse_output).item()}/{maxk_sparse_output.numel()}")
    
    # Step 3: Convert MaxK sparse output back to full dimensions
    maxk_full_output = torch.zeros_like(input_features)
    for i in range(input_features.shape[0]):
        for j in range(dim_k):
            idx = topk_indices[i, j].item()
            maxk_full_output[i, idx] = maxk_sparse_output[i, j]
    
    print(f"📊 MaxK full output:")
    print(f"   maxk_full_output shape: {maxk_full_output.shape}")
    print(f"   maxk_full_output[0,:10]: {maxk_full_output[0,:10]}")
    print(f"   maxk_full_output non-zero: {torch.count_nonzero(maxk_full_output).item()}/{maxk_full_output.numel()}")
    
    # Step 4: Run cuSPARSE reference 
    v_num = input_features.shape[0]
    edge_index = torch.stack([
        torch.arange(v_num, device='cuda').repeat_interleave(
            graph_data['indptr'][1:] - graph_data['indptr'][:-1]
        ),
        graph_data['indices']
    ])
    
    sparse_adj = torch.sparse_coo_tensor(
        edge_index, graph_data['values'], (v_num, v_num)
    ).coalesce()
    
    cusparse_output = torch.sparse.mm(sparse_adj, sparse_input_full)
    
    print(f"📊 cuSPARSE output:")
    print(f"   cusparse_output shape: {cusparse_output.shape}")
    print(f"   cusparse_output[0,:10]: {cusparse_output[0,:10]}")
    print(f"   cusparse_output range: [{cusparse_output.min():.6f}, {cusparse_output.max():.6f}]")
    print(f"   cusparse_output non-zero: {torch.count_nonzero(cusparse_output).item()}/{cusparse_output.numel()}")
    
    # Step 5: Now both are (v_num, 256) - compare them
    error = torch.abs(maxk_full_output - cusparse_output).max().item()
    avg_error = torch.abs(maxk_full_output - cusparse_output).mean().item()
    
    print(f"📊 Comparison:")
    print(f"   Max error: {error:.8f}")
    print(f"   Avg error: {avg_error:.8f}")
    print(f"   Tolerance: {tolerance}")
    
    # Show first few differences
    diff = torch.abs(maxk_full_output - cusparse_output)
    print(f"   First 10 differences: {diff[0,:10]}")
    
    is_valid = error < tolerance
    
    if is_valid:
        print("✅ Validation PASSED!")
    else:
        print("❌ Validation FAILED!")
        
    return is_valid
    
    def benchmark_all_k_values(self, graph_data, dim_origin=256, k_values=[16, 32, 64], 
                              num_runs=4):
        """
        Benchmark across different k values (replicates main.cu benchmark loop)
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("❌ Direct kernels not available for benchmarking")
            return {}
        
        print(f"\n🏃‍♂️ Benchmarking Direct MaxK Kernels")
        print(f"Graph: {self.graph_name}")
        print("num graph dim_origin dim_k kernel time(ms)")
        print("-" * 50)
        
        v_num = graph_data['indptr'].size(0) - 1
        results = {}
        
        # Generate test input (same seed as main.cu)
        torch.manual_seed(123)
        input_features = torch.rand(v_num, dim_origin, device='cuda', dtype=torch.float32)
        
        for dim_k in k_values:
            if dim_k > 64:  # Skip if exceeds limit
                print(f"⏭️  Skipping k={dim_k} (exceeds limit)")
                continue
                
            print(f"\n📊 Testing k = {dim_k}")
            
            try:
                # Forward kernel
                output_forward, time_forward = self.run_forward_kernel(
                    graph_data, input_features, dim_k, timing=True
                )
                
                # Backward kernel  
                grad_output = torch.rand_like(input_features)
                grad_input, time_backward = self.run_backward_kernel(
                    graph_data, grad_output, dim_k, timing=True
                )
                
                # Store results
                results[dim_k] = {
                    'forward_time': time_forward,
                    'backward_time': time_backward
                }
                
                # Print in main.cu format
                print(f"1/1 {self.graph_name} {dim_origin} {dim_k} maxk {time_forward:.3f}")
                print(f"1/1 {self.graph_name} {dim_origin} {dim_k} maxk_backward {time_backward:.3f}")
                
            except Exception as e:
                print(f"❌ Failed for k={dim_k}: {e}")
                results[dim_k] = {'forward_time': -1, 'backward_time': -1}
        
        return results

def test_direct_kernels():
    """Test the direct kernel interface"""
    print("🧪 Testing Direct MaxK-GNN CUDA Kernels")
    print("=" * 50)
    
    if not DIRECT_KERNELS_AVAILABLE:
        print("❌ Direct kernels not available!")
        print("   Build with: python setup_direct_kernels.py build_ext --inplace")
        return False
    
    # Load test graph
    loader = GraphDataLoader()
    graphs = loader.get_available_graphs()
    
    if not graphs:
        print("❌ No graphs available for testing")
        return False
    
    test_graph = graphs[0]
    print(f"📊 Testing with graph: {test_graph}")
    
    try:
        # Load graph data
        graph_data = loader.load_graph(test_graph)
        graph_data = loader.to_cuda_tensors(graph_data)
        
        # Initialize direct kernel interface
        kernels = DirectMaxKKernels(test_graph)
        
        # Load warp4 metadata
        if not kernels.load_warp4_metadata():
            print("❌ Cannot proceed without warp4 metadata")
            return False
        
        # Test validation with proper logic
        v_num = graph_data['v_num']
        test_features = torch.rand(v_num, 256, device='cuda', dtype=torch.float32)
        
        print(f"\n🔍 Validating kernel correctness...")
        is_valid = kernels.validate_against_cusparse(graph_data, test_features, dim_k=32)
        
        if not is_valid:
            print("⚠️ Validation failed - results may be incorrect")
        
        # Run benchmark only if validation passes
        if is_valid:
            print(f"\n📈 Running benchmark...")
            results = kernels.benchmark_all_k_values(
                graph_data, dim_origin=256, k_values=[16, 32], num_runs=2
            )
            
            print(f"\n✅ Direct kernel testing completed successfully!")
        else:
            print(f"\n⚠️ Skipping benchmark due to validation failure")
            
        return is_valid
        
    except Exception as e:
        print(f"❌ Direct kernel testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_kernels()
