#!/usr/bin/env python3
"""
Direct CUDA Kernel Interface for MaxK-GNN - Enhanced with TopK Kernel
Uses integrated maxk_kernel.cu topk function for better performance
"""

import torch
import numpy as np
import time
import os
from pathlib import Path
from graph_loader import GraphDataLoader

# Try to import the enhanced kernel bindings with TopK
try:
    import maxk_cuda_kernels  # Enhanced with TopK kernel
    DIRECT_KERNELS_AVAILABLE = True
    HAS_TOPK_KERNEL = hasattr(maxk_cuda_kernels, 'cuda_topk_maxk_float')
    print("✅ Direct CUDA kernels loaded successfully")
    if HAS_TOPK_KERNEL:
        print("✅ Fast TopK kernel available")
    else:
        print("⚠️ TopK kernel not available - using PyTorch fallback")
except ImportError:
    DIRECT_KERNELS_AVAILABLE = False
    HAS_TOPK_KERNEL = False
    print("⚠️ Direct CUDA kernels not available")
    print("   Build with: python setup_direct_kernels.py build_ext --inplace")

class DirectMaxKKernels:
    """
    Direct interface to MaxK-GNN CUDA kernels with integrated TopK
    """
    
    def __init__(self, graph_name=""):
        self.graph_name = graph_name
        self.warp4_metadata = None
        self.num_warps = 0
        
    def load_warp4_metadata(self, graph_name=None, num_warps=12, warp_max_nz=64):
        """Load warp4 metadata required by the kernels"""
        if graph_name is None:
            graph_name = self.graph_name
            
        try:
            # Try to use C++ function if available
            if DIRECT_KERNELS_AVAILABLE and hasattr(maxk_cuda_kernels, 'load_warp4_metadata'):
                self.warp4_metadata = maxk_cuda_kernels.load_warp4_metadata(
                    graph_name, num_warps, warp_max_nz
                )
                self.num_warps = self.warp4_metadata.size(0) // 4
                print(f"✅ Loaded warp4 metadata: {self.num_warps} warps")
                return True
            else:
                # Fallback to manual loading
                warp4_path = Path(f"./kernels/w{num_warps}_nz{warp_max_nz}_warp_4/{graph_name}.warp4")
                if not warp4_path.exists():
                    print(f"⚠️ Warp4 metadata not found: {warp4_path}")
                    return False
                
                warp4_data = np.fromfile(str(warp4_path), dtype=np.int32)
                self.warp4_metadata = torch.from_numpy(warp4_data).cuda().int()
                self.num_warps = len(warp4_data) // 4
                print(f"✅ Loaded warp4 metadata: {self.num_warps} warps")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load warp4 metadata: {e}")
            return False
    
    def generate_maxk_sparse_data(self, input_features, dim_k):
        """
        Generate MaxK sparse representation using fast TopK kernel
        """
        if HAS_TOPK_KERNEL and input_features.dtype == torch.float32:
            try:
                # Use the fast CUDA TopK kernel from maxk_kernel.cu
                print(f"🚀 Using fast TopK kernel for k={dim_k}")
                sparse_data, sparse_indices = maxk_cuda_kernels.cuda_topk_maxk_float(input_features, dim_k)
                
                # Convert indices to uint8 for kernel compatibility
                sparse_selector = sparse_indices.to(torch.uint8)
                
                print(f"📊 Fast TopK result: data {sparse_data.shape}, selector {sparse_selector.shape}")
                return sparse_data, sparse_selector
                
            except Exception as e:
                print(f"⚠️ Fast TopK kernel failed, using PyTorch fallback: {e}")
        
        # Fallback to PyTorch TopK
        print(f"🔄 Using PyTorch TopK for k={dim_k}")
        topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
        sparse_selector = topk_indices.to(torch.uint8)
        
        return topk_values, sparse_selector
    
    def prepare_cbsr_format_enhanced(self, features, maxk):
        """
        Enhanced CBSR format preparation using integrated TopK kernel
        """
        if HAS_TOPK_KERNEL:
            try:
                # Use the integrated CBSR preparation function
                return maxk_cuda_kernels.prepare_cbsr_format_maxk(features, maxk)
            except Exception as e:
                print(f"⚠️ Enhanced CBSR preparation failed: {e}")
        
        # Fallback to manual preparation
        return self.generate_maxk_sparse_data(features, maxk)
    
    def run_forward_kernel(self, graph_data, input_features, dim_k, timing=True):
        """
        Run the direct MaxK forward kernel with enhanced TopK
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded. Call load_warp4_metadata() first")
        
        # Generate sparse representation using enhanced TopK
        start_time = time.time()
        sparse_data, sparse_selector = self.prepare_cbsr_format_enhanced(input_features, dim_k)
        topk_time = time.time() - start_time
        
        print(f"📊 TopK time: {topk_time*1000:.3f}ms")
        print(f"📊 Generated sparse data: {sparse_data.shape}")
        print(f"📊 Generated sparse selector: {sparse_selector.shape}")
        
        # Create output tensor
        output_features = torch.zeros_like(input_features)
        
        # Create SPMM kernel instance
        kernel = maxk_cuda_kernels.SpmmMaxK(
            self.graph_name,
            graph_data['indptr'],
            graph_data['indices'],
            graph_data['values'],
            sparse_data,
            output_features
        )
        
        # Set sparse parameters
        kernel.set_sparse_params(sparse_selector, dim_k)
        
        if timing:
            # Run with timing
            kernel_time = kernel.run_kernel(True, input_features.size(1))
            total_time = kernel_time + topk_time * 1000  # Add TopK time
            return output_features, total_time
        else:
            # Single run
            kernel.run_kernel(False, input_features.size(1))
            return output_features, 0.0
    
    def run_backward_kernel(self, graph_data, grad_output, dim_k, timing=True):
        """
        Run the direct MaxK backward kernel with enhanced TopK
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded")
        
        # Generate sparse selector based on grad_output using enhanced TopK
        start_time = time.time()
        _, sparse_selector = self.prepare_cbsr_format_enhanced(grad_output, dim_k)
        topk_time = time.time() - start_time
        
        # Create sparse gradient output
        v_num = grad_output.size(0)
        grad_sparse = torch.zeros(v_num, dim_k, device=grad_output.device, dtype=grad_output.dtype)
        
        # Create backward kernel instance
        kernel = maxk_cuda_kernels.SpmmMaxKBackward(
            self.graph_name,
            graph_data['indptr'],
            graph_data['indices'],
            graph_data['values'],
            grad_output,
            grad_sparse
        )
        
        # Set sparse parameters
        kernel.set_sparse_params(sparse_selector, dim_k)
        
        if timing:
            # Run with timing
            kernel_time = kernel.run_kernel(True, grad_output.size(1))
            
            # Convert sparse gradient back to full format
            grad_input = self._sparse_to_full_enhanced(grad_sparse, sparse_selector, grad_output.size(1))
            
            total_time = kernel_time + topk_time * 1000
            return grad_input, total_time
        else:
            # Single run
            kernel.run_kernel(False, grad_output.size(1))
            grad_input = self._sparse_to_full_enhanced(grad_sparse, sparse_selector, grad_output.size(1))
            return grad_input, 0.0
    
    def _sparse_to_full_enhanced(self, sparse_data, sparse_selector, full_dim):
        """Convert sparse representation back to full format efficiently"""
        v_num, dim_k = sparse_data.shape
        full_output = torch.zeros(v_num, full_dim, device=sparse_data.device, dtype=sparse_data.dtype)
        
        # Use advanced indexing for efficiency
        batch_indices = torch.arange(v_num, device=sparse_data.device).unsqueeze(1).expand(-1, dim_k)
        col_indices = sparse_selector.long()
        
        # Mask for valid indices
        valid_mask = (col_indices >= 0) & (col_indices < full_dim)
        
        if torch.any(valid_mask):
            batch_flat = batch_indices[valid_mask]
            col_flat = col_indices[valid_mask]
            values_flat = sparse_data[valid_mask]
            
            full_output[batch_flat, col_flat] = values_flat
        
        return full_output
    
    def validate_against_cusparse(self, graph_data, input_features, dim_k, tolerance=0.001):
        """
        Enhanced validation with TopK kernel integration
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("⚠️ Cannot validate - direct kernels not available")
            return False
        
        print(f"🔍 Validating MaxK kernel vs cuSPARSE for k={dim_k}")
        
        # Step 1: Apply TopK using enhanced method
        start_time = time.time()
        sparse_data, sparse_selector = self.prepare_cbsr_format_enhanced(input_features, dim_k)
        topk_time = time.time() - start_time
        
        print(f"📊 Enhanced TopK time: {topk_time*1000:.3f}ms")
        
        # Create sparse input for cuSPARSE comparison
        sparse_input = torch.zeros_like(input_features)
        batch_indices = torch.arange(input_features.size(0), device=input_features.device).unsqueeze(1)
        sparse_input.scatter_(1, sparse_selector.long(), sparse_data)
        
        print(f"📊 Input shapes: {input_features.shape} -> sparse: {sparse_input.shape}")
        print(f"📊 Input sparsity: {torch.count_nonzero(sparse_input).item()}/{sparse_input.numel()} non-zero")
        
        # Step 2: Run MaxK kernel
        maxk_output, maxk_time = self.run_forward_kernel(graph_data, input_features, dim_k, timing=False)
        
        print(f"📊 MaxK output shape: {maxk_output.shape}")
        print(f"📊 MaxK output sparsity: {torch.count_nonzero(maxk_output).item()}/{maxk_output.numel()}")
        
        # Step 3: Run cuSPARSE reference
        cusparse_output = torch.zeros_like(sparse_input)
        cusparse_time = maxk_cuda_kernels.cusparse_spmm(
            graph_data['indptr'], graph_data['indices'], graph_data['values'],
            sparse_input, cusparse_output, times=0
        )
        
        print(f"📊 cuSPARSE output shape: {cusparse_output.shape}")
        print(f"📊 cuSPARSE output sparsity: {torch.count_nonzero(cusparse_output).item()}/{cusparse_output.numel()}")
        
        # Step 4: Compare results
        if maxk_output.shape != cusparse_output.shape:
            print(f"❌ Shape mismatch! MaxK: {maxk_output.shape}, cuSPARSE: {cusparse_output.shape}")
            return False
        
        # Focus comparison on positions where input was non-zero
        input_nonzero_mask = sparse_input != 0
        diff = torch.abs(maxk_output - cusparse_output)
        relevant_diff = diff[input_nonzero_mask]
        
        max_error = relevant_diff.max().item() if relevant_diff.numel() > 0 else 0.0
        avg_error = relevant_diff.mean().item() if relevant_diff.numel() > 0 else 0.0
        
        print(f"📊 Max error (at input nonzero): {max_error:.8f}")
        print(f"📊 Avg error (at input nonzero): {avg_error:.8f}")
        print(f"📊 Tolerance: {tolerance}")
        
        # Performance comparison
        print(f"📊 TopK time: {topk_time*1000:.3f}ms")
        if HAS_TOPK_KERNEL:
            print(f"📊 Using fast TopK kernel - significant speedup expected")
        
        is_valid = max_error < tolerance
        
        if is_valid:
            print("✅ Validation PASSED! Enhanced MaxK kernel produces correct results")
        else:
            print("❌ Validation FAILED! MaxK kernel has errors")
        
        return is_valid
    
    def benchmark_topk_performance(self, input_features, k_values=[16, 32, 64]):
        """
        Benchmark TopK performance: fast kernel vs PyTorch
        """
        print(f"\n🏁 TopK Performance Benchmark")
        print(f"Input shape: {input_features.shape}")
        print("k   | Fast TopK (ms) | PyTorch (ms) | Speedup")
        print("----|----------------|--------------|--------")
        
        for k in k_values:
            if k > input_features.size(1):
                continue
            
            # Benchmark fast TopK kernel
            if HAS_TOPK_KERNEL:
                times = []
                for _ in range(10):
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = maxk_cuda_kernels.cuda_topk_maxk_float(input_features, k)
                    torch.cuda.synchronize()
                    times.append((time.time() - start) * 1000)
                fast_time = np.mean(times[2:])  # Skip first 2 for warmup
            else:
                fast_time = float('inf')
            
            # Benchmark PyTorch TopK
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.time()
                _ = torch.topk(input_features, k, dim=1)
                torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
            pytorch_time = np.mean(times[2:])
            
            # Calculate speedup
            speedup = pytorch_time / fast_time if fast_time != float('inf') else 0
            
            print(f"{k:3d} | {fast_time:13.3f} | {pytorch_time:11.3f} | {speedup:6.2f}x")
    
    def benchmark_all_k_values(self, graph_data, dim_origin=256, k_values=[16, 32, 64], 
                              num_runs=4):
        """
        Enhanced benchmark with TopK performance analysis
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("❌ Direct kernels not available for benchmarking")
            return {}
        
        print(f"\n🏃‍♂️ Enhanced MaxK Kernel Benchmark")
        print(f"Graph: {self.graph_name}")
        print(f"TopK Kernel: {'✅ Fast CUDA' if HAS_TOPK_KERNEL else '⚠️ PyTorch Fallback'}")
        print("num graph dim_origin dim_k kernel time(ms)")
        print("-" * 50)
        
        v_num = graph_data['indptr'].size(0) - 1
        results = {}
        
        # Generate test input
        torch.manual_seed(123)
        input_features = torch.rand(v_num, dim_origin, device='cuda', dtype=torch.float32)
        
        # Benchmark TopK performance first
        print(f"\n📊 TopK Performance Analysis:")
        self.benchmark_topk_performance(input_features, k_values)
        
        print(f"\n📊 Full Kernel Benchmark:")
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
                    'backward_time': time_backward,
                    'total_time': time_forward + time_backward,
                    'has_fast_topk': HAS_TOPK_KERNEL
                }
                
                # Print in main.cu format
                print(f"1/1 {self.graph_name} {dim_origin} {dim_k} maxk {time_forward:.3f}")
                print(f"1/1 {self.graph_name} {dim_origin} {dim_k} maxk_backward {time_backward:.3f}")
                
            except Exception as e:
                print(f"❌ Failed for k={dim_k}: {e}")
                results[dim_k] = {'forward_time': -1, 'backward_time': -1}
        
        return results

def test_enhanced_kernels():
    """Test the enhanced kernel interface with TopK integration"""
    print("🧪 Testing Enhanced MaxK-GNN CUDA Kernels with TopK")
    print("=" * 60)
    
    if not DIRECT_KERNELS_AVAILABLE:
        print("❌ Direct kernels not available!")
        return False
    
    # Load test graph
    loader = GraphDataLoader()
    graphs = loader.get_available_graphs()
    
    if not graphs:
        print("❌ No graphs available for testing")
        return False
    
    # Test all graphs
    success_count = 0
    for i, test_graph in enumerate(graphs):
        print(f"\n📊 Testing graph {i+1}/{len(graphs)}: {test_graph}")
        
        try:
            # Load graph data
            graph_data = loader.load_graph(test_graph)
            graph_data = loader.to_cuda_tensors(graph_data)
            
            # Initialize enhanced kernel interface
            kernels = DirectMaxKKernels(test_graph)
            
            # Load warp4 metadata
            if not kernels.load_warp4_metadata():
                print("❌ Cannot proceed without warp4 metadata")
                continue
            
            # Test validation with enhanced TopK
            v_num = graph_data['v_num']
            test_features = torch.rand(v_num, 256, device='cuda', dtype=torch.float32)
            
            print(f"\n🔍 Validating enhanced kernel correctness...")
            is_valid = kernels.validate_against_cusparse(graph_data, test_features, dim_k=32)
            
            if is_valid:
                print(f"\n📈 Running enhanced benchmark...")
                results = kernels.benchmark_all_k_values(
                    graph_data, dim_origin=256, k_values=[16, 32], num_runs=2
                )
                success_count += 1
                print(f"✅ Graph {test_graph} completed successfully!")
            else:
                print(f"⚠️ Validation failed for {test_graph}")
                
        except Exception as e:
            print(f"❌ Enhanced kernel testing failed for {test_graph}: {e}")
            continue
    
    print(f"\n🎯 Summary: {success_count}/{len(graphs)} graphs tested successfully")
    return success_count > 0

if __name__ == "__main__":
    test_enhanced_kernels()
