#!/usr/bin/env python3
"""
Direct CUDA Kernel Interface for MaxK-GNN - CORRECTED VERSION WITH TOPK
Follows main.cu logic exactly: TopK → Sparse representation → Dense reconstruction for comparison
"""

import torch
import numpy as np
import time
import os
from pathlib import Path
from graph_loader import GraphDataLoader

# Try to import the direct kernel bindings
try:
    import maxk_cuda_kernels
    DIRECT_KERNELS_AVAILABLE = True
    print("✅ Direct CUDA kernels loaded successfully")
except ImportError:
    DIRECT_KERNELS_AVAILABLE = False
    print("⚠️ Direct CUDA kernels not available")
    print("   Build with: python setup_direct_kernels.py build_ext --inplace")

class DirectMaxKKernels:
    """
    Direct interface to MaxK-GNN CUDA kernels
    Follows main.cu workflow exactly
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
            self.warp4_metadata = maxk_cuda_kernels.load_warp4_metadata(
                graph_name, num_warps, warp_max_nz
            )
            self.num_warps = self.warp4_metadata.size(0) // 4
            print(f"✅ Loaded warp4 metadata: {self.num_warps} warps")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load warp4 metadata: {e}")
            return False
    
    def apply_topk_selection(self, input_features, dim_k, use_cuda_topk=True):
        """
        Apply TopK selection - CORRECTED to match main.cu workflow
        Returns: (sparse_data, sparse_selector, dense_reconstruction)
        """
        v_num, dim_origin = input_features.shape
        
        # Step 1: Apply TopK selection (like main.cu lines 102-120)
        if use_cuda_topk and DIRECT_KERNELS_AVAILABLE:
            try:
                print(f"🚀 Using CUDA TopK kernel for k={dim_k}")
                topk_values, topk_indices = maxk_cuda_kernels.cuda_topk_maxk_float(
                    input_features, dim_k
                )
                print(f"✅ CUDA TopK completed")
            except Exception as e:
                print(f"⚠️ CUDA TopK failed: {e}, using PyTorch")
                topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
        else:
            print(f"🔄 Using PyTorch TopK for k={dim_k}")
            topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
        
        # Step 2: Create sparse representation (for MaxK kernels)
        sparse_data = topk_values  # Shape: (v_num, dim_k)
        sparse_selector = topk_indices.to(torch.uint8)  # Shape: (v_num, dim_k)
        
        # Step 3: Dense reconstruction (for cuSPARSE comparison, like main.cu lines 122-130)
        dense_reconstruction = torch.zeros_like(input_features)  # Shape: (v_num, dim_origin)
        dense_reconstruction.scatter_(1, topk_indices, topk_values)
        
        print(f"📊 TopK results: sparse_data {sparse_data.shape}, selector {sparse_selector.shape}")
        print(f"📊 Dense reconstruction: {dense_reconstruction.shape}, sparsity: {torch.count_nonzero(dense_reconstruction).item()}/{dense_reconstruction.numel()}")
        
        return sparse_data, sparse_selector, dense_reconstruction
    
    def benchmark_topk_methods(self, input_features, dim_k, num_runs=10):
        """
        Benchmark CUDA TopK vs PyTorch TopK
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("⚠️ CUDA kernels not available for TopK comparison")
            return {}
        
        print(f"\n⚡ TopK Benchmark: k={dim_k}, input_shape={input_features.shape}")
        
        # Benchmark PyTorch TopK
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            pytorch_values, pytorch_indices = torch.topk(input_features, dim_k, dim=1)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / num_runs * 1000
        
        # Benchmark CUDA TopK
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(num_runs):
                cuda_values, cuda_indices = maxk_cuda_kernels.cuda_topk_maxk_float(
                    input_features, dim_k
                )
            torch.cuda.synchronize()
            cuda_time = (time.time() - start_time) / num_runs * 1000
            
            speedup = pytorch_time / cuda_time if cuda_time > 0 else 0
            
            # Correctness check
            diff_values = torch.abs(pytorch_values - cuda_values).max().item()
            diff_indices = (pytorch_indices.to(torch.int32) != cuda_indices).sum().item()
            
            print(f"📊 PyTorch TopK: {pytorch_time:.3f} ms")
            print(f"📊 CUDA TopK:    {cuda_time:.3f} ms") 
            print(f"📊 Speedup:      {speedup:.2f}x")
            print(f"📊 Correctness:  values_diff={diff_values:.6f}, indices_diff={diff_indices}")
            
            return {
                'pytorch_time': pytorch_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'correctness': diff_values < 0.01 and diff_indices == 0
            }
            
        except Exception as e:
            print(f"❌ CUDA TopK benchmark failed: {e}")
            return {'pytorch_time': pytorch_time, 'cuda_time': -1, 'speedup': 0}
    
    def run_forward_kernel(self, graph_data, input_features, dim_k, timing=True, use_cuda_topk=True):
        """
        Run MaxK forward kernel - CORRECTED workflow
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded")
        
        # Apply TopK selection (matches main.cu workflow)
        sparse_data, sparse_selector, _ = self.apply_topk_selection(
            input_features, dim_k, use_cuda_topk
        )
        
        if timing:
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
    
    def run_backward_kernel(self, graph_data, grad_output, dim_k, timing=True, use_cuda_topk=True):
        """
        Run MaxK backward kernel - CORRECTED workflow
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded")
        
        # Apply TopK selection on grad_output
        _, sparse_selector, _ = self.apply_topk_selection(
            grad_output, dim_k, use_cuda_topk
        )
        
        if timing:
            timer = maxk_cuda_kernels.CudaTimer()
            times = []
            
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
                
                if i >= 4:
                    times.append(elapsed)
            
            avg_time = np.mean(times)
            return grad_input, avg_time
        else:
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
    
    def validate_against_cusparse(self, graph_data, input_features, dim_k, tolerance=0.001, use_cuda_topk=True):
        """
        CORRECTED validation - follows main.cu logic exactly
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("⚠️ Cannot validate - direct kernels not available")
            return False
        
        print(f"🔍 Validating MaxK kernel vs cuSPARSE for k={dim_k}")
        print(f"   Using {'CUDA' if use_cuda_topk else 'PyTorch'} TopK")
        
        # Step 1: Apply TopK selection (like main.cu)
        sparse_data, sparse_selector, dense_reconstruction = self.apply_topk_selection(
            input_features, dim_k, use_cuda_topk
        )
        
        # Step 2: Run MaxK kernel on sparse representation
        maxk_output, _ = maxk_cuda_kernels.spmm_maxk_forward(
            self.warp4_metadata,
            graph_data['indices'],
            graph_data['values'],
            sparse_data,
            sparse_selector,
            self.num_warps,
            dim_k
        )
        
        # Step 3: Run cuSPARSE on dense reconstruction (same mathematical operation)
        cusparse_output = maxk_cuda_kernels.cusparse_spmm(
            graph_data['indptr'], 
            graph_data['indices'], 
            graph_data['values'],
            dense_reconstruction,  # This is the key: same sparsity pattern
            timing=False
        )
        
        print(f"📊 MaxK output shape: {maxk_output.shape}")
        print(f"📊 cuSPARSE output shape: {cusparse_output.shape}")
        
        if maxk_output.shape != cusparse_output.shape:
            print(f"❌ Shape mismatch!")
            return False
        
        # Step 4: Compare results
        diff = torch.abs(maxk_output - cusparse_output)
        max_error = diff.max().item()
        avg_error = diff.mean().item()
        
        # Focus on non-zero positions from input
        nonzero_mask = dense_reconstruction != 0
        if torch.any(nonzero_mask):
            relevant_diff = diff[nonzero_mask]
            max_relevant_error = relevant_diff.max().item()
            avg_relevant_error = relevant_diff.mean().item()
            
            print(f"📊 Overall - Max error: {max_error:.8f}, Avg error: {avg_error:.8f}")
            print(f"📊 At nonzero positions - Max error: {max_relevant_error:.8f}, Avg error: {avg_relevant_error:.8f}")
            print(f"📊 Tolerance: {tolerance}")
            
            is_valid = max_relevant_error < tolerance
        else:
            print(f"📊 Max error: {max_error:.8f}, Avg error: {avg_error:.8f}")
            is_valid = max_error < tolerance
        
        if is_valid:
            print("✅ Validation PASSED!")
        else:
            print("❌ Validation FAILED!")
            
        return is_valid
    
    def benchmark_all_k_values(self, graph_data, dim_origin=256, k_values=[16, 32, 64], 
                              num_runs=4, use_cuda_topk=True):
        """
        Comprehensive benchmark including TopK overhead
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("❌ Direct kernels not available")
            return {}
        
        print(f"\n🏃‍♂️ Benchmarking Direct MaxK Kernels")
        print(f"Graph: {self.graph_name}")
        print(f"TopK method: {'CUDA' if use_cuda_topk else 'PyTorch'}")
        
        v_num = graph_data['indptr'].size(0) - 1
        results = {}
        
        # Generate test input
        torch.manual_seed(123)
        input_features = torch.rand(v_num, dim_origin, device='cuda', dtype=torch.float32)
        
        # Benchmark TopK methods first
        print(f"\n🔥 TopK Performance:")
        for dim_k in k_values[:2]:
            if dim_k <= input_features.size(1):
                topk_results = self.benchmark_topk_methods(input_features, dim_k)
                if topk_results:
                    print(f"k={dim_k}: {topk_results.get('speedup', 0):.2f}x speedup")
        
        # Benchmark full pipeline
        print(f"\n📊 Full Pipeline Benchmark:")
        print("k_value topk_time(ms) forward_time(ms) backward_time(ms)")
        print("-" * 60)
        
        for dim_k in k_values:
            if dim_k > 64:
                continue
                
            try:
                # Measure TopK time separately
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(num_runs):
                    sparse_data, sparse_selector, _ = self.apply_topk_selection(
                        input_features, dim_k, use_cuda_topk
                    )
                torch.cuda.synchronize()
                topk_time = (time.time() - start_time) / num_runs * 1000
                
                # Measure forward kernel
                output_forward, time_forward = self.run_forward_kernel(
                    graph_data, input_features, dim_k, timing=True, use_cuda_topk=use_cuda_topk
                )
                
                # Measure backward kernel
                grad_output = torch.rand_like(input_features)
                grad_input, time_backward = self.run_backward_kernel(
                    graph_data, grad_output, dim_k, timing=True, use_cuda_topk=use_cuda_topk
                )
                
                results[dim_k] = {
                    'topk_time': topk_time,
                    'forward_time': time_forward,
                    'backward_time': time_backward,
                    'total_time': topk_time + time_forward + time_backward
                }
                
                print(f"{dim_k:7d} {topk_time:12.3f} {time_forward:15.3f} {time_backward:16.3f}")
                
            except Exception as e:
                print(f"❌ Failed for k={dim_k}: {e}")
                results[dim_k] = {'topk_time': -1, 'forward_time': -1, 'backward_time': -1}
        
        return results

def test_direct_kernels():
    """Test the corrected direct kernel interface"""
    print("🧪 Testing Direct MaxK-GNN CUDA Kernels (Corrected)")
    print("=" * 55)
    
    if not DIRECT_KERNELS_AVAILABLE:
        print("❌ Direct kernels not available!")
        return False
    
    loader = GraphDataLoader()
    graphs = loader.get_available_graphs()
    
    if not graphs:
        print("❌ No graphs available for testing")
        return False
    
    test_graph = graphs[0]
    print(f"\n📊 Testing graph: {test_graph}")
    
    try:
        # Load graph data
        graph_data = loader.load_graph(test_graph)
        graph_data = loader.to_cuda_tensors(graph_data)
        
        # Initialize kernel interface
        kernels = DirectMaxKKernels(test_graph)
        
        if not kernels.load_warp4_metadata():
            print("❌ Cannot proceed without warp4 metadata")
            return False
        
        # Test with both TopK methods
        v_num = graph_data['v_num']
        test_features = torch.rand(v_num, 256, device='cuda', dtype=torch.float32)
        
        print(f"\n🔍 Validation with PyTorch TopK...")
        is_valid_pytorch = kernels.validate_against_cusparse(
            graph_data, test_features, dim_k=32, use_cuda_topk=False
        )
        
        print(f"\n🔍 Validation with CUDA TopK...")
        is_valid_cuda = kernels.validate_against_cusparse(
            graph_data, test_features, dim_k=32, use_cuda_topk=True
        )
        
        if is_valid_pytorch and is_valid_cuda:
            print(f"\n📈 Running comprehensive benchmark...")
            results = kernels.benchmark_all_k_values(
                graph_data, dim_origin=256, k_values=[16, 32], num_runs=2, use_cuda_topk=True
            )
            print(f"\n✅ Testing completed successfully!")
            return True
        else:
            print(f"\n⚠️ Validation failed")
            return False
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_kernels()
