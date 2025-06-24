First, add the cuSPARSE kernel to your bindings. Update your cuda_kernel_bindings.cpp:
cpp// Add this include at the top
#include "spmm_cusparse.h"

// Add this function declaration in the binding section
extern "C" {
    // Declare the cuSPARSE function from spmm_cusparse.cu
    double spmm_cusparse(int *ptr, int *idx, float *val, float *vin, float *vout, 
                        int num_v, int num_e, int dim, int times);
}

// Add this wrapper function
torch::Tensor cusparse_spmm_wrapper(
    torch::Tensor indptr,
    torch::Tensor indices, 
    torch::Tensor values,
    torch::Tensor input_features,
    bool timing = false) {
    
    TORCH_CHECK(indptr.is_cuda() && indptr.is_contiguous(), "indptr must be CUDA and contiguous");
    TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be CUDA and contiguous");
    TORCH_CHECK(values.is_cuda() && values.is_contiguous(), "values must be CUDA and contiguous");
    TORCH_CHECK(input_features.is_cuda() && input_features.is_contiguous(), "input_features must be CUDA and contiguous");
    
    int num_v = indptr.size(0) - 1;
    int num_e = indices.size(0);
    int dim = input_features.size(1);
    
    // Create output tensor
    auto output = torch::zeros_like(input_features);
    
    // Call the actual cuSPARSE kernel
    double exec_time = spmm_cusparse(
        indptr.data_ptr<int>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        input_features.data_ptr<float>(),
        output.data_ptr<float>(),
        num_v, num_e, dim,
        timing ? 10 : 0  // number of timing runs
    );
    
    return output;
}

// Add to the PyBind11 module section:
m.def("cusparse_spmm", &cusparse_spmm_wrapper, "cuSPARSE SpMM reference",
      py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("input_features"), py::arg("timing") = false);
Then update your direct_kernel_interface.py validation function:
pythondef validate_against_cusparse(self, graph_data, input_features, dim_k, tolerance=0.001):
    """
    FIXED: Use actual cuSPARSE kernel instead of slow torch.sparse.mm
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
    print(f"   sparse_input_full: {sparse_input_full.shape}")
    
    # Step 2: Run MaxK kernel
    maxk_sparse_output, _ = self.run_forward_kernel(graph_data, input_features, dim_k, timing=False)
    
    print(f"📊 MaxK kernel output:")
    print(f"   maxk_sparse_output shape: {maxk_sparse_output.shape}")
    print(f"   maxk_sparse_output[0,:5]: {maxk_sparse_output[0,:5]}")
    
    # Step 3: Convert MaxK sparse output back to full dimensions
    maxk_full_output = torch.zeros_like(input_features)
    for i in range(input_features.shape[0]):
        for j in range(dim_k):
            idx = topk_indices[i, j].item()
            maxk_full_output[i, idx] = maxk_sparse_output[i, j]
    
    # Step 4: Run ACTUAL cuSPARSE kernel (fast!)
    print("🚀 Running cuSPARSE kernel...")
    cusparse_output = maxk_cuda_kernels.cusparse_spmm(
        graph_data['indptr'],
        graph_data['indices'], 
        graph_data['values'],
        sparse_input_full,
        timing=False
    )
    
    print(f"📊 cuSPARSE output:")
    print(f"   cusparse_output shape: {cusparse_output.shape}")
    print(f"   cusparse_output[0,:10]: {cusparse_output[0,:10]}")
    
    # Step 5: Compare results
    error = torch.abs(maxk_full_output - cusparse_output).max().item()
    avg_error = torch.abs(maxk_full_output - cusparse_output).mean().item()
    
    print(f"📊 Comparison:")
    print(f"   Max error: {error:.8f}")
    print(f"   Avg error: {avg_error:.8f}")
    
    is_valid = error < tolerance
    
    if is_valid:
        print("✅ Validation PASSED!")
    else:
        print("❌ Validation FAILED!")
        
    return is_valid
Finally, update your setup_direct_kernels.py to include the cuSPARSE source:
pythonsources=[
    'cuda_kernel_bindings.cpp',
    'cuda_kernel_wrappers.cu', 
    'kernels/spmm_maxk.cu',
    'kernels/spmm_maxk_backward.cu',
    'kernels/spmm_cusparse.cu',  # Add this line
],
