#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>

// Include the existing SPMM kernel headers
#include "spmm_maxk.h"
#include "spmm_maxk_backward.h"
#include "spmm_cusparse.h"

// Forward declaration of the topk kernel from maxk_kernel.cu
extern "C" {
    void launch_topk_kernel(
        uint8_t* data, uint8_t* value, uint8_t* index, 
        int N, int dim_origin, int dim_k, 
        cudaStream_t stream = 0
    );
}

// TopK kernel wrapper function
std::tuple<torch::Tensor, torch::Tensor> cuda_topk_maxk(
    torch::Tensor input, int k) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8 tensor");
    TORCH_CHECK(k > 0 && k <= input.size(1), "Invalid k value");
    
    int N = input.size(0);
    int dim_origin = input.size(1);
    
    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    torch::Tensor values = torch::zeros({N, k}, options);
    torch::Tensor indices = torch::zeros({N, k}, options);
    
    // Launch the CUDA kernel
    launch_topk_kernel(
        input.data_ptr<uint8_t>(),
        values.data_ptr<uint8_t>(),
        indices.data_ptr<uint8_t>(),
        N, dim_origin, k
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return std::make_tuple(values, indices);
}

// Enhanced TopK function that works with float tensors and converts internally
std::tuple<torch::Tensor, torch::Tensor> cuda_topk_maxk_float(
    torch::Tensor input, int k) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(k > 0 && k <= input.size(1), "Invalid k value");
    
    // If input is float, convert to uint8 for kernel processing
    torch::Tensor input_uint8;
    if (input.dtype() == torch::kFloat32) {
        // Scale and convert to uint8 (0-255 range)
        auto normalized = torch::clamp((input * 255.0).round(), 0, 255);
        input_uint8 = normalized.to(torch::kUInt8);
    } else if (input.dtype() == torch::kUInt8) {
        input_uint8 = input;
    } else {
        TORCH_CHECK(false, "Input must be float32 or uint8");
    }
    
    // Run the uint8 kernel
    auto [values_uint8, indices_uint8] = cuda_topk_maxk(input_uint8, k);
    
    // Convert results back to appropriate types
    torch::Tensor values, indices;
    
    if (input.dtype() == torch::kFloat32) {
        // Convert values back to float
        values = values_uint8.to(torch::kFloat32) / 255.0;
        indices = indices_uint8.to(torch::kInt32);
    } else {
        values = values_uint8;
        indices = indices_uint8.to(torch::kInt32);
    }
    
    return std::make_tuple(values, indices);
}

// Prepare CBSR format using the MaxK kernel
std::tuple<torch::Tensor, torch::Tensor> prepare_cbsr_format_maxk(
    torch::Tensor features, int maxk) {
    
    TORCH_CHECK(features.is_cuda(), "Features must be on CUDA");
    TORCH_CHECK(features.dim() == 2, "Features must be 2D tensor");
    TORCH_CHECK(maxk > 0 && maxk <= features.size(1), "Invalid maxk value");
    
    // Use our enhanced TopK function
    auto [sparse_data, sparse_indices] = cuda_topk_maxk_float(features, maxk);
    
    return std::make_tuple(sparse_data, sparse_indices);
}

// Existing SPMM kernel wrappers (keep these as before)
class PySpmmMaxK {
private:
    std::unique_ptr<SPMM_MAXK> kernel_;
    std::string graph_name_;
    
public:
    PySpmmMaxK(const std::string& graph_name,
               torch::Tensor indptr,
               torch::Tensor indices, 
               torch::Tensor values,
               torch::Tensor input_features,
               torch::Tensor output_features)
        : graph_name_(graph_name) {
        
        // Validate tensors
        TORCH_CHECK(indptr.is_cuda() && indptr.is_contiguous(), "indptr must be CUDA and contiguous");
        TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be CUDA and contiguous");
        TORCH_CHECK(values.is_cuda() && values.is_contiguous(), "values must be CUDA and contiguous");
        TORCH_CHECK(input_features.is_cuda() && input_features.is_contiguous(), "input_features must be CUDA and contiguous");
        TORCH_CHECK(output_features.is_cuda() && output_features.is_contiguous(), "output_features must be CUDA and contiguous");
        
        // Extract raw pointers
        int* ptr_data = indptr.data_ptr<int>();
        int* idx_data = indices.data_ptr<int>();
        float* val_data = values.data_ptr<float>();
        float* vin_data = input_features.data_ptr<float>();
        float* vout_data = output_features.data_ptr<float>();
        
        int num_v = indptr.size(0) - 1;
        int num_e = indices.size(0);
        int dim = input_features.size(1);
        
        // Create SPMM_MAXK instance
        kernel_ = std::make_unique<SPMM_MAXK>(
            graph_name_, ptr_data, idx_data, val_data, 
            vin_data, vout_data, num_v, num_e, dim
        );
    }
    
    void update_input_output(torch::Tensor input_features, torch::Tensor output_features) {
        TORCH_CHECK(input_features.is_cuda() && input_features.is_contiguous(), 
                   "input_features must be CUDA and contiguous");
        TORCH_CHECK(output_features.is_cuda() && output_features.is_contiguous(), 
                   "output_features must be CUDA and contiguous");
        
        kernel_->update_tensors(input_features.data_ptr<float>(), 
                               output_features.data_ptr<float>());
    }
    
    void set_sparse_params(torch::Tensor sparse_selector, int maxk) {
        TORCH_CHECK(sparse_selector.is_cuda() && sparse_selector.is_contiguous(), 
                   "sparse_selector must be CUDA and contiguous");
        
        if (sparse_selector.dtype() == torch::kUInt8) {
            kernel_->vin_sparse_selector = sparse_selector.data_ptr<uint8_t>();
        } else if (sparse_selector.dtype() == torch::kInt32) {
            // Convert int32 to uint8 (assuming values are < 256)
            auto selector_uint8 = sparse_selector.to(torch::kUInt8);
            kernel_->vin_sparse_selector = selector_uint8.data_ptr<uint8_t>();
        } else {
            TORCH_CHECK(false, "sparse_selector must be uint8 or int32");
        }
        
        kernel_->dim_sparse = maxk;
    }
    
    double run_kernel(bool timing, int dim) {
        return kernel_->do_test(timing, dim);
    }
    
    std::string get_graph_name() const {
        return graph_name_;
    }
};

class PySpmmMaxKBackward {
private:
    std::unique_ptr<SPMM_MAXK_BACKWARD> kernel_;
    std::string graph_name_;
    
public:
    PySpmmMaxKBackward(const std::string& graph_name,
                       torch::Tensor indptr,
                       torch::Tensor indices,
                       torch::Tensor values, 
                       torch::Tensor input_features,
                       torch::Tensor output_features)
        : graph_name_(graph_name) {
        
        // Validate tensors
        TORCH_CHECK(indptr.is_cuda() && indptr.is_contiguous(), "indptr must be CUDA and contiguous");
        TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be CUDA and contiguous");
        TORCH_CHECK(values.is_cuda() && values.is_contiguous(), "values must be CUDA and contiguous");
        TORCH_CHECK(input_features.is_cuda() && input_features.is_contiguous(), "input_features must be CUDA and contiguous");
        TORCH_CHECK(output_features.is_cuda() && output_features.is_contiguous(), "output_features must be CUDA and contiguous");
        
        // Extract raw pointers
        int* ptr_data = indptr.data_ptr<int>();
        int* idx_data = indices.data_ptr<int>();
        float* val_data = values.data_ptr<float>();
        float* vin_data = input_features.data_ptr<float>();
        float* vout_data = output_features.data_ptr<float>();
        
        int num_v = indptr.size(0) - 1;
        int num_e = indices.size(0);
        int dim = input_features.size(1);
        
        // Create SPMM_MAXK_BACKWARD instance
        kernel_ = std::make_unique<SPMM_MAXK_BACKWARD>(
            graph_name_, ptr_data, idx_data, val_data,
            vin_data, vout_data, num_v, num_e, dim
        );
    }
    
    void update_input_output(torch::Tensor input_features, torch::Tensor output_features) {
        TORCH_CHECK(input_features.is_cuda() && input_features.is_contiguous(), 
                   "input_features must be CUDA and contiguous");
        TORCH_CHECK(output_features.is_cuda() && output_features.is_contiguous(), 
                   "output_features must be CUDA and contiguous");
        
        kernel_->update_tensors(input_features.data_ptr<float>(), 
                               output_features.data_ptr<float>());
    }
    
    void set_sparse_params(torch::Tensor sparse_selector, int maxk) {
        TORCH_CHECK(sparse_selector.is_cuda() && sparse_selector.is_contiguous(), 
                   "sparse_selector must be CUDA and contiguous");
        
        if (sparse_selector.dtype() == torch::kUInt8) {
            kernel_->vin_sparse_selector = sparse_selector.data_ptr<uint8_t>();
        } else if (sparse_selector.dtype() == torch::kInt32) {
            auto selector_uint8 = sparse_selector.to(torch::kUInt8);
            kernel_->vin_sparse_selector = selector_uint8.data_ptr<uint8_t>();
        } else {
            TORCH_CHECK(false, "sparse_selector must be uint8 or int32");
        }
        
        kernel_->dim_sparse = maxk;
    }
    
    double run_kernel(bool timing, int dim) {
        return kernel_->do_test(timing, dim);
    }
    
    std::string get_graph_name() const {
        return graph_name_;
    }
};

// cuSPARSE wrapper functions
double cusparse_spmm_wrapper(torch::Tensor indptr, torch::Tensor indices, torch::Tensor values,
                            torch::Tensor input, torch::Tensor output, int times = 10) {
    TORCH_CHECK(indptr.is_cuda() && indptr.is_contiguous(), "indptr must be CUDA and contiguous");
    TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be CUDA and contiguous");
    TORCH_CHECK(values.is_cuda() && values.is_contiguous(), "values must be CUDA and contiguous");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be CUDA and contiguous");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "output must be CUDA and contiguous");
    
    int num_v = indptr.size(0) - 1;
    int num_e = indices.size(0);
    int dim = input.size(1);
    
    return spmm_cusparse(
        indptr.data_ptr<int>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_v, num_e, dim, times
    );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MaxK-GNN CUDA Kernels with Integrated TopK";
    
    // TopK kernel functions
    m.def("cuda_topk_maxk", &cuda_topk_maxk,
          "Fast CUDA TopK kernel for uint8 tensors",
          pybind11::arg("input"), pybind11::arg("k"));
    
    m.def("cuda_topk_maxk_float", &cuda_topk_maxk_float,
          "Fast CUDA TopK kernel for float tensors",
          pybind11::arg("input"), pybind11::arg("k"));
    
    m.def("prepare_cbsr_format_maxk", &prepare_cbsr_format_maxk,
          "Prepare CBSR format using MaxK TopK kernel",
          pybind11::arg("features"), pybind11::arg("maxk"));
    
    // SPMM kernel classes
    pybind11::class_<PySpmmMaxK>(m, "SpmmMaxK")
        .def(pybind11::init<const std::string&, torch::Tensor, torch::Tensor, torch::Tensor, 
                          torch::Tensor, torch::Tensor>(),
             "Initialize SPMM_MAXK kernel",
             pybind11::arg("graph_name"), pybind11::arg("indptr"), pybind11::arg("indices"), 
             pybind11::arg("values"), pybind11::arg("input_features"), pybind11::arg("output_features"))
        .def("update_input_output", &PySpmmMaxK::update_input_output,
             "Update input and output tensor pointers",
             pybind11::arg("input_features"), pybind11::arg("output_features"))
        .def("set_sparse_params", &PySpmmMaxK::set_sparse_params,
             "Set sparse selector and maxk parameters", 
             pybind11::arg("sparse_selector"), pybind11::arg("maxk"))
        .def("run_kernel", &PySpmmMaxK::run_kernel,
             "Execute the SPMM kernel",
             pybind11::arg("timing") = false, pybind11::arg("dim") = -1)
        .def("get_graph_name", &PySpmmMaxK::get_graph_name,
             "Get the graph name");
    
    pybind11::class_<PySpmmMaxKBackward>(m, "SpmmMaxKBackward")
        .def(pybind11::init<const std::string&, torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor>(),
             "Initialize SPMM_MAXK_BACKWARD kernel",
             pybind11::arg("graph_name"), pybind11::arg("indptr"), pybind11::arg("indices"),
             pybind11::arg("values"), pybind11::arg("input_features"), pybind11::arg("output_features"))
        .def("update_input_output", &PySpmmMaxKBackward::update_input_output,
             "Update input and output tensor pointers",
             pybind11::arg("input_features"), pybind11::arg("output_features"))
        .def("set_sparse_params", &PySpmmMaxKBackward::set_sparse_params,
             "Set sparse selector and maxk parameters",
             pybind11::arg("sparse_selector"), pybind11::arg("maxk"))
        .def("run_kernel", &PySpmmMaxKBackward::run_kernel,
             "Execute the backward SPMM kernel", 
             pybind11::arg("timing") = false, pybind11::arg("dim") = -1)
        .def("get_graph_name", &PySpmmMaxKBackward::get_graph_name,
             "Get the graph name");
    
    // cuSPARSE wrapper
    m.def("cusparse_spmm", &cusparse_spmm_wrapper,
          "cuSPARSE SpMM wrapper function",
          pybind11::arg("indptr"), pybind11::arg("indices"), pybind11::arg("values"),
          pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("times") = 10);
    
    // Version info
    m.attr("__version__") = "1.1.0";
    m.attr("__author__") = "MaxK-GNN Team";
    m.attr("has_topk_kernel") = true;
}
