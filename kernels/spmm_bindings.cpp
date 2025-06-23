#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <memory>

// Include the SPMM kernel headers
#include "spmm_maxk.h"
#include "spmm_maxk_backward.h"

/**
 * Python wrapper for SPMM_MAXK kernel
 * Fixed for C++17 compatibility and protected member access
 */
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
        
        // Validate tensors are on CUDA and contiguous
        TORCH_CHECK(indptr.is_cuda(), "indptr must be CUDA tensor");
        TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
        TORCH_CHECK(values.is_cuda(), "values must be CUDA tensor");
        TORCH_CHECK(input_features.is_cuda(), "input_features must be CUDA tensor");
        TORCH_CHECK(output_features.is_cuda(), "output_features must be CUDA tensor");
        
        TORCH_CHECK(indptr.is_contiguous(), "indptr must be contiguous");
        TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
        TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
        TORCH_CHECK(input_features.is_contiguous(), "input_features must be contiguous");
        TORCH_CHECK(output_features.is_contiguous(), "output_features must be contiguous");
        
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
        
        // Use public method or friend access instead of direct member access
        kernel_->update_tensors(input_features.data_ptr<float>(), 
                               output_features.data_ptr<float>());
    }
    
    void set_sparse_params(torch::Tensor sparse_selector, int maxk) {
        TORCH_CHECK(sparse_selector.is_cuda() && sparse_selector.is_contiguous(), 
                   "sparse_selector must be CUDA and contiguous");
        TORCH_CHECK(sparse_selector.dtype() == torch::kInt32, 
                   "sparse_selector must be int32");
        
        kernel_->set_sparse_selector(sparse_selector.data_ptr<int>(), maxk);
    }
    
    double run_kernel(bool timing, int dim) {
        return kernel_->do_test(timing, dim);
    }
    
    std::string get_graph_name() const {
        return graph_name_;
    }
};

/**
 * Python wrapper for SPMM_MAXK_BACKWARD kernel
 * Fixed for C++17 compatibility and protected member access
 */
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
        
        // Use public method instead of direct protected member access
        kernel_->update_tensors(input_features.data_ptr<float>(), 
                               output_features.data_ptr<float>());
    }
    
    void set_sparse_params(torch::Tensor sparse_selector, int maxk) {
        TORCH_CHECK(sparse_selector.is_cuda() && sparse_selector.is_contiguous(), 
                   "sparse_selector must be CUDA and contiguous");
        TORCH_CHECK(sparse_selector.dtype() == torch::kInt32, 
                   "sparse_selector must be int32");
        
        kernel_->set_sparse_selector(sparse_selector.data_ptr<int>(), maxk);
    }
    
    double run_kernel(bool timing, int dim) {
        return kernel_->do_test(timing, dim);
    }
    
    std::string get_graph_name() const {
        return graph_name_;
    }
};

/**
 * Utility functions for CBSR format preparation
 */
std::tuple<torch::Tensor, torch::Tensor> prepare_cbsr_format(torch::Tensor features, int maxk) {
    TORCH_CHECK(features.is_cuda(), "Features must be on CUDA");
    TORCH_CHECK(features.dim() == 2, "Features must be 2D tensor");
    TORCH_CHECK(maxk > 0 && maxk <= features.size(1), "Invalid maxk value");
    
    int num_nodes = features.size(0);
    int feature_dim = features.size(1);
    
    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(features.device());
    auto index_options = torch::TensorOptions().dtype(torch::kInt32).device(features.device());
    
    torch::Tensor sparse_data = torch::zeros({num_nodes, maxk}, options);
    torch::Tensor sparse_index = torch::zeros({num_nodes, maxk}, index_options);
    
    // Apply top-k selection
    auto [values, indices] = torch::topk(features, maxk, /*dim=*/1);
    sparse_data.copy_(values);
    sparse_index.copy_(indices);
    
    return std::make_tuple(sparse_data, sparse_index);
}

/**
 * Top-k nonlinearity function
 */
torch::Tensor topk_nonlinearity(torch::Tensor input, int k) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(k > 0 && k <= input.size(1), "Invalid k value");
    
    // Apply top-k and create sparse output
    auto [values, indices] = torch::topk(input, k, /*dim=*/1);
    
    // Create sparse representation
    auto output = torch::zeros_like(input);
    auto batch_indices = torch::arange(input.size(0), input.device()).unsqueeze(1).expand({-1, k});
    
    output.index_put_({batch_indices, indices}, values);
    
    return output;
}

/**
 * PyBind11 module definition
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MaxK-GNN SPMM Kernels - C++17 Compatible Version";
    
    // PySpmmMaxK class binding
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
    
    // PySpmmMaxKBackward class binding  
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
    
    // Utility functions
    m.def("prepare_cbsr_format", &prepare_cbsr_format,
          "Convert dense features to CBSR format",
          pybind11::arg("features"), pybind11::arg("maxk"));
    
    m.def("topk_nonlinearity", &topk_nonlinearity,
          "Apply top-k nonlinearity function",
          pybind11::arg("input"), pybind11::arg("k"));
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "MaxK-GNN Team";
}