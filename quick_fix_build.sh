#!/bin/bash

echo "🚀 MaxK-GNN Quick Fix - C++17 and Protected Access"
echo "=================================================="
echo "This script fixes the build errors from the log"

# Step 1: Fix the C++17 issue by updating setup to use C++17
echo "🔧 Step 1: Fixing C++17 compilation flags..."

# Backup current spmm_base.h
if [ -f "kernels/spmm_base.h" ]; then
    cp kernels/spmm_base.h kernels/spmm_base.h.backup
    echo "✅ Backed up original spmm_base.h"
fi

# Add public accessor methods to SPMM_BASE class
echo "🔧 Step 2: Adding public accessor methods to SPMM_BASE..."

cat > kernels/spmm_base_patch.h << 'EOF'
#pragma once
#include "util.h"
#include <string>
using namespace std;

class SPMM_BASE
{
public:
    SPMM_BASE(string _graph, int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim)
    {
        this->_graph = _graph;
        this->ptr = ptr;
        this->idx = idx;
        this->val = val;
        this->vin = vin;
        this->vout = vout;
        this->num_v = num_v;
        this->num_e = num_e;
        this->dim = dim;
    }
    ~SPMM_BASE() {}

    virtual double do_test(bool timing, int dim) = 0;
    
    // PUBLIC METHODS - Fix for protected member access
    void update_tensors(float* new_vin, float* new_vout) {
        this->vin = new_vin;
        this->vout = new_vout;
    }
    
    // Add any missing methods that derived classes need
    virtual void set_sparse_selector(int* selector, int maxk) {
        // Default implementation - override in derived classes
    }

protected:
    string _graph;
    int *ptr, *idx;
    float *val, *vin, *vout;
    int num_v, num_e, dim;
    dim3 grid, block;

protected:
    virtual void run(int dim) = 0;
    double timing_body(bool timing, int dim)
    {
        double ret = 0;
        if (!timing)
        {
            run(dim);
            cudaDeviceSynchronize();
        }
        else
        {
            int times = 4;
            // warmup
            for (int i = 0; i < times; i++)
            {
                run(dim);
            }
            cudaDeviceSynchronize();
            double measured_time = 0;
            for (int i = 0; i < times; i++)
            {
                timestamp(t0);
                run(dim);
                cudaDeviceSynchronize();
                timestamp(t1);
                measured_time += getDuration(t0, t1);
            }
            ret = measured_time / times;
        }
        return ret;
    }
};
EOF

# Replace the original with patched version
cp kernels/spmm_base_patch.h kernels/spmm_base.h
echo "✅ Patched spmm_base.h with public accessor methods"

# Step 3: Create C++17 compatible setup script
echo "🔧 Step 3: Creating C++17 compatible setup script..."

cat > setup_cpp17_fix.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os

# Force C++17 standard
os.environ['CXXFLAGS'] = '-std=c++17'

# Get CUDA architectures
def get_cuda_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f'{major}.{minor}']
    return ['7.5', '8.0', '8.6']

cuda_arches = get_cuda_arch()

# NVCC arguments with explicit C++17
nvcc_args = [
    '-O3', 
    '--expt-relaxed-constexpr',
    '-std=c++17',  # EXPLICIT C++17
    '--use_fast_math'
]

# Add architecture flags
for arch in cuda_arches:
    clean_arch = arch.replace('.', '')
    nvcc_args.extend([f'-gencode=arch=compute_{clean_arch},code=sm_{clean_arch}'])

# CXX arguments with explicit C++17
cxx_args = [
    '-O3',
    '-std=c++17',  # EXPLICIT C++17
    '-fPIC'
]

print(f"🔧 Using CUDA architectures: {cuda_arches}")
print(f"🔧 NVCC args: {nvcc_args}")
print(f"🔧 CXX args: {cxx_args}")

ext = CUDAExtension(
    name='spmm_kernels',
    sources=[
        'kernels/spmm_bindings.cpp',
        'kernels/spmm_maxk.cu',
        'kernels/spmm_maxk_backward.cu',
        'kernels/spmm_cusparse.cu'
    ],
    include_dirs=[
        'kernels/',
        '/usr/local/cuda/include'
    ],
    libraries=['cusparse', 'cublas'],
    library_dirs=['/usr/local/cuda/lib64', '/usr/local/cuda-12.8/lib64'],
    extra_compile_args={
        'cxx': cxx_args,    # C++17 for C++
        'nvcc': nvcc_args   # C++17 for NVCC
    },
    extra_link_args=['-lcusparse', '-lcublas']
)

setup(
    name='spmm_kernels_cpp17',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
EOF

echo "✅ Created C++17 compatible setup script"

# Step 4: Clean and rebuild
echo "🧹 Step 4: Cleaning previous builds..."
rm -rf build/
rm -f *.so
rm -f spmm_kernels*.so

echo "🔨 Step 5: Building with C++17 support..."
python setup_cpp17_fix.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "🎉 BUILD SUCCESSFUL!"
    echo ""
    echo "🧪 Testing import..."
    python -c "import spmm_kernels; print('✅ Import successful!')" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Module import works!"
        echo ""
        echo "📋 Next steps:"
        echo "1. Update utils/models.py to use the corrected version"
        echo "2. Run: python maxk_gnn_integrated.py --dataset reddit --model sage --maxk 32"
        echo "3. Check for the debug prints from MaxKSpGEMMFunction.backward()"
    else
        echo "⚠️ Import failed, but extension built. Check Python path."
    fi
else
    echo "❌ Build failed. Check error messages above."
    echo ""
    echo "💡 If still failing, try:"
    echo "   export TORCH_CUDA_ARCH_LIST='8.0'"
    echo "   python setup_cpp17_fix.py clean --all"
    echo "   python setup_cpp17_fix.py build_ext --inplace --force"
fi

echo ""
echo "📋 Files created/modified:"
echo "  - kernels/spmm_base.h (patched with public methods)"
echo "  - kernels/spmm_base.h.backup (original backup)"
echo "  - setup_cpp17_fix.py (C++17 compatible setup)"
echo "  - kernels/spmm_base_patch.h (patch file)"