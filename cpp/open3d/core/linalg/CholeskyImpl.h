#pragma once

#include "open3d/core/linalg/LU.h"

namespace open3d {
namespace core {

void CholeskyCPU(void* A_data,
           void* ipiv_data,
           int64_t rows,
           int64_t cols,
           Dtype dtype,
           const Device& device);

#ifdef BUILD_CUDA_MODULE
void CholeskyCUDA(void* A_data,
            int64_t cols,
            Dtype dtype,
            const Device& device);
#endif
}  // namespace core
}  // namespace open3d
