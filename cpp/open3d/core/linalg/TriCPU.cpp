// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/TriImpl.h"

namespace open3d {
namespace core {

void TriuCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx >= diagonal) {
                output_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

void TrilCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx <= diagonal) {
                output_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

void Tril3DCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *output_ptr = static_cast<scalar_t *>(output.GetDataPtr());

        SizeVector A_shape = A.GetShape();
        int batch_size = A_shape[0];
        int cols = A_shape[1];
        int n = batch_size * cols * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
            const int64_t idx = workload_idx / (cols * cols);
            const int64_t idy = (workload_idx / cols) % cols;
            const int64_t idz = workload_idx % cols;

            if (idz - idy <= diagonal) {
                output_ptr[workload_idx] =
                        A_ptr[(idx * cols + idy) * cols + idz];
            }
        });
    });
}

void TriulCPU(const Tensor &A,
              Tensor &upper,
              Tensor &lower,
              const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const scalar_t *A_ptr = static_cast<const scalar_t *>(A.GetDataPtr());
        scalar_t *upper_ptr = static_cast<scalar_t *>(upper.GetDataPtr());
        scalar_t *lower_ptr = static_cast<scalar_t *>(lower.GetDataPtr());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        ParallelFor(A.GetDevice(), n, [&] OPEN3D_DEVICE(int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx < diagonal) {
                lower_ptr[workload_idx] = A_ptr[idx * cols + idy];
            } else if (idy - idx > diagonal) {
                upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
            } else {
                lower_ptr[workload_idx] = 1;
                upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

}  // namespace core
}  // namespace open3d
