#include "open3d/core/linalg/SolveLLT.h"

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime_api.h>
#endif

#include <unordered_map>

#include "open3d/core/linalg/LinalgHeadersCPU.h"

namespace open3d {
namespace core {

void SolveLLTBatched(const Tensor &A, const Tensor &B, Tensor &X) {
    AssertTensorDtypes(A, {Float32, Float64});
    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    AssertTensorDtype(B, dtype);
    AssertTensorDevice(B, device);

    // Check dimensions
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();
    if (A_shape.size() != 3) {
        utility::LogError("Tensor A must be 3D, but got {}D", A_shape.size());
    }
    if (A_shape[1] != A_shape[2]) {
        utility::LogError("Tensor A must be square in dims 1 and 2, but got {} x {}.",
                          A_shape[1], A_shape[2]);
    }
    if (B_shape.size() != 3) {
        utility::LogError(
                "Tensor B must be 3D (Tensor) but got {}D",
                B_shape.size());
    }
    if (B_shape[0] != A_shape[0] || B_shape[1] != A_shape[1] || B_shape [2] != A_shape[2]) {
        utility::LogError("Tensor A and B's dimension mismatch.");
    }

    int64_t n = A_shape[1];
    int64_t k = B_shape[1];
    if (n == 0 || k == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    // A and B are modified in-place
    Tensor A_copy = A.Transpose(1,2).Clone();
    void *A_data = A_copy.GetDataPtr();

    X = B.Transpose(1,2).Clone();
    void *B_data = X.GetDataPtr();

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        utility::LogError("Unimplemented device.");
        // SolveCUDALLTBatched(A_data, B_data, n, k, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        SolveCPULLTBatched(A_data, B_data, n, k, dtype, device);
    }
    X = X.Transpose(1,2);
}
}  // namespace core
}  // namespace open3d
