#include "open3d/core/linalg/Cholesky.h"

#include "open3d/core/linalg/CholeskyImpl.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/Tri.h"

namespace open3d {
namespace core {

// Decompose output in P, L, U matrix form.
void CholeskyHelper(const Tensor& A, Tensor& output) {
    AssertTensorDtypes(A, {Float32, Float64});

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions.
    const SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor must be 2D, but got {}D.", A_shape.size());
    }

    const int64_t rows = A_shape[0];
    const int64_t cols = A_shape[1];
    if (rows == 0 || cols == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    // "output" tensor is modified in-place as output.
    // Operations are COL_MAJOR.
    output = A.T().Clone();
    void* A_data = output.GetDataPtr();

    // Returns Cholesky decomposition in form of an output matrix,
    // with lower triangular elements as L, upper triangular and diagonal
    // elements as U, (diagonal elements of L are unity), and ipiv array,
    // which has the pivot indices (for 1 <= i <= min(M,N), row i of the
    // matrix was interchanged with row IPIV(i).
    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CholeskyCUDA(A_data, cols, dtype, device);
#else
        utility::LogInfo("Unimplemented device.");
#endif
    }
    else {
      CholeskyCPU(A_data, cols, dtype, device);
    }

    // COL_MAJOR -> ROW_MAJOR.
    output = output.T().Contiguous();
}

static void OutputToL(const Tensor& output,
		      Tensor& lower)
{
    // Get upper and lower matrix from output matrix.
    Tril(output, lower, 0);
}

void Cholesky(const Tensor& A,
	      Tensor& L) {
    AssertTensorDtypes(A, {Float32, Float64});

    core::Tensor lower;

    // Get output matrix and ipiv.
    CholeskyHelper(A, lower);

    OutputToL(lower, L);
}
}  // namespace core
}  // namespace open3d
