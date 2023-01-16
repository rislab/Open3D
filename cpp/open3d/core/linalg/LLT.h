#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

void LLTHelper(const Tensor& A, Tensor& output);

// See documentation for `core::Tensor::LLT`.
void LLT(const Tensor& A,
	      Tensor& lower);

}  // namespace core
}  // namespace open3d
