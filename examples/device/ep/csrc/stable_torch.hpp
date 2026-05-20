/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION 0x020A000000000000ULL
#endif

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#ifndef TORCH_SUCCESS
#define TORCH_SUCCESS 0
#endif

#ifndef TORCH_ERROR_CODE_CHECK
#define TORCH_ERROR_CODE_CHECK(call)                                       \
    if ((call) != TORCH_SUCCESS) {                                         \
        throw std::runtime_error(std::string(#call) + " failed");         \
    }
#endif

namespace nixl_ep {

using StableTensor = torch::stable::Tensor;
using StableScalarType = torch::headeronly::ScalarType;

inline cudaStream_t get_current_cuda_stream(int32_t device_index = -1) {
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

inline torch::stable::Device cuda_device(int32_t device_index = -1) {
    return torch::stable::Device(torch::headeronly::DeviceType::CUDA, device_index);
}

inline torch::headeronly::IntHeaderOnlyArrayRef shape_ref(const std::vector<int64_t>& shape) {
    return torch::headeronly::IntHeaderOnlyArrayRef(shape.data(), shape.size());
}

inline StableTensor empty_cuda(std::initializer_list<int64_t> shape,
                               StableScalarType dtype,
                               int32_t device_index = -1) {
    std::vector<int64_t> shape_vec(shape);
    return torch::stable::empty(shape_ref(shape_vec), dtype, std::nullopt, cuda_device(device_index));
}

inline StableTensor empty_like_shape(const StableTensor& like,
                                     std::initializer_list<int64_t> shape,
                                     std::optional<StableScalarType> dtype = std::nullopt) {
    std::vector<int64_t> shape_vec(shape);
    return torch::stable::empty(shape_ref(shape_vec), dtype.value_or(like.scalar_type()),
                                std::nullopt, like.device());
}

inline StableTensor from_blob_cuda(void* data,
                                   std::initializer_list<int64_t> sizes,
                                   std::initializer_list<int64_t> strides,
                                   StableScalarType dtype,
                                   int32_t device_index = -1) {
    std::vector<int64_t> sizes_vec(sizes);
    std::vector<int64_t> strides_vec(strides);
    return torch::stable::from_blob(
        data, shape_ref(sizes_vec), shape_ref(strides_vec), cuda_device(device_index), dtype);
}

inline cudaDataType_t scalar_type_to_cuda_data_type(StableScalarType dtype) {
    switch (dtype) {
    case StableScalarType::Half:
        return CUDA_R_16F;
    case StableScalarType::BFloat16:
        return CUDA_R_16BF;
    case StableScalarType::Float:
        return CUDA_R_32F;
    case StableScalarType::Float8_e4m3fn:
        return CUDA_R_8F_E4M3;
    default:
        throw std::runtime_error("Unsupported CUDA data type for NIXL EP");
    }
}

inline StableScalarType dtype_from_code(int64_t dtype_code) {
    return static_cast<StableScalarType>(dtype_code);
}

inline int64_t dtype_code(StableScalarType dtype) {
    return static_cast<int64_t>(dtype);
}

inline int64_t element_size(StableScalarType dtype) {
    switch (dtype) {
    case StableScalarType::Byte:
    case StableScalarType::Char:
    case StableScalarType::Bool:
    case StableScalarType::Float8_e4m3fn:
        return 1;
    case StableScalarType::Short:
    case StableScalarType::Half:
    case StableScalarType::BFloat16:
        return 2;
    case StableScalarType::Int:
    case StableScalarType::Float:
        return 4;
    case StableScalarType::Long:
    case StableScalarType::Double:
        return 8;
    default:
        throw std::runtime_error("Unsupported NIXL EP dtype");
    }
}

inline StableScalarType topk_idx_scalar_type() {
#if TOPK_IDX_BITS == 64
    return StableScalarType::Long;
#elif TOPK_IDX_BITS == 32
    return StableScalarType::Int;
#else
#error "Unsupported TOPK_IDX_BITS"
#endif
}

} // namespace nixl_ep

namespace torch {
using Tensor = stable::Tensor;
using ScalarType = headeronly::ScalarType;
constexpr auto kByte = headeronly::ScalarType::Byte;
constexpr auto kBool = headeronly::ScalarType::Bool;
constexpr auto kChar = headeronly::ScalarType::Char;
constexpr auto kShort = headeronly::ScalarType::Short;
constexpr auto kInt = headeronly::ScalarType::Int;
constexpr auto kInt32 = headeronly::ScalarType::Int;
constexpr auto kInt64 = headeronly::ScalarType::Long;
constexpr auto kLong = headeronly::ScalarType::Long;
constexpr auto kHalf = headeronly::ScalarType::Half;
constexpr auto kFloat32 = headeronly::ScalarType::Float;
constexpr auto kDouble = headeronly::ScalarType::Double;
constexpr auto kBFloat16 = headeronly::ScalarType::BFloat16;
constexpr auto kFloat8_e4m3fn = headeronly::ScalarType::Float8_e4m3fn;
} // namespace torch
