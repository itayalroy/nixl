/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <memory>

#include "stable_torch.hpp"
#include "kernels/exception.cuh"

namespace nixl_ep {

struct EventHandle {
    std::shared_ptr<cudaEvent_t> event;

    EventHandle() {
        event = std::shared_ptr<cudaEvent_t>(new cudaEvent_t{}, [](cudaEvent_t* e) {
            if (e != nullptr && *e != nullptr) {
                cudaEventDestroy(*e);
            }
            delete e;
        });
        CUDA_CHECK(cudaEventCreateWithFlags(event.get(), cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(*event, get_current_cuda_stream()));
    }

    explicit EventHandle(cudaStream_t stream) {
        event = std::shared_ptr<cudaEvent_t>(new cudaEvent_t{}, [](cudaEvent_t* e) {
            if (e != nullptr && *e != nullptr) {
                cudaEventDestroy(*e);
            }
            delete e;
        });
        CUDA_CHECK(cudaEventCreateWithFlags(event.get(), cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(*event, stream));
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        CUDA_CHECK(cudaStreamWaitEvent(get_current_cuda_stream(), *event, 0));
    }
};

cudaEvent_t create_event(cudaStream_t s) {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(event, s));
    return event;
}

void stream_wait(cudaStream_t s_0, cudaStream_t s_1) {
    if (s_0 == s_1) {
        return;
    }
    cudaEvent_t event = create_event(s_1);
    CUDA_CHECK(cudaStreamWaitEvent(s_0, event, 0));
    CUDA_CHECK(cudaEventDestroy(event));
}

void stream_wait(cudaStream_t s, const EventHandle& event) {
    CUDA_CHECK(cudaStreamWaitEvent(s, *event.event, 0));
}

} // namespace nixl_ep
