# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from datetime import timedelta
from multiprocessing import Process

import torch.distributed as dist

_KEY_NEXT_GLOBAL_RANK = "rank_manager/next_global_rank"
_KEY_RELEASED_RANKS = "rank_manager/released_ranks"
_KEY_LOCK = "rank_manager/lock"


def _run_master_store(port: int, timeout_sec: float) -> None:
    store = dist.TCPStore(
        host_name="0.0.0.0",
        port=port,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_sec),
    )
    store.set(_KEY_NEXT_GLOBAL_RANK, "0")
    store.set(_KEY_RELEASED_RANKS, "[]")
    store.set(_KEY_LOCK, "0")
    while True:
        time.sleep(3600)


def start_master_store_process(
    port: int = 9999,
    timeout_sec: float = 365 * 24 * 3600,
) -> Process:
    process = Process(target=_run_master_store, args=(port, timeout_sec), daemon=True)
    process.start()
    time.sleep(1)
    return process


def create_client_store(
    master_addr: str = "127.0.0.1",
    port: int = 9999,
    timeout_sec: float = 300.0,
) -> dist.TCPStore:
    return dist.TCPStore(
        host_name=master_addr,
        port=port,
        is_master=False,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_sec),
    )
