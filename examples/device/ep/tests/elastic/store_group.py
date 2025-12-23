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

"""
TCPStore group management for distributed coordination.

This module provides utilities for creating and managing torch TCPStore
instances that can be shared between:
- RankClient (for rank management)
- nixl_ep.Buffer (for NIXL metadata exchange)
"""

from datetime import timedelta

import torch.distributed as dist


def create_master_store(
    port: int = 9999,
    timeout_sec: float = 300.0,
) -> dist.TCPStore:
    """
    Create a TCPStore master (server).

    This should be called by exactly one process in the group.
    The master holds the store server that other processes connect to.

    Args:
        port: Port for the TCPStore server
        timeout_sec: Timeout for store operations

    Returns:
        A TCPStore instance acting as the master
    """
    return dist.TCPStore(
        host_name="0.0.0.0",
        port=port,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_sec),
    )


def create_client_store(
    master_addr: str = "127.0.0.1",
    port: int = 9999,
    timeout_sec: float = 300.0,
) -> dist.TCPStore:
    """
    Create a TCPStore client (worker).

    This connects to an existing master store.

    Args:
        master_addr: Address of the master node
        port: Port of the TCPStore server
        timeout_sec: Timeout for store operations

    Returns:
        A TCPStore instance connected to the master
    """
    return dist.TCPStore(
        host_name=master_addr,
        port=port,
        is_master=False,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_sec),
    )


def create_store(
    master_addr: str = "127.0.0.1",
    port: int = 9999,
    is_master: bool = False,
    timeout_sec: float = 300.0,
) -> dist.TCPStore:
    """
    Create a TCPStore (master or client based on is_master flag).

    Convenience function that combines create_master_store and create_client_store.

    Args:
        master_addr: Address of the master node (ignored if is_master=True)
        port: Port for the TCPStore
        is_master: Whether this process is the master
        timeout_sec: Timeout for store operations

    Returns:
        A TCPStore instance
    """
    if is_master:
        return create_master_store(port=port, timeout_sec=timeout_sec)
    else:
        return create_client_store(
            master_addr=master_addr, port=port, timeout_sec=timeout_sec
        )

