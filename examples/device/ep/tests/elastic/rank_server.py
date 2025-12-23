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
Rank management using torch.distributed.TCPStore.

Provides distributed rank assignment with:
- Local rank assignment per hostname
- Global rank assignment with recycling of released ranks
- User context passing from released ranks to new owners

The TCPStore client is created externally and passed
to RankClient, allowing the same store client to be shared
with nixl_ep for metadata exchange.
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import torch.distributed as dist

_KEY_NEXT_GLOBAL_RANK = "rank_server/next_global_rank"
_KEY_RELEASED_RANKS = "rank_server/released_ranks"
_KEY_LOCK = "rank_server/lock"


def _host_local_ranks_key(hostname: str) -> str:
    return f"rank_server/host/{hostname}/local_ranks"


def _rank_context_key(global_rank: int) -> str:
    return f"rank_server/rank/{global_rank}/context"


def _rank_hostname_key(global_rank: int) -> str:
    return f"rank_server/rank/{global_rank}/hostname"


def _rank_local_key(global_rank: int) -> str:
    return f"rank_server/rank/{global_rank}/local_rank"


def init_rank_server_keys(store: dist.TCPStore) -> None:
    """
    Initialize the rank server metadata keys in the store.

    This should be called once by the master process after creating
    the store but before any RankClient operations.

    Args:
        store: The TCPStore to initialize
    """
    store.set(_KEY_NEXT_GLOBAL_RANK, "0")
    store.set(_KEY_RELEASED_RANKS, "[]")
    store.set(_KEY_LOCK, "0")  # 0 = unlocked, 1 = locked


class RankClient:
    """
    Client for distributed rank management using TCPStore.

    Provides methods to acquire and release distributed ranks with
    optional user context passing between rank owners.
    """

    def __init__(
        self,
        store: dist.TCPStore,
    ):
        """
        Initialize the rank client.

        Args:
            store: TCPStore instance (created via store_group)
        """
        self.self_global_rank: Optional[int] = None
        self._hostname = os.uname().nodename
        self._store = store

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Context manager for distributed lock using compare_set."""
        my_id = f"{self._hostname}_{os.getpid()}"
        while True:
            result = self._store.compare_set(_KEY_LOCK, "0", my_id)
            if result.decode() == my_id:
                break
            if result == b"0":
                continue
            time.sleep(0.001)
        try:
            yield
        finally:
            self._store.set(_KEY_LOCK, "0")

    def _get_json_list(self, key: str) -> list:
        """Get a JSON list from store."""
        self._store.compare_set(key, "", "[]")
        data = self._store.get(key).decode()
        return json.loads(data) if data else []

    def _set_json_list(self, key: str, value: list):
        """Set a JSON list in store."""
        self._store.set(key, json.dumps(value))

    def get_rank(self) -> Tuple[int, int, Optional[int]]:
        """
        Get a rank assignment.

        Returns:
            Tuple of (local_rank, global_rank, user_context)
            - local_rank: The local rank on this host (0, 1, 2, ...)
            - global_rank: The global rank across all hosts
            - user_context: Context from previous owner if this is a recycled rank,
                          None otherwise. Returned as int if parseable, else None.

        Raises:
            RuntimeError: If a rank is already assigned to this client
        """
        if self.self_global_rank is not None:
            print(
                f"WARNING: rank already assigned - returning existing rank {self.self_global_rank}",
                flush=True,
            )
            return self.self_global_rank, self.self_global_rank, None

        with self._lock():
            user_context: Optional[int] = None
            released = self._get_json_list(_KEY_RELEASED_RANKS)

            if released:
                # Take the lowest released rank and remove it from the list
                global_rank = min(released)
                released.remove(global_rank)
                self._set_json_list(_KEY_RELEASED_RANKS, released)

                # Get user context if any
                ctx_data = self._store.get(_rank_context_key(global_rank)).decode()
                if ctx_data and ctx_data != "None":
                    user_context = int(ctx_data)
                self._store.delete_key(_rank_context_key(global_rank))
            else:
                next_rank = int(self._store.get(_KEY_NEXT_GLOBAL_RANK).decode())
                global_rank = next_rank
                self._store.set(_KEY_NEXT_GLOBAL_RANK, str(next_rank + 1))

            # Find lowest unused local rank for this hostname
            local_ranks_key = _host_local_ranks_key(self._hostname)
            used_local_ranks = set(self._get_json_list(local_ranks_key))
            local_rank = 0
            while local_rank in used_local_ranks:
                local_rank += 1

            # Allocate local rank
            used_local_ranks.add(local_rank)
            self._set_json_list(local_ranks_key, list(used_local_ranks))

            # Store rank metadata
            self._store.set(_rank_hostname_key(global_rank), self._hostname)
            self._store.set(_rank_local_key(global_rank), str(local_rank))

        self.self_global_rank = global_rank
        return local_rank, global_rank, user_context

    def release_rank(self, user_context: str | None = None) -> bool:
        """
        Release the currently held rank.

        Args:
            user_context: Optional context to pass to the next process
                         that acquires this rank.

        Returns:
            True if successful, False otherwise.
        """
        if self.self_global_rank is None:
            return False

        global_rank = self.self_global_rank

        with self._lock():
            hostname = self._store.get(_rank_hostname_key(global_rank)).decode()
            local_rank = int(self._store.get(_rank_local_key(global_rank)).decode())

            # Release local rank from host
            local_ranks_key = _host_local_ranks_key(hostname)
            used_local_ranks = self._get_json_list(local_ranks_key)
            if local_rank in used_local_ranks:
                used_local_ranks.remove(local_rank)
            self._set_json_list(local_ranks_key, used_local_ranks)

            # Store user context for next owner
            if user_context is not None:
                self._store.set(_rank_context_key(global_rank), str(user_context))

            # Clean up rank metadata
            self._store.delete_key(_rank_hostname_key(global_rank))
            self._store.delete_key(_rank_local_key(global_rank))

            # Add to released ranks pool
            released = self._get_json_list(_KEY_RELEASED_RANKS)
            released.append(global_rank)
            self._set_json_list(_KEY_RELEASED_RANKS, released)

        self.self_global_rank = None
        return True
