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

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import torch.distributed as dist

_KEY_NEXT_GLOBAL_RANK = "rank_manager/next_global_rank"
_KEY_RELEASED_RANKS = "rank_manager/released_ranks"
_KEY_LOCK = "rank_manager/lock"


def _host_local_ranks_key(hostname: str) -> str:
    return f"rank_manager/host/{hostname}/local_ranks"


def _rank_context_key(global_rank: int) -> str:
    return f"rank_manager/rank/{global_rank}/context"


def _rank_hostname_key(global_rank: int) -> str:
    return f"rank_manager/rank/{global_rank}/hostname"


def _rank_local_key(global_rank: int) -> str:
    return f"rank_manager/rank/{global_rank}/local_rank"


def init_keys(store: dist.TCPStore) -> None:
    store.add(_KEY_NEXT_GLOBAL_RANK, 0)
    store.set(_KEY_RELEASED_RANKS, "[]")
    store.set(_KEY_LOCK, "0")


class RankManager:

    def __init__(self, store: dist.TCPStore):
        self._global_rank: Optional[int] = None
        self._hostname = os.uname().nodename
        self._store = store

    @contextmanager
    def _lock(self) -> Iterator[None]:
        my_id = f"{self._hostname}_{os.getpid()}_{threading.get_ident()}"
        while True:
            result = self._store.compare_set(_KEY_LOCK, "0", my_id)
            if result.decode() == my_id:
                break
            if result == b"0":
                continue
            time.sleep(0.0001)
        try:
            yield
        finally:
            self._store.set(_KEY_LOCK, "0")

    def _get_json_list(self, key: str) -> list:
        return (
            json.loads(self._store.get(key).decode())
            if self._store.check([key])
            else []
        )

    def _set_json_list(self, key: str, value: list):
        self._store.set(key, json.dumps(value))

    def get_rank(self) -> Tuple[int, int, Optional[str]]:
        """Returns (local_rank, global_rank, user_context)."""
        if self._global_rank is not None:
            print(
                f"WARNING: rank already assigned - returning existing rank {self._global_rank}",
                flush=True,
            )
            return 0, self._global_rank, None

        start = time.perf_counter()
        user_context: Optional[str] = None

        with self._lock():
            released = self._get_json_list(_KEY_RELEASED_RANKS)

            if released:
                global_rank = min(released)
                released.remove(global_rank)
                self._set_json_list(_KEY_RELEASED_RANKS, released)
                ctx_data = self._store.get(_rank_context_key(global_rank)).decode()
                if ctx_data and ctx_data != "None":
                    user_context = ctx_data
                self._store.delete_key(_rank_context_key(global_rank))
            else:
                global_rank = self._store.add(_KEY_NEXT_GLOBAL_RANK, 1) - 1

            local_ranks_key = _host_local_ranks_key(self._hostname)
            used_local_ranks = set(self._get_json_list(local_ranks_key))
            local_rank = 0
            while local_rank in used_local_ranks:
                local_rank += 1
            used_local_ranks.add(local_rank)
            self._store.multi_set(
                [
                    local_ranks_key,
                    _rank_hostname_key(global_rank),
                    _rank_local_key(global_rank),
                ],
                [json.dumps(list(used_local_ranks)), self._hostname, str(local_rank)],
            )

        self._global_rank = global_rank
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[rank_manager] get_rank took {elapsed_ms:.2f} ms", flush=True)
        return local_rank, global_rank, user_context

    def release_rank(self, user_context: Optional[str] = None) -> bool:
        if self._global_rank is None:
            return False

        global_rank = self._global_rank

        with self._lock():
            hostname_key = _rank_hostname_key(global_rank)
            local_key = _rank_local_key(global_rank)
            values = self._store.multi_get([hostname_key, local_key])
            hostname = values[0].decode()
            local_rank = int(values[1].decode())

            local_ranks_key = _host_local_ranks_key(hostname)
            used_local_ranks = self._get_json_list(local_ranks_key)
            if local_rank in used_local_ranks:
                used_local_ranks.remove(local_rank)
            self._set_json_list(local_ranks_key, used_local_ranks)

            if user_context is not None:
                self._store.set(_rank_context_key(global_rank), str(user_context))

            self._store.delete_key(hostname_key)
            self._store.delete_key(local_key)

            released = self._get_json_list(_KEY_RELEASED_RANKS)
            released.append(global_rank)
            self._set_json_list(_KEY_RELEASED_RANKS, released)

        self._global_rank = None
        return True
