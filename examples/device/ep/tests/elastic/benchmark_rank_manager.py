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
Benchmark for RankManager.

Usage:
    # Terminal 1: Start the server
    python benchmark_rank_manager.py --server

    # Terminal 2: Run the benchmark
    python benchmark_rank_manager.py --num-ranks 8
"""

import argparse
import statistics
import threading
import time

import rank_manager
import store_group


def run_server(port: int):
    print(f"Starting TCPStore master on port {port}...")
    master_store = store_group.create_master_store(port=port, timeout_sec=3600)
    rank_manager.init_keys(master_store)
    print("Server ready. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nServer stopped.")


def worker(
    rank_id: int,
    host: str,
    port: int,
    results: list,
    barrier: threading.Barrier,
):
    store = store_group.create_client_store(master_addr=host, port=port)
    mgr = rank_manager.RankManager(store)

    # Wait for all threads to be ready
    barrier.wait()

    start = time.perf_counter()
    local_rank, global_rank, _ = mgr.get_rank()
    elapsed_ms = (time.perf_counter() - start) * 1000

    results[rank_id] = {
        "local_rank": local_rank,
        "global_rank": global_rank,
        "elapsed_ms": elapsed_ms,
    }


def run_benchmark(host: str, port: int, num_ranks: int):
    print(f"Connecting to TCPStore at {host}:{port}")
    print(f"Spawning {num_ranks} threads...")

    results = [None] * num_ranks
    barrier = threading.Barrier(num_ranks)
    threads = []

    for i in range(num_ranks):
        t = threading.Thread(target=worker, args=(i, host, port, results, barrier))
        threads.append(t)

    # Start all threads
    start_all = time.perf_counter()
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()
    total_time_ms = (time.perf_counter() - start_all) * 1000

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    times = [r["elapsed_ms"] for r in results]
    for i, r in enumerate(results):
        print(
            f"  Thread {i:2d}: global_rank={r['global_rank']:2d}, "
            f"local_rank={r['local_rank']:2d}, time={r['elapsed_ms']:.2f} ms"
        )

    print("\n" + "-" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"  Total threads:    {num_ranks}")
    print(f"  Total time:       {total_time_ms:.2f} ms")
    print(f"  Min latency:      {min(times):.2f} ms")
    print(f"  Max latency:      {max(times):.2f} ms")
    print(f"  Mean latency:     {statistics.mean(times):.2f} ms")
    print(f"  Median latency:   {statistics.median(times):.2f} ms")
    if len(times) > 1:
        print(f"  Stddev:           {statistics.stdev(times):.2f} ms")
    print(f"  p50:              {statistics.quantiles(times, n=100)[49]:.2f} ms")
    print(f"  p90:              {statistics.quantiles(times, n=100)[89]:.2f} ms")
    print(f"  p99:              {statistics.quantiles(times, n=100)[98]:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RankManager")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as server (creates TCPStore master)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="TCPStore host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="TCPStore port (default: 9999)",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=8,
        help="Number of ranks to simulate (default: 8)",
    )

    args = parser.parse_args()

    if args.server:
        run_server(args.port)
    else:
        run_benchmark(args.host, args.port, args.num_ranks)


if __name__ == "__main__":
    main()

