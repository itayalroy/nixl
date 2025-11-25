#!/usr/bin/env python3

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

import torch

from nixl_cu12._api import nixl_agent, nixl_agent_config
from nixl_cu12.logging import get_logger

from isend_irecv import NixlP2PManager, NixlRequestStatus

# Configure logging
logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Initializing NIXL agents")
    agent_config = nixl_agent_config(backends=["UCX"])
    receiver_agent = nixl_agent("receiver", agent_config)
    sender_agent = nixl_agent("sender", None)

    # Exchange metadata bidirectionally (connection info only, no buffers needed)
    logger.info("Exchanging metadata between agents")
    receiver_meta = receiver_agent.get_agent_metadata()
    sender_meta = sender_agent.get_agent_metadata()
    sender_agent.add_remote_agent(receiver_meta)
    receiver_agent.add_remote_agent(sender_meta)

    # Create tensors for 20 send/recv operations
    send_tensors = [torch.arange(10, dtype=torch.float32) + (i * 10 + 1.0) for i in range(20)]
    recv_tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(20)]
    
    # Create managers
    receiver_manager = NixlP2PManager(receiver_agent)
    sender_manager = NixlP2PManager(sender_agent)
    
    # Use the same tag for all operations to test ordering
    tag = "test"
    
    logger.info("Starting 20 isends with the same tag")
    send_reqs = []
    for i, send_tensor in enumerate(send_tensors):
        send_req = sender_manager.isend(send_tensor, dst_agent="receiver", tag=tag)
        send_reqs.append(send_req)
    
    logger.info("Starting 20 irecvs with the same tag (should receive in order)")
    recv_reqs = []
    for i, recv_tensor in enumerate(recv_tensors):
        recv_req = receiver_manager.irecv(recv_tensor, src_agent="sender", tag=tag)
        recv_reqs.append(recv_req)
    
    # Progress until all operations complete
    import time
    while True:
        recv_statuses = receiver_manager.batch_progress(recv_reqs)
        send_statuses = sender_manager.batch_progress(send_reqs)
        
        # Get all statuses
        all_statuses = {**send_statuses, **recv_statuses}
        all_statuses = [all_statuses[req] for req in send_reqs + recv_reqs]
        
        # Check for errors first
        if any(status == NixlRequestStatus.ERROR for status in all_statuses):
            raise RuntimeError("Transfer failed")
        
        # Check if all requests are done
        all_done = all(status == NixlRequestStatus.DONE for status in all_statuses)
        if all_done:
            break
        
        time.sleep(0.001)
    
    # Verify order: check that received tensors match sent tensors in order
    logger.info("Verifying received tensors match sent tensors in order")
    for i, (sent, recv) in enumerate(zip(send_tensors, recv_tensors)):
        logger.info(f"Tensor {i+1}: sent={sent}, received={recv}")
        if not torch.equal(sent, recv):
            raise RuntimeError(
                f"Order mismatch! Tensor {i+1}: expected {sent}, got {recv}"
            )
    
    logger.info(f"All {len(send_tensors)} tensors received in correct order!")
    
    # Cleanup
    sender_agent.remove_remote_agent("receiver")

    logger.info("Test complete")

