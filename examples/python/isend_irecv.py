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

"""
Implementation of isend/irecv semantics using NIXL Python API.
"""

import enum
import struct
import time
from typing import Optional, Dict
import torch

try:
    from nixl._api import nixl_agent, nixl_xfer_handle, nixl_agent_config
    from nixl.logging import get_logger
except ImportError:
    from nixl_cu12._api import nixl_agent, nixl_xfer_handle, nixl_agent_config
    from nixl_cu12.logging import get_logger

logger = get_logger(__name__)


class NixlRequestStatus(enum.Enum):
    """Status of an async request."""
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    ERROR = "ERROR"


class NixlRequestType(enum.Enum):
    """Type of request."""
    SEND = "SEND"
    RECV = "RECV"


class NixlRequest:
    """
    Request handle for isend/irecv operations.
    """
    def __init__(self, request_id: str, request_type: NixlRequestType, remote_agent: str, 
                 xfer_handle: Optional[nixl_xfer_handle] = None):
        self.request_id = request_id
        self.type = request_type
        self.remote_agent = remote_agent
        self.xfer_handle = xfer_handle  # Only set for RECV (receiver-initiated READ)
        self._completed = False


class NixlP2PManager:
    """
    Manager for point-to-point communication using NIXL.
    
    Flow:
    - isend(): Register buffer -> Get partial MD -> Send MD + descriptors via notif -> Return request
    - irecv(): Register buffer -> Wait for notif -> Extract MD -> add_remote_agent() -> RDMA READ -> Return request
    """
    
    def __init__(self, agent: nixl_agent):
        """
        Initialize P2P manager.
        
        Args:
            agent: NIXL agent instance
        """
        self.agent = agent
        self._notif_buffer: Dict[str, Dict[str, Dict[int, bytes]]] = {}  # {src_agent: {tag: {seq: msg}}}
        self._send_seq_counters: Dict[str, int] = {}  # {tag: next_seq} for isend only
    
    def _get_notification(self, src_agent: str, tag: str) -> Optional[bytes]:
        """Get next notification in order (lowest sequence), buffering others."""
        # Ensure nested structure exists
        if src_agent not in self._notif_buffer:
            self._notif_buffer[src_agent] = {}
        if tag not in self._notif_buffer[src_agent]:
            self._notif_buffer[src_agent][tag] = {}
        
        # Check buffer for lowest sequence
        if self._notif_buffer[src_agent][tag]:
            min_seq = min(self._notif_buffer[src_agent][tag].keys())
            return self._notif_buffer[src_agent][tag].pop(min_seq)
        
        # Fetch and buffer new notifications
        tag_bytes = tag.encode('utf-8')
        notifs = self.agent.get_new_notifs()
        if src_agent in notifs:
            for msg in notifs[src_agent]:
                if not msg.startswith(tag_bytes + b":"):
                    continue
                
                # Parse: tag:seq:metadata_len:metadata:descriptors
                # Split on first 3 colons to get tag, seq, md_len_str
                parts = msg.split(b":", 3)
                if len(parts) < 4:
                    raise RuntimeError(
                        f"Invalid notification format from {src_agent} with tag {tag}. "
                        f"Expected format: tag:seq:metadata_len:metadata:descriptors. "
                        f"Got {len(parts)} parts instead of at least 4. Message preview: {msg[:100]}"
                    )
                
                tag_part, seq_bytes, md_len_bytes, rest = parts
                if tag_part != tag_bytes:
                    raise RuntimeError(
                        f"Tag mismatch in notification from {src_agent}. "
                        f"Expected tag: {tag_bytes}, got: {tag_part}"
                    )
                
                try:
                    seq_num = int(seq_bytes.decode('utf-8'))
                    md_len = struct.unpack('>I', md_len_bytes)[0]
                except (ValueError, UnicodeDecodeError, struct.error) as e:
                    raise RuntimeError(
                        f"Invalid sequence number or metadata length in notification from {src_agent} with tag {tag}. "
                        f"Sequence bytes: {seq_bytes}, md_len bytes: {md_len_bytes}, error: {e}"
                    )
                
                # Extract metadata and descriptors using md_len
                if len(rest) < md_len + 1:  # +1 for the colon separator
                    raise RuntimeError(
                        f"Notification too short from {src_agent} with tag {tag}. "
                        f"Expected at least {md_len + 1} bytes after metadata_len, got {len(rest)}"
                    )
                
                md_part = rest[:md_len]
                desc_part = rest[md_len + 1:]  # Skip the colon after metadata
                
                # Verify metadata length matches
                if len(md_part) != md_len:
                    raise RuntimeError(
                        f"Metadata length mismatch in notification from {src_agent} with tag {tag}. "
                        f"Expected {md_len} bytes, got {len(md_part)} bytes"
                    )
                
                self._notif_buffer[src_agent][tag][seq_num] = msg
        
        # Check buffer again for lowest sequence
        if self._notif_buffer[src_agent][tag]:
            min_seq = min(self._notif_buffer[src_agent][tag].keys())
            return self._notif_buffer[src_agent][tag].pop(min_seq)
        
        return None
    
    def isend(self, tensor: torch.Tensor, dst_agent: str, tag: Optional[str] = None) -> NixlRequest:
        """
        Non-blocking send operation.
        
        1. Register buffer
        2. Get partial metadata for this buffer
        3. Send metadata + descriptors via notification
        
        Args:
            tensor: Tensor to send
            dst_agent: Destination agent name
            tag: Optional tag string (if None, uses "default")
        
        Returns:
            NixlRequest handle
        """
        if tag is None:
            tag = "default"
        
        # Get sequence number for this tag
        seq = self._send_seq_counters.get(tag, 0)
        self._send_seq_counters[tag] = seq + 1
        
        tag_bytes = tag.encode('utf-8')
        seq_bytes = str(seq).encode('utf-8')
        
        # 1. Register buffer
        reg_descs = self.agent.register_memory([tensor])
        if not reg_descs:
            raise RuntimeError("Failed to register memory")
        
        xfer_descs = reg_descs.trim()
        
        # 2. Get partial metadata for this buffer only
        partial_md = self.agent.get_partial_agent_metadata(
            reg_descs,
            inc_conn_info=False,  # Connection info should already be exchanged initially
            backends=[]
        )
        
        # 3. Serialize descriptors
        desc_str = self.agent.get_serialized_descs(xfer_descs)
        
        # 4. Construct notification: tag:seq:metadata_len:metadata:descriptors
        md_len = struct.pack('>I', len(partial_md))
        notif_msg = tag_bytes + b":" + seq_bytes + b":" + md_len + b":" + partial_md + b":" + desc_str
        
        self.agent.send_notif(dst_agent, notif_msg)
        
        request = NixlRequest(tag, NixlRequestType.SEND, dst_agent)
        return request
    
    def irecv(self, tensor: torch.Tensor, src_agent: str, tag: Optional[str] = None) -> NixlRequest:
        """
        Non-blocking receive operation.
        
        1. Register buffer
        2. Wait for notification from sender (contains metadata + descriptors)
        3. Extract metadata and add it incrementally via add_remote_agent()
        4. Wait for metadata to be ready
        5. RDMA READ from sender buffer to local buffer
        
        Args:
            tensor: Tensor to receive into
            src_agent: Source agent name
            tag: Optional tag string (if None, uses "default")
        
        Returns:
            NixlRequest handle
        """
        if tag is None:
            tag = "default"
        
        # 1. Register buffer
        reg_descs = self.agent.register_memory([tensor])
        if not reg_descs:
            raise RuntimeError("Failed to register memory")
        
        local_descs = reg_descs.trim()
        
        # 2. Wait for next notification in order (lowest sequence)
        # Format: tag:seq:metadata_len:metadata:descriptors
        msg = None
        while msg is None:
            msg = self._get_notification(src_agent, tag)
            if msg is None:
                time.sleep(0.001)
        
        # Parse notification: tag:seq:metadata_len:metadata:descriptors
        # Split on first 3 colons to get tag, seq, md_len_str
        parts = msg.split(b":", 3)
        if len(parts) < 4:
            raise RuntimeError(
                f"Invalid notification format. "
                f"Expected format: tag:seq:metadata_len:metadata:descriptors. "
                f"Got {len(parts)} parts instead of at least 4"
            )
        
        tag_part, seq_bytes, md_len_bytes, rest = parts
        
        try:
            md_len = struct.unpack('>I', md_len_bytes)[0]
        except struct.error as e:
            raise RuntimeError(
                f"Invalid metadata length format. md_len bytes: {md_len_bytes}, error: {e}"
            )
        
        # Extract metadata and descriptors using md_len
        if len(rest) < md_len + 1:  # +1 for the colon separator
            raise RuntimeError(
                f"Notification too short. "
                f"Expected at least {md_len + 1} bytes after metadata_len, got {len(rest)}"
            )
        
        md_part = rest[:md_len]
        desc_part = rest[md_len + 1:]  # Skip the colon after metadata
        
        # Verify metadata length matches
        if len(md_part) != md_len:
            raise RuntimeError(
                f"Metadata length mismatch. Expected {md_len} bytes, got {len(md_part)} bytes"
            )
        
        partial_md = md_part
        remote_descs = self.agent.deserialize_descs(desc_part)
        
        # 3. Add partial metadata incrementally (merges with existing metadata)
        try:
            added_agent_name = self.agent.add_remote_agent(partial_md)
            # add_remote_agent returns bytes, decode to string for comparison
            if isinstance(added_agent_name, bytes):
                added_agent_name = added_agent_name.decode('utf-8')
        except Exception as e:
            raise RuntimeError(
                f"Failed to add remote agent metadata: {e}. "
                f"Make sure initial metadata exchange completed successfully."
            )
        
        if added_agent_name != src_agent:
            raise RuntimeError(
                f"Metadata agent name mismatch: expected {src_agent}, got {added_agent_name}. "
                f"This indicates metadata corruption or receiving metadata from wrong agent."
            )
        
        # 4. Verify metadata is ready (should be immediate after add_remote_agent)
        if not self.agent.check_remote_metadata(src_agent, remote_descs):
            raise RuntimeError(
                f"Failed to verify metadata for remote buffer from {src_agent}. "
                f"Metadata should be available immediately after add_remote_agent()."
            )
        
        # 5. RDMA READ from sender buffer to local buffer
        tag_bytes = tag.encode('utf-8')
        xfer_handle = self.agent.initialize_xfer(
            "READ",
            local_descs,      # Destination: receiver's buffer
            remote_descs,     # Source: sender's buffer
            src_agent,
            tag_bytes
        )
        
        if not xfer_handle:
            raise RuntimeError("Failed to create transfer handle")
        
        # Initiate transfer
        state = self.agent.transfer(xfer_handle, tag_bytes)
        if state == "ERR":
            raise RuntimeError("Failed to initiate transfer")
        
        request = NixlRequest(tag, NixlRequestType.RECV, src_agent, xfer_handle=xfer_handle)
        return request
    
    def progress(self, request: NixlRequest) -> NixlRequestStatus:
        """
        Progress a request and return its status.
        
        Args:
            request: NixlRequest handle
        
        Returns:
            NixlRequestStatus enum value
        """
        if request._completed:
            return NixlRequestStatus.DONE
        
        if request.type == NixlRequestType.SEND:
            # For send: check if remote completed the READ by looking for completion notification
            # The receiver sends a notification with "COMPLETE:" prefix when READ finishes
            # The remote_agent in the send request is the destination (receiver), so notifications come FROM that agent
            completion_tag = b"COMPLETE:" + request.request_id.encode('utf-8')
            
            # Check for completion notification from the receiver (remote_agent is the receiver)
            done = self.agent.check_remote_xfer_done(request.remote_agent, completion_tag, tag_is_prefix=False)
            if done:
                request._completed = True
                return NixlRequestStatus.DONE
            return NixlRequestStatus.IN_PROGRESS
        else:  # NixlRequestType.RECV
            # For receive: check transfer state
            if request.xfer_handle is None:
                return NixlRequestStatus.ERROR
            
            state = self.agent.check_xfer_state(request.xfer_handle)
            if state == "DONE":
                # Send completion notification to sender so it knows the READ is done
                # This only happens once since _completed will prevent re-entry
                completion_msg = b"COMPLETE:" + request.request_id.encode('utf-8')
                self.agent.send_notif(request.remote_agent, completion_msg)
                request._completed = True
                return NixlRequestStatus.DONE
            elif state == "ERR":
                return NixlRequestStatus.ERROR
            return NixlRequestStatus.IN_PROGRESS
    
    def batch_progress(self, requests: list[NixlRequest]) -> Dict[NixlRequest, NixlRequestStatus]:
        """
        Progress multiple requests at once.
        
        Args:
            requests: List of NixlRequest handles
        
        Returns:
            Dictionary mapping each request to its NixlRequestStatus
        """
        results = {}
        for request in requests:
            results[request] = self.progress(request)
        return results
