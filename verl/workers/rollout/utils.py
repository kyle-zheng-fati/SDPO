# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    """Bind a uvicorn HTTP server in the background and return (port, task).

    2026-04-16 fix: previous implementation set ``server.should_exit = True``
    BEFORE ``await server.serve()`` — this made uvicorn run startup() (briefly
    binding the socket) and then shutdown() (releasing it) before any request
    could land. The follow-up ``asyncio.create_task(server.main_loop())`` did
    not re-bind the socket, so the HTTP port stayed unbound forever and every
    rollout request returned 503. Replace the hack with a real background task
    and synchronously wait for ``server.started`` so callers receive a port
    that is actually listening.
    """
    import time as _t
    server_port, server_task = None, None

    for _ in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            sock.close()  # release the probe socket; uvicorn will rebind
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())
            # Wait until uvicorn has finished startup() and is ready to accept
            # requests. Bound the wait so a stuck server can be retried.
            t_wait_start = _t.time()
            while not server.started:
                if server_task.done():
                    raise RuntimeError(f"uvicorn task exited before startup completed: {server_task.exception()}")
                if _t.time() - t_wait_start > 30:
                    raise TimeoutError("uvicorn did not signal started within 30s")
                await asyncio.sleep(0.05)
            break
        except (OSError, SystemExit, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to start HTTP server on port {server_port}, error: {e}")
            if server_task is not None and not server_task.done():
                server_task.cancel()
                try:
                    await server_task
                except (asyncio.CancelledError, Exception):
                    pass
                server_task = None
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
