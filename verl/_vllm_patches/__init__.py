# SPDX-License-Identifier: Apache-2.0
# Project-local monkeypatches against pinned vLLM 0.12.0.
#
# Patches applied at import time:
#
#   1. _gen_lora_extra_hash_keys: hash on lora_int_id, NOT lora_name.
#      Stock vLLM 0.12 (kv_cache_utils.py:450-462) keys the prefix-cache hash
#      on lora_request.lora_name. AgentFlow reloads the same lora_name with
#      different weights every training step, so KV blocks cached under the
#      old weights are reused for the new weights — silently corrupting
#      generations. lora_int_id changes on every reload and is the correct
#      cache key. Upstream fix: vLLM PR #31069.
#
#      Cross-refs: AA/.planning/ISSUES.md ISSUE-035.
#
# Activation: imported from external/verl/verl/__init__.py so any `import verl`
# (which runs before any vLLM engine instantiation in this codebase) installs
# the patches before vLLM's scheduler captures the function reference.

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def _patch_lora_extra_hash_keys() -> None:
    """Replace _gen_lora_extra_hash_keys to hash on lora_int_id."""
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.request import Request

    def _gen_lora_extra_hash_keys_int_id(request: Request) -> list[int]:
        if not request.lora_request:
            return []
        return [request.lora_request.lora_int_id]

    kv_cache_utils._gen_lora_extra_hash_keys = _gen_lora_extra_hash_keys_int_id
    logger.info(
        "[_vllm_patches] applied: _gen_lora_extra_hash_keys -> lora_int_id "
        "(was lora_name; ISSUE-035)"
    )


def apply_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    try:
        _patch_lora_extra_hash_keys()
    except ImportError:
        # vLLM not installed in this env (e.g. CPU-only test runs). Skip.
        logger.warning("[_vllm_patches] vLLM not importable; skipping patches.")
        return
    _PATCHES_APPLIED = True


apply_patches()

__all__: list[str] = []
