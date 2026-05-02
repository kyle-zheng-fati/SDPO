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

import json
import os
from typing import Any

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

# Multi-LoRA role registry: maps each agent role to a unique vLLM adapter slot.
# Used when multi_lora=True to serve 5 per-role LoRA adapters from one vLLM instance.
ROLE_LORA_REGISTRY: dict[str, dict] = {
    "planner":               {"int_id": 1, "name": "planner",               "path": "lora_planner"},
    "executor":              {"int_id": 2, "name": "executor",              "path": "lora_executor"},
    "Python_Coder_Tool":     {"int_id": 3, "name": "Python_Coder_Tool",     "path": "lora_python_coder"},
    "Web_Search":            {"int_id": 4, "name": "Web_Search",            "path": "lora_web_search"},
    "Wikipedia_Search_Tool": {"int_id": 5, "name": "Wikipedia_Search_Tool", "path": "lora_wikipedia"},
}

# Total number of adapters for vLLM max_loras config.
MULTI_LORA_MAX_ADAPTERS = len(ROLE_LORA_REGISTRY)


def resolve_multi_lora_resume_paths(
    parent_dir: str,
    registry_keys: list[str],
) -> list[tuple[str, str]]:
    """Resolve per-role subdirectories under a multi-LoRA checkpoint parent.

    Expected on-disk layout (as produced by the multi-LoRA save loop in
    ``fsdp_workers.py``)::

        <parent_dir>/
            planner/
                adapter_model.safetensors
                adapter_config.json
            executor/
                ...
            Python_Coder_Tool/
                ...
            ...

    The function iterates ``registry_keys`` (preserving order — the first
    entry is used for ``PeftModel.from_pretrained`` and the rest are attached
    via ``load_adapter``) and for each expected adapter verifies the subdir
    exists and contains ``adapter_model.safetensors`` + ``adapter_config.json``.

    On ANY missing subdir or missing weight file, a ``RuntimeError`` is
    raised listing every missing path. This is intentional: silently
    partial-loading 1/5 adapters (the pre-fix behavior) corrupted training
    by leaving 4/5 adapters at random init while loss curves looked fine.

    Args:
        parent_dir: Filesystem path containing per-role subdirectories.
        registry_keys: Ordered list of adapter names expected under
            ``parent_dir`` (typically ``list(ROLE_LORA_REGISTRY.keys())``).

    Returns:
        List of ``(adapter_name, full_subdir_path)`` tuples, in the order
        given by ``registry_keys``.

    Raises:
        RuntimeError: If ``parent_dir`` is not a directory, or if any
            expected subdirectory / required file is missing.
    """
    if not os.path.isdir(parent_dir):
        raise RuntimeError(
            f"Multi-LoRA resume parent path is not a directory: {parent_dir}. "
            f"Expected a parent containing per-role subdirectories "
            f"(e.g. {parent_dir}/planner/, {parent_dir}/executor/, ...)."
        )

    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in registry_keys:
        subdir = os.path.join(parent_dir, name)
        weights = os.path.join(subdir, "adapter_model.safetensors")
        config = os.path.join(subdir, "adapter_config.json")
        if not os.path.isdir(subdir):
            missing.append(f"{subdir} (directory)")
            continue
        if not os.path.isfile(weights):
            missing.append(weights)
            continue
        if not os.path.isfile(config):
            missing.append(config)
            continue
        resolved.append((name, subdir))

    if missing:
        raise RuntimeError(
            f"Multi-LoRA resume from {parent_dir}: {len(missing)} expected "
            f"adapter artifact(s) are missing. Refusing to silently reset "
            f"adapters to random init. Missing: {missing}. Expected one "
            f"subdirectory per registry key {registry_keys}, each containing "
            f"adapter_model.safetensors + adapter_config.json."
        )

    return resolved


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.

    Reference: https://github.com/vllm-project/vllm/blob/8a297115e2367d463b781adb86b55ac740594cf6/vllm/config/lora.py#L27
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args
