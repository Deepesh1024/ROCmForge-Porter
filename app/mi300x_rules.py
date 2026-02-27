"""
ROCmForge Studio — MI300X Deterministic Expert Rules Engine

Applies architecture-specific CUDA-to-ROCm transformations with
full reasoning traces for every rule triggered.
"""

import re
from typing import List, Tuple


def apply_rules(cuda_code: str) -> Tuple[str, List[str]]:
    """
    Apply MI300X-specific deterministic rules to CUDA code.

    Parameters
    ----------
    cuda_code : str
        Raw or hipified CUDA source code.

    Returns
    -------
    tuple[str, list[str]]
        (mutated_code, triggered_reasons)
        - mutated_code: the transformed source string
        - triggered_reasons: list of exact trace reason strings for each rule fired
    """
    mutated = cuda_code
    reasons: List[str] = []

    # ── Rule 1: warpSize → 64 ───────────────────────────────────
    # Replace the literal keyword `warpSize` with 64
    if re.search(r'\bwarpSize\b', mutated):
        mutated = re.sub(r'\bwarpSize\b', '64', mutated)
        reasons.append(
            "[ARCHITECTURE] MI300X uses Wave64. "
            "Hardcoding wavefront boundary from 32 to 64."
        )

    # Replace bare `32` in thread/warp contexts (e.g. warp-level shuffles,
    # lane counts) but NOT arbitrary 32s in array sizes, etc.
    # Target: literal 32 near warp/lane/shuffle/thread keywords on the same line
    _warp_context_lines = []
    for line in mutated.splitlines():
        if re.search(r'(?:warp|lane|shfl|shuffle|__ballot)', line, re.IGNORECASE):
            new_line = re.sub(r'\b32\b', '64', line)
            if new_line != line:
                _warp_context_lines.append(True)
                line = new_line
        _warp_context_lines.append(False)
    # Re-apply line-level substitution
    out_lines = []
    for line in mutated.splitlines():
        if re.search(r'(?:warp|lane|shfl|shuffle|__ballot)', line, re.IGNORECASE):
            new_line = re.sub(r'\b32\b', '64', line)
            if new_line != line and "[ARCHITECTURE]" not in " ".join(reasons):
                reasons.append(
                    "[ARCHITECTURE] MI300X uses Wave64. "
                    "Hardcoding wavefront boundary from 32 to 64."
                )
            out_lines.append(new_line)
        else:
            out_lines.append(line)
    mutated = "\n".join(out_lines)

    # ── Rule 2: __shfl_down_sync → __shfl_down ──────────────────
    if re.search(r'\b__shfl_down_sync\b', mutated):
        # __shfl_down_sync(mask, val, offset) → __shfl_down(val, offset)
        # Remove the first argument (mask)
        mutated = re.sub(
            r'\b__shfl_down_sync\s*\(\s*[^,]+,\s*',
            '__shfl_down(',
            mutated,
        )
        reasons.append(
            "[COMPATIBILITY] Sync variants of shuffle are NVIDIA-specific. "
            "Reverting to base shuffle for ROCm compatibility."
        )

    # ── Rule 3: blockIdx → hipBlockIdx, threadIdx → hipThreadIdx ─
    if re.search(r'\bblockIdx\b', mutated):
        mutated = re.sub(r'\bblockIdx\b', 'hipBlockIdx', mutated)
        reasons.append(
            "[SYNTAX] Mapping standard grid coordinates to HIP."
        )

    if re.search(r'\bthreadIdx\b', mutated):
        mutated = re.sub(r'\bthreadIdx\b', 'hipThreadIdx', mutated)
        # Only add the trace once (combined with blockIdx if both present)
        if "[SYNTAX]" not in " ".join(reasons):
            reasons.append(
                "[SYNTAX] Mapping standard grid coordinates to HIP."
            )

    # ── Rule 4: __shared__ detection (no mutation, advisory only) ─
    if re.search(r'\b__shared__\b', mutated):
        reasons.append(
            "[MEMORY] Local Data Share (LDS) allocated. "
            "Note: MI300X LDS occupancy limit is 64KB per CU."
        )

    return mutated, reasons
