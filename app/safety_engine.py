"""
ROCmForge Studio — Responsible-AI Safety Engine

Performs deterministic, static safety analysis on generated ROCm code:
  • Wave64 awareness  — penalises hard-coded warp-size-32 assumptions
  • Vectorisation      — penalises scalar (non-vector) global loads
  • LDS bank conflicts — heuristic for shared-memory access patterns
  • General safety     — bounds checks, error handling, etc.

Returns a structured safety report with score, details, risk_flags, and attribution.
"""

import re
from typing import Any, Dict, List


# ──────────────────────────────────────────────────────────────────
# Check functions  (each returns penalty, detail-str, risk_flag)
# ──────────────────────────────────────────────────────────────────

def _check_wave64(code: str) -> tuple:
    """Detect hard-coded warp-size-32 assumptions (ROCm uses wave64 by default)."""
    issues: List[str] = []
    penalty = 0

    # Explicit warpSize == 32 or WARP_SIZE=32
    if re.search(r'\bwarpSize\s*==\s*32\b', code) or re.search(r'WARP_SIZE\s*=\s*32', code):
        issues.append("Hard-coded warpSize==32 detected; ROCm default wavefront is 64")
        penalty += 20

    # __shfl_sync with mask 0xFFFFFFFF (32-thread mask)
    if re.search(r'__shfl_sync\s*\(\s*0xFFFFFFFF', code, re.IGNORECASE):
        issues.append("__shfl_sync uses 0xFFFFFFFF mask (32-lane); consider 64-lane mask")
        penalty += 15

    # Lane index >= 32 not handled (heuristic: threadIdx.x % 32)
    if re.search(r'threadIdx\.x\s*%\s*32', code):
        issues.append("threadIdx.x % 32 suggests warp32 assumption")
        penalty += 10

    if not issues:
        issues.append("Wave64 check passed — no warp32 assumptions detected")

    flag = "WAVE64_ISSUE" if penalty > 0 else None
    return penalty, issues, flag


def _check_vectorisation(code: str) -> tuple:
    """Check whether global loads / stores use vector types."""
    issues: List[str] = []
    penalty = 0

    # Look for scalar global loads: float* or double* pointer dereference
    scalar_load = len(re.findall(r'\b(?:float|double|int)\s*\*', code))
    vector_load = len(re.findall(r'\b(?:float4|float2|double2|int4)\b', code))

    if scalar_load > 0 and vector_load == 0:
        issues.append(f"Only scalar memory accesses found ({scalar_load} ptrs); consider vectorised loads (float4, etc.)")
        penalty += 15
    elif vector_load > 0:
        issues.append(f"Vectorised loads detected ({vector_load} vector types) — good")
    else:
        issues.append("No explicit pointer types found — unable to assess vectorisation")

    flag = "VECTORISATION_ISSUE" if penalty > 0 else None
    return penalty, issues, flag


def _check_lds_bank_conflict(code: str) -> tuple:
    """Heuristic: detect potential LDS (shared memory) bank conflicts."""
    issues: List[str] = []
    penalty = 0

    # Shared memory declarations
    shared_decls = re.findall(r'__shared__\s+\w+\s+(\w+)\s*\[', code)

    if shared_decls:
        # Check for column-major access pattern (common bank-conflict source)
        for var in shared_decls:
            # pattern: var[...][threadIdx.x]  →  column-major → likely conflict
            if re.search(rf'{var}\s*\[.*?\]\s*\[\s*threadIdx\.x\s*\]', code):
                issues.append(f"Shared variable '{var}' accessed column-major ([…][threadIdx.x]) — risk of LDS bank conflicts")
                penalty += 10

        if penalty == 0:
            issues.append("Shared memory access patterns look reasonable")
    else:
        issues.append("No shared memory usage detected — LDS bank-conflict check skipped")

    flag = "LDS_BANK_CONFLICT" if penalty > 0 else None
    return penalty, issues, flag


def _check_general_safety(code: str) -> tuple:
    """General best-practice checks."""
    issues: List[str] = []
    penalty = 0

    # Missing bounds check on global index
    if re.search(r'__global__', code) and not re.search(r'if\s*\(.*idx.*<', code):
        issues.append("Kernel may lack bounds checking on global index")
        penalty += 5

    # Unchecked hipMalloc / hipMemcpy return
    hip_calls = re.findall(r'\b(hipMalloc|hipMemcpy)\b', code)
    error_checks = re.findall(r'\b(hipSuccess|hipGetLastError)\b', code)
    if hip_calls and not error_checks:
        issues.append("HIP API calls without error checking detected")
        penalty += 5

    if not issues:
        issues.append("General safety checks passed")

    flag = "GENERAL_SAFETY" if penalty > 0 else None
    return penalty, issues, flag


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def _check_pattern_bonus(code: str, pattern: str, meta: dict) -> tuple:
    """Provide a score bonus and risk flag optimizations based on recognized patterns and execution drift metrics."""
    issues: List[str] = []
    penalty = 0

    if pattern == "wavefront_reduce":
        issues.append("Pattern Bonus: Wavefront reduction intrinsic usage reduces shared memory risks")
        penalty -= 15
    elif pattern == "fused_relu":
        issues.append("Pattern Bonus: Fused activation minimizes global memory roundtrips")
        penalty -= 10
    elif pattern == "tiled_shared":
        issues.append("Pattern Bonus: Tiled shared memory optimizes global bounds")
        penalty -= 5
    elif pattern == "vectorized":
        issues.append("Pattern Bonus: Vectorized elementwise maximizes memory bus utility")
        penalty -= 5
        
    # Apply dynamic dimensional safety drift calculation
    if meta and "dims" in meta:
        dims = meta["dims"]
        if "M" in dims and "N" in dims:
            m, n = dims["M"], dims["N"]
            # Alignment check: If dimensions are not aligned to typical 16/32 byte boundaries
            if m % 16 != 0 or n % 16 != 0:
                issues.append(f"Safety Drift Warning: Dimensions ({m}x{n}) misaligned with 16-byte boundaries. Assuming boundary guard logic liability.")
                penalty += 8  # Scale dynamic penalty based on misalignment

    flag = "PATTERN_OPTIMIZED" if penalty < 0 else "DRIFT_LIABILITY" if penalty > 0 else None
    return penalty, issues, flag

def analyse(code: str, pattern: str = None, meta: dict = None) -> Dict[str, Any]:
    """
    Run all safety checks on *code* and return a structured report.

    Returns
    -------
    dict
        score       : int        — safety score 0-100
        details     : list[str]  — human-readable findings
        risk_flags  : list[str]  — machine-readable flag ids
        attribution : list[str]  — provenance of rules applied
    """
    checks = [
        _check_wave64,
        _check_vectorisation,
        _check_lds_bank_conflict,
        _check_general_safety,
    ]

    total_penalty = 0
    all_details: List[str] = []
    risk_flags: List[str] = []

    for check_fn in checks:
        penalty, details, flag = check_fn(code)
        total_penalty += penalty
        all_details.extend(details)
        if flag:
            risk_flags.append(flag)

    if pattern:
        p_penalty, p_details, p_flag = _check_pattern_bonus(code, pattern, meta)
        total_penalty += p_penalty
        all_details.extend(p_details)
        if p_flag:
            risk_flags.append(p_flag)

    score = max(0, min(100, 100 - total_penalty))

    return {
        "score": score,
        "details": all_details,
        "risk_flags": risk_flags,
        "attribution": [
            "Template from ROCm 7.2 docs",
            "Safety rules v1.0 — ROCmForge Studio",
            "Wave64 guidance: AMD ISA reference",
        ],
    }
