# ROCmForge Studio — Backend API v3.0 (QuickPort)

Responsible-AI CUDA-to-ROCm porting engine with **Semantic Router + MI300X Expert System**.

## Architecture Overview

```
CUDA Code
   │
   ├─▶ Hipify Runner (CUDA→HIP API translation)
   │
   ├─▶ Semantic Extractor (LLM via Groq)
   │       └─▶ Returns: { primitive, pattern, memory_bound, shared_memory_used }
   │
   └─▶ Semantic Router
           ├─ primitive in TEMPLATE_MAP? ──▶ Template Engine (deterministic code gen)
           └─ unknown / not mapped?     ──▶ MI300X Rules Engine (4 arch-specific rules)
```

**Key Design:**
- **LLM classifies** mathematical intent (never generates code)
- **Template Engine** produces all kernel code deterministically
- **MI300X Rules Engine** handles fallback with Wave64, shuffle, grid, and LDS rules
- **Reasoning traces** are dynamically populated from actual routing decisions

## Quick Start

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Authentication

All POST endpoints require:
```
Authorization: Bearer dev-token
```

Set a custom token via env var: `BEARER_TOKEN=your-secret-token`

---

## API Reference

### `GET /health`

No auth required.

**Response:**
```json
{
  "status": "ok",
  "service": "ROCmForge Studio",
  "version": "3.0.0-quickport",
  "hardware": {
    "backend": "cpu_mock",
    "hipcc_available": false,
    "rocminfo_gpu": false,
    "mi300x_config_exists": false
  }
}
```

---

### `POST /parse`

Hipify CUDA code → LLM semantic classification → safety analysis → **semantic routing**.

**Request:**
```json
{
  "cuda_code": "__global__ void k(float* A) { int tid = threadIdx.x; __shared__ float smem[32]; smem[tid] = A[tid]; float val = __shfl_down_sync(0xffffffff, smem[tid], warpSize/2); }"
}
```

**Response (routed to Rules Engine):**
```json
{
  "status": "success",
  "audit_id": "2a4c1628179f",
  "safety_score": 80,
  "hardware_backend_used": "mi300x_remote",
  "risk_flags": ["VECTORISATION_ISSUE", "GENERAL_SAFETY"],
  "attribution": ["Safety rules v2.0 — ROCmForge Studio Nationals Build", "..."],
  "reasoning_trace": [
    "[1] Received parse request for primitive: unknown",
    "[2] Hardware backend detected: mi300x_remote",
    "[3] Primitive unknown or not in TEMPLATE_MAP — routed to MI300X rules engine",
    "[4] [ARCHITECTURE] MI300X uses Wave64. Hardcoding wavefront boundary from 32 to 64.",
    "[5] [COMPATIBILITY] Sync variants of shuffle are NVIDIA-specific. Reverting to base shuffle for ROCm compatibility.",
    "[6] [SYNTAX] Mapping standard grid coordinates to HIP.",
    "[7] [MEMORY] Local Data Share (LDS) allocated. Note: MI300X LDS occupancy limit is 64KB per CU."
  ],
  "data": {
    "hipify": { "hipified_code": "...", "changes": [], "method": "mock hipify (regex)" },
    "classification": {
      "primitive": "unknown",
      "dtype": "float",
      "shape": "N=1024",
      "pattern": "unknown",
      "semantic_extraction": { "primitive": "unknown", "pattern": "unknown", "memory_bound": false, "shared_memory_used": false }
    },
    "safety": { "score": 80, "details": ["..."], "risk_flags": ["VECTORISATION_ISSUE"] },
    "route": "rules_engine",
    "rules_trace": [
      "[ARCHITECTURE] MI300X uses Wave64. Hardcoding wavefront boundary from 32 to 64.",
      "[COMPATIBILITY] Sync variants of shuffle are NVIDIA-specific. Reverting to base shuffle for ROCm compatibility.",
      "[SYNTAX] Mapping standard grid coordinates to HIP.",
      "[MEMORY] Local Data Share (LDS) allocated. Note: MI300X LDS occupancy limit is 64KB per CU."
    ],
    "rules_engine_output": "__global__ void custom(float* A) { int tid = hipThreadIdx.x; __shared__ float smem[64]; ... }"
  }
}
```

---

### `POST /generate`

Generate ROCm HIP + Triton code from templates (NO LLM code gen).

**Request:**
```json
{
  "primitive": "gemm",
  "meta": {
    "dtype": "float",
    "dims": {"M": 1024, "N": 1024, "K": 1024},
    "tile_size": 16,
    "block_size": 256
  }
}
```

| `primitive` values | Description |
|---|---|
| `"gemm"` | Matrix multiplication |
| `"fused_matmul"` | GEMM + activation fusion |
| `"reduction"` | Sum/max reduction |
| `"elementwise"` | Pointwise ops (add, mul, relu) |
| `"softmax"` | Fused softmax |
| `"layernorm"` | Layer normalization |
| `"conv"` | Convolution |
| `"attention"` | Flash attention |
| `"dropout"` | Fused dropout |

**Response:**
```json
{
  "status": "success",
  "audit_id": "...",
  "safety_score": 100,
  "execution_confidence": null,
  "hardware_backend_used": "cpu_mock",
  "risk_flags": [],
  "attribution": ["Template: AMD ROCm 7.2 HIP GEMM example, wave64-adapted", "..."],
  "reasoning_trace": ["..."],
  "data": {
    "generation": {
      "rocm_code": "// HIP C++ GEMM kernel code...",
      "triton_code": "# Triton GEMM kernel code...",
      "template_used": "gemm_hip_template.cpp",
      "metadata": {
        "primitive": "GEMM",
        "source": "AMD ROCm 7.2 examples + manual wave64 adaptation"
      }
    },
    "safety": {
      "score": 100,
      "details": [],
      "risk_flags": []
    }
  }
}
```

---

### `POST /verify`

Verify code with **real CPU timing** + **MI300X cache-first** lookup.

**Request:**
```json
{
  "rocm_code": "// generated HIP code",
  "meta": {
    "primitive": "gemm",
    "dtype": "float",
    "dims": {"M": 1024, "N": 1024, "K": 1024}
  }
}
```

**Response (cache HIT):**
```json
{
  "status": "success",
  "audit_id": "...",
  "safety_score": 100,
  "execution_confidence": 95,
  "hardware_backend_used": "mi300x_remote_cached",
  "risk_flags": [],
  "attribution": ["..."],
  "reasoning_trace": ["..."],
  "data": {
    "verification": {
      "pass": true,
      "cache_hit": true,
      "cache_key": "gemm:1024x1024:float",
      "cpu_reference_time_ms": 1.826,
      "gpu_time_ms": 0.118,
      "speedup_vs_cpu": 15.47,
      "l2_norm": 7.4e-08,
      "occupancy": 83,
      "bandwidth_gbps": 1450,
      "mfma_util": 76,
      "cpu_output_sample": [12.97, -16.47, -48.89],
      "hardware_backend_used": "mi300x_remote_cached"
    },
    "safety": { "score": 100, "details": [], "risk_flags": [] }
  }
}
```

**Response (cache MISS — falls back to CPU):**
```json
{
  "status": "success",
  "execution_confidence": 70,
  "hardware_backend_used": "cpu_fallback",
  "data": {
    "verification": {
      "pass": true,
      "cache_hit": false,
      "cache_key": "gemm:77x77:float",
      "cpu_reference_time_ms": 0.041,
      "gpu_time_ms": null,
      "speedup_vs_cpu": null,
      "l2_norm": 3.2e-08,
      "occupancy": "N/A",
      "bandwidth_gbps": "N/A",
      "mfma_util": "N/A",
      "cpu_output_sample": [1.23, -0.45, 2.67],
      "hardware_backend_used": "cpu_fallback"
    }
  }
}
```

**Available cache keys** (in `cache/mi300x_cache.json`):
```
gemm:64x64:float       gemm:128x128:float     gemm:256x256:float
gemm:512x512:float     gemm:1024x1024:float   gemm:1024x1024:half
gemm:2048x2048:float   gemm:4096x4096:float
reduction:512:float    reduction:1024:float    reduction:2048:float
reduction:4096:float   reduction:1024:half
elementwise:512:float  elementwise:1024:float  elementwise:2048:float
elementwise:1024x1024:float                    elementwise:1024x1024:half
```

---

### `POST /verify_remote`

Same as `/verify` but always attempts MI300X cache first.

**Request/Response:** identical to `/verify`.

---

### `POST /parse_extension`

Parse a PyTorch C++ extension, detect `AT_DISPATCH`, extract embedded CUDA kernels.

**Request:**
```json
{
  "extension_code": "#include <torch/extension.h>\n__global__ void k(float* A) { ... }\ntorch::Tensor forward(torch::Tensor x) {\n  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), \"fwd\", [&] { k<<<1,256>>>(x.data_ptr<scalar_t>()); });\n  return x;\n}"
}
```

**Response:**
```json
{
  "status": "success",
  "audit_id": "...",
  "safety_score": 75,
  "execution_confidence": null,
  "hardware_backend_used": "cpu_mock",
  "data": {
    "extension_analysis": {
      "is_pytorch_extension": true,
      "at_dispatch_calls": [
        {"scalar_type_expr": "x.scalar_type()", "op_name": "fwd", "position": 120}
      ],
      "embedded_kernels": ["__global__ void k(float* A) { ... }"],
      "cuda_launch_sites": 1,
      "torch_tensors": 2,
      "extracted_cuda_code": "__global__ void k(float* A) { ... }"
    },
    "classification": {
      "primitive": "elementwise",
      "dtype": "float",
      "dims": {"N": 1024}
    },
    "safety": { "score": 75, "details": ["..."], "risk_flags": [] }
  }
}
```

---

### `POST /register_mi300x_droplet`

Register an MI300X droplet configuration. Changes hardware detection.

**Request:**
```json
{
  "region": "",
  "size": "gpu-mi300x8-1536gb-devcloud",
  "image": "rocm-7-1-software",
  "ssh_keys": [],
  "backups": false,
  "ipv6": false,
  "monitoring": false,
  "tags": [],
  "user_data": "",
  "vpc_uuid": ""
}
```

**Response:**
```json
{
  "status": "success",
  "audit_id": "...",
  "hardware_backend_used": "mi300x_remote",
  "data": {
    "registered": true,
    "config_path": "/path/to/mi300x_config.json",
    "size": "gpu-mi300x8-1536gb-devcloud",
    "image": "rocm-7-1-software"
  }
}
```

---

## Error Response

All endpoints return this on failure (never crashes):
```json
{
  "status": "error",
  "message": "Description of what went wrong",
  "hardware_backend_used": "cpu_fallback",
  "data": null
}
```

---

## Frontend Integration Cheatsheet

```javascript
const API_BASE = "https://your-server.com";
const TOKEN = "dev-token";

const headers = {
  "Authorization": `Bearer ${TOKEN}`,
  "Content-Type": "application/json",
};

// Parse CUDA code
const parseRes = await fetch(`${API_BASE}/parse`, {
  method: "POST",
  headers,
  body: JSON.stringify({ cuda_code: cudaSource }),
});

// Generate ROCm code
const genRes = await fetch(`${API_BASE}/generate`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    primitive: "gemm",
    meta: { dtype: "float", dims: { M: 1024, N: 1024, K: 1024 } },
  }),
});

// Verify
const verifyRes = await fetch(`${API_BASE}/verify`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    rocm_code: generatedCode,
    meta: { primitive: "gemm", dtype: "float", dims: { M: 1024, N: 1024, K: 1024 } },
  }),
});

const data = await verifyRes.json();
// data.execution_confidence  → 95 (cache hit) or 70 (cpu fallback)
// data.data.verification.speedup_vs_cpu  → 15.47
// data.data.verification.cache_hit  → true/false
```

---

## Testing

```bash
python -m pytest app/tests.py -v    # 12 tests
```

## Deployment

```bash
# Production (with gunicorn)
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or simple
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BEARER_TOKEN` | `dev-token` | API auth token |
| `GROQ_API_KEY` | (required) | Groq API key for LLM semantic extraction |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |

## Project Structure

```
backend/
├── app/
│   ├── main.py                 ← FastAPI (7 endpoints) + Semantic Router
│   ├── config.py               ← Constants + paths + Groq config
│   ├── models.py               ← Pydantic request/response models
│   ├── hardware_detector.py    ← rocm_local / cpu_mock / mi300x_remote
│   ├── hipify_runner.py        ← Subprocess hipify-clang + mock fallback
│   ├── primitive_classifier.py ← LLM-based semantic CUDA classifier (Groq)
│   ├── mi300x_rules.py         ← MI300X deterministic expert rules engine [NEW]
│   ├── pytorch_parser.py       ← AT_DISPATCH + kernel extraction
│   ├── template_engine.py      ← YAML metadata + placeholder fill (UNCHANGED)
│   ├── responsible_ai.py       ← Dynamic reasoning trace + attribution
│   ├── safety_engine.py        ← Wave64/vectorisation/LDS checks
│   ├── verifier.py             ← Real CPU timing + MI300X cache
│   ├── mi300x_runner.py        ← Droplet registration
│   ├── llm_explainer.py        ← LLM explanations (Groq)
│   ├── audit_logger.py         ← Extended JSON audit logs
│   ├── utils.py                ← L2 norm, helpers
│   └── sample_inputs/          ← Test .cu files
├── cache/
│   └── mi300x_cache.json       ← Pre-recorded MI300X metrics (18 entries)
├── templates/                  ← HIP C++ & Triton Python templates (9 primitives)
├── audit_logs/                 ← Auto-generated per request
├── requirements.txt
└── README.md
```

## MI300X Rules Engine

When the LLM classifier returns `"unknown"` or a primitive not in `TEMPLATE_MAP`, the code is routed through `mi300x_rules.apply_rules()` which applies these deterministic transformations:

| Rule | Match | Replacement | Trace Category |
|------|-------|-------------|----------------|
| Wave64 | `warpSize` / `32` in warp context | `64` | `[ARCHITECTURE]` |
| Shuffle | `__shfl_down_sync` | `__shfl_down` | `[COMPATIBILITY]` |
| Grid coords | `blockIdx` / `threadIdx` | `hipBlockIdx` / `hipThreadIdx` | `[SYNTAX]` |
| LDS detect | `__shared__` | No mutation (advisory) | `[MEMORY]` |
