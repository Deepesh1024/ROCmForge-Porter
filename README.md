# ROCmForge Studio – Nationals Backend v2.0

Responsible-AI CUDA-to-ROCm porting engine with hardware-adaptive execution.

## Features

| Feature | Status |
|---------|--------|
| Hardware detection (ROCm local / MI300X remote / CPU mock) | ✅ |
| CPU fallback with deterministic timings | ✅ |
| MI300X remote mock execution | ✅ |
| Mock hipify (regex) + real hipify-clang fallback | ✅ |
| Deterministic template-based code gen (NO LLM) | ✅ |
| PyTorch extension parser (AT_DISPATCH) | ✅ |
| Responsible-AI: safety_score, risk_flags, attribution, reasoning_trace | ✅ |
| Full audit logging with provenance | ✅ |
| Bearer token auth (`dev-token`) | ✅ |
| Async endpoints with 30s timeout | ✅ |

## Quick Start

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/parse` | Hipify + classify + safety analysis |
| POST | `/generate` | Template-based ROCm code generation |
| POST | `/verify` | Hardware-adaptive verification |
| POST | `/verify_remote` | MI300X remote verification |
| POST | `/parse_extension` | PyTorch extension parser |
| POST | `/register_mi300x_droplet` | Register MI300X config |
| GET | `/health` | Health + backend info |

All POST endpoints require: `Authorization: Bearer dev-token`

## curl Examples

### Health Check
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

### Parse CUDA
```bash
curl -X POST http://localhost:8000/parse \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"cuda_code": "__global__ void k(float* A) { cudaMalloc(&A, 1024); }"}'
```

### Generate ROCm Code
```bash
curl -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"primitive": "gemm", "meta": {"dtype": "float", "dims": {"M": 1024, "N": 1024, "K": 1024}}}'
```

### Verify
```bash
curl -X POST http://localhost:8000/verify \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"rocm_code": "// test", "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 128, "N": 128, "K": 128}}}'
```

### Verify Remote (MI300X)
```bash
curl -X POST http://localhost:8000/verify_remote \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"rocm_code": "// test", "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 128, "N": 128, "K": 128}}}'
```

### Register MI300X Droplet
```bash
curl -X POST http://localhost:8000/register_mi300x_droplet \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"region": "", "size": "gpu-mi300x8-1536gb-devcloud", "image": "rocm-7-1-software", "ssh_keys": [], "backups": false, "ipv6": false, "monitoring": false, "tags": [], "user_data": "", "vpc_uuid": ""}'
```

### Parse PyTorch Extension
```bash
curl -X POST http://localhost:8000/parse_extension \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"extension_code": "#include <torch/extension.h>\ntorch::Tensor forward(torch::Tensor x) { return x; }"}'
```

## Testing

```bash
python -m pytest app/tests.py -v
```

## Architecture

```
CUDA code → hipify_runner → primitive_classifier → template_engine → safety_engine
                                                                         ↓
                                                              responsible_ai (scoring)
                                                                         ↓
                                                              verifier (CPU/ROCm/MI300X)
                                                                         ↓
                                                              audit_logger (JSON logs)
```

## Project Structure

```
backend/
├── app/
│   ├── main.py               ← FastAPI (7 endpoints)
│   ├── config.py              ← Constants + timings
│   ├── models.py              ← Pydantic models
│   ├── hardware_detector.py   ← rocm_local / cpu_mock / mi300x_remote
│   ├── hipify_runner.py       ← Subprocess + mock fallback
│   ├── primitive_classifier.py ← Regex GEMM/reduction/elementwise
│   ├── pytorch_parser.py      ← AT_DISPATCH + kernel extraction
│   ├── template_engine.py     ← YAML metadata + placeholder fill
│   ├── responsible_ai.py      ← Reasoning trace + attribution
│   ├── safety_engine.py       ← Wave64/vectorisation/LDS checks
│   ├── verifier.py            ← Hardware-adaptive verification
│   ├── mi300x_runner.py       ← Remote mock execution
│   ├── audit_logger.py        ← Extended JSON audit logs
│   ├── utils.py               ← L2 norm, helpers
│   └── sample_inputs/         ← Test .cu files
├── templates/                 ← HIP C++ & Triton templates
├── audit_logs/                ← Auto-generated
├── requirements.txt
└── README.md
```
