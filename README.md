# ROCmForge Studio — Backend

**Responsible-AI CUDA → ROCm porting engine.**  
Template-based code generation, deterministic safety analysis, CPU-reference verification, and full audit logging — zero hallucinated kernels.

---

## Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
uvicorn app.main:app --reload --port 8000

# 3. Run tests
python -m pytest app/tests.py -v
```

---

## API Endpoints

All endpoints require `Authorization: Bearer test-token`.

### POST `/parse`

Run mock hipify + CUDA primitive classification + safety analysis.

```bash
curl -s -X POST http://localhost:8000/parse \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "cuda_code": "#include <cuda_runtime.h>\n\n__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {\n  __shared__ float As[16][16];\n  float sum = 0.0f;\n  int row = blockIdx.y * 16 + threadIdx.y;\n  int col = blockIdx.x * 16 + threadIdx.x;\n  C[row*N+col] = sum;\n}\n\nint main() {\n  float *dA;\n  cudaMalloc(&dA, 1024*sizeof(float));\n  matmul<<<dim3(64),dim3(16,16)>>>(dA, dA, dA, 1024, 1024, 1024);\n  cudaFree(dA);\n}"
  }' | python -m json.tool
```

### POST `/generate`

Generate ROCm (HIP) code from templates.

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "primitive": "gemm",
    "meta": {"dtype": "float", "dims": {"M": 1024, "N": 1024, "K": 1024}}
  }' | python -m json.tool
```

### POST `/verify`

Verify generated ROCm code via CPU reference + mock ROCm output.

```bash
curl -s -X POST http://localhost:8000/verify \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "rocm_code": "// generated HIP code",
    "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 128, "N": 128, "K": 128}}
  }' | python -m json.tool
```

### GET `/health`

```bash
curl http://localhost:8000/health
```

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI endpoints
│   ├── config.py               # Constants
│   ├── models.py               # Pydantic models
│   ├── hipify_runner.py        # Mock hipify-clang
│   ├── primitive_classifier.py # GEMM / reduction / elementwise
│   ├── template_engine.py      # Deterministic code generation
│   ├── safety_engine.py        # Responsible-AI safety layer
│   ├── verifier.py             # CPU reference verification
│   ├── audit_logger.py         # JSON audit logs
│   ├── utils.py                # Helpers
│   ├── tests.py                # Pytest suite
│   └── sample_inputs/
│       ├── test_gemm.cu
│       ├── test_reduction.cu
│       └── test_elemwise.cu
├── templates/
│   ├── gemm_hip_template.cpp
│   ├── gemm_triton_template.py
│   ├── reduction_hip_template.cpp
│   ├── reduction_triton_template.py
│   ├── elemwise_hip_template.cpp
│   └── elemwise_triton_template.py
├── audit_logs/                 # Auto-created at runtime
├── requirements.txt
└── README.md
```

---

## Architecture

```
CUDA code ──► hipify_runner ──► primitive_classifier ──► template_engine
                                                            │
                                                    safety_engine ◄── verifier
                                                            │
                                                     audit_logger
```

- **No LLM code generation** — only templates produce kernels
- **Deterministic** — same input always produces same output
- **Responsible-AI** — wave64 awareness, vectorisation checks, attribution
- **Auditable** — every request logged with UUID, timestamp, full I/O

---

## Testing

```bash
# Run all tests
python -m pytest app/tests.py -v

# Run a specific test
python -m pytest app/tests.py::test_parse_endpoint -v
```

---

## License

MIT
