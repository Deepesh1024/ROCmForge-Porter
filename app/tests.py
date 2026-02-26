"""
ROCmForge Studio — Test Suite (Nationals Final v2.1)
"""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app

pytestmark = pytest.mark.asyncio

HEADERS = {"Authorization": "Bearer dev-token"}
TRANSPORT = ASGITransport(app=app)


# ── /parse ───────────────────────────────────────────────────────

async def test_parse_endpoint():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/parse", json={
            "cuda_code": "__global__ void k(float* A) { cudaMalloc(&A, 1024); }"
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["safety_score"] is not None
    assert isinstance(body["risk_flags"], list)
    assert isinstance(body["attribution"], list)
    assert isinstance(body["reasoning_trace"], list)
    assert body["audit_id"] is not None
    assert "hipify" in body["data"]
    assert "classification" in body["data"]


# ── /generate ────────────────────────────────────────────────────

async def test_generate_endpoint():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/generate", json={
            "primitive": "gemm",
            "meta": {"dtype": "float", "dims": {"M": 128, "N": 128, "K": 128}}
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    gen = body["data"]["generation"]
    assert "rocm_code" in gen
    assert gen["template_used"] == "tiled_gemm_hip_template.cpp"
    assert body["safety_score"] is not None


# ── /verify (cache HIT — 128x128 is in cache) ───────────────────

async def test_verify_cache_hit():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/verify", json={
            "rocm_code": "// test",
            "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 128, "N": 128, "K": 128}}
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    v = body["data"]["verification"]
    assert v["cache_hit"] is True
    assert v["hardware_backend_used"] == "mi300x_remote_cached"
    assert v["gpu_time_ms"] is not None
    assert v["cpu_reference_time_ms"] > 0
    assert v["speedup_vs_cpu"] is not None
    assert body["execution_confidence"] == 95


# ── /verify (cache MISS — 77x77 is NOT in cache) ────────────────

async def test_verify_cache_miss():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/verify", json={
            "rocm_code": "// test",
            "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 77, "N": 77, "K": 77}}
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    v = body["data"]["verification"]
    assert v["cache_hit"] is False
    assert v["hardware_backend_used"] == "cpu_fallback"
    assert v["gpu_time_ms"] is None
    assert v["cpu_reference_time_ms"] > 0
    assert body["execution_confidence"] == 70


# ── /verify_remote (cache HIT — 1024x1024 is in cache) ──────────

async def test_verify_remote_cache_hit():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/verify_remote", json={
            "rocm_code": "// test",
            "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 1024, "N": 1024, "K": 1024}}
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    v = body["data"]["verification"]
    assert v["cache_hit"] is True
    assert v["hardware_backend_used"] == "mi300x_remote_cached"
    assert v["gpu_time_ms"] == 0.118
    assert body["execution_confidence"] == 95


# ── /parse_extension ─────────────────────────────────────────────

async def test_parse_extension_endpoint():
    ext_code = '''
    #include <torch/extension.h>
    __global__ void matmul_kernel(float* A, float* B, float* C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) C[idx] = A[idx] * B[idx];
    }
    torch::Tensor forward(torch::Tensor input) {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward", [&] {
            matmul_kernel<<<1, 256>>>(input.data_ptr<scalar_t>(), nullptr, nullptr, 256);
        });
        return input;
    }
    '''
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/parse_extension", json={
            "extension_code": ext_code,
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    ea = body["data"]["extension_analysis"]
    assert ea["is_pytorch_extension"] is True


# ── /register_mi300x_droplet ────────────────────────────────────

async def test_register_mi300x():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/register_mi300x_droplet", json={
            "region": "",
            "size": "gpu-mi300x8-1536gb-devcloud",
            "image": "rocm-7-1-software",
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["data"]["registered"] is True


# ── Auth rejection ───────────────────────────────────────────────

async def test_auth_rejected():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/parse", json={"cuda_code": "test"})
    assert resp.status_code == 401


async def test_wrong_token():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/parse", json={"cuda_code": "test"},
                             headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


# ── /health ──────────────────────────────────────────────────────

async def test_health():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["hardware"]["backend"] in ("cpu_mock", "rocm_local", "mi300x_remote")


# ── /parse_project ───────────────────────────────────────────────

async def test_parse_project_endpoint():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/parse_project", json={
            "files": [
                {"filename": "kernel1.cu", "content": "__global__ void matmul(float* A){ for(int i=0;i<10;i++){ for(int j=0;j<10;j++){ A[i]=1; } } }"},
                {"filename": "kernel2.cu", "content": "__global__ void reduce(float* A){ float val = __shfl_down_sync(0xffffffff, A[0], 16); }"}
            ]
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    project_map = body["data"]["project_map"]
    assert "kernel1.cu" in project_map
    assert "kernel2.cu" in project_map
    assert project_map["kernel1.cu"]["classification"]["primitive"] == "gemm"
    assert project_map["kernel2.cu"]["classification"]["primitive"] == "reduction"


# ── /generate_project ────────────────────────────────────────────

async def test_generate_project_endpoint():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/generate_project", json={
            "files": [
                # Mocking metadata inside content
                {"filename": "kernel1.cu", "content": '{"primitive": "gemm", "pattern": "tiled_shared", "dims": {"M": 128}}'},
                {"filename": "kernel2.cu", "content": '{"primitive": "reduction", "pattern": "wavefront_reduce", "dims": {"N": 1024}}'}
            ]
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    gen_map = body["data"]["generated_map"]
    assert "kernel1.cu" in gen_map
    assert "kernel2.cu" in gen_map
    assert "tiled_gemm_hip_template.cpp" in gen_map["kernel1.cu"]["generation"]["template_used"]
    assert "wavefront_reduction_hip_template.cpp" in gen_map["kernel2.cu"]["generation"]["template_used"]
