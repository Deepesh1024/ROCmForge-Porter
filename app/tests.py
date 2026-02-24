"""
ROCmForge Studio — Test Suite (Nationals Build)
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
    assert body["hardware_backend_used"] in ("cpu_mock", "rocm_local", "mi300x_remote")
    assert body["safety_score"] is not None
    assert isinstance(body["risk_flags"], list)
    assert isinstance(body["attribution"], list)
    assert isinstance(body["reasoning_trace"], list)
    assert body["audit_id"] is not None
    assert "hipify" in body["data"]
    assert "classification" in body["data"]
    assert "safety" in body["data"]


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
    assert gen["template_used"] == "gemm_hip_template.cpp"
    assert "metadata" in gen
    assert body["safety_score"] is not None
    assert body["hardware_backend_used"] is not None


# ── /verify ──────────────────────────────────────────────────────

async def test_verify_endpoint():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/verify", json={
            "rocm_code": "// test",
            "meta": {"primitive": "gemm", "dtype": "float", "dims": {"M": 64, "N": 64, "K": 64}}
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    v = body["data"]["verification"]
    assert v["pass"] is True
    assert v["l2_norm"] < 1e-5
    assert body["safety_score"] is not None
    assert body["hardware_backend_used"] is not None


# ── /verify_remote ───────────────────────────────────────────────

async def test_verify_remote_endpoint():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/verify_remote", json={
            "rocm_code": "// test",
            "meta": {"primitive": "reduction", "dtype": "float", "dims": {"N": 512}}
        }, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["hardware_backend_used"] == "mi300x_remote"
    v = body["data"]["verification"]
    assert v["pass"] is True
    assert v["speed_ms"] == 0.08  # MI300X mock timing for reduction


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
    assert len(ea["at_dispatch_calls"]) >= 1
    assert len(ea["embedded_kernels"]) >= 1


# ── /register_mi300x_droplet ────────────────────────────────────

async def test_register_mi300x():
    async with AsyncClient(transport=TRANSPORT, base_url="http://test") as ac:
        resp = await ac.post("/register_mi300x_droplet", json={
            "region": "",
            "size": "gpu-mi300x8-1536gb-devcloud",
            "image": "rocm-7-1-software",
            "ssh_keys": [],
            "backups": False,
            "ipv6": False,
            "monitoring": False,
            "tags": [],
            "user_data": "",
            "vpc_uuid": "",
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
    assert "hardware" in body
    assert body["hardware"]["backend"] in ("cpu_mock", "rocm_local", "mi300x_remote")
