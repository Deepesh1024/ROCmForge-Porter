"""
ROCmForge Studio — Pytest Tests

Covers all 3 endpoints: /parse, /generate, /verify
Uses httpx.AsyncClient for async FastAPI testing.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.asyncio

HEADERS = {"Authorization": "Bearer test-token"}

SAMPLE_GEMM_CUDA = """
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define K 1024

__global__ void matmul(const float* A, const float* B, float* C,
                       int M, int N, int K)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    matmul<<<dim3(64,64), dim3(16,16)>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
"""



async def test_parse_endpoint():
    """POST /parse should hipify, classify as GEMM, and return a safety score."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/parse", json={"cuda_code": SAMPLE_GEMM_CUDA}, headers=HEADERS)
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "success"
    assert body["audit_id"] is not None

    data = body["data"]
    # Hipify should have replaced CUDA APIs
    assert "hipMalloc" in data["hipify"]["hipified_code"]
    assert len(data["hipify"]["changes"]) > 0

    # Classification should detect GEMM
    assert data["classification"]["primitive"] == "gemm"
    assert data["classification"]["dtype"] == "float"

    # Safety engine should return a score
    assert 0 <= data["safety"]["score"] <= 100



async def test_generate_endpoint():
    """POST /generate should return template-generated ROCm code."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/generate",
            json={
                "primitive": "gemm",
                "meta": {"dtype": "float", "dims": {"M": 512, "N": 512, "K": 512}},
            },
            headers=HEADERS,
        )
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "success"

    data = body["data"]
    assert "gemm_hip_template.cpp" in data["generation"]["template_used"]
    assert "rocm_code" in data["generation"]
    assert len(data["generation"]["rocm_code"]) > 100

    # Safety check
    assert 0 <= data["safety"]["score"] <= 100



async def test_verify_endpoint():
    """POST /verify should return verification results with L2 norm pass."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/verify",
            json={
                "rocm_code": "// placeholder HIP code",
                "meta": {
                    "primitive": "gemm",
                    "dtype": "float",
                    "dims": {"M": 64, "N": 64, "K": 64},
                },
            },
            headers=HEADERS,
        )
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "success"

    v = body["data"]["verification"]
    assert v["pass"] is True
    assert v["l2_norm"] < 1e-5
    assert v["speed_ms"] == 42.0
    assert v["occupancy"] == 72.0
    assert v["bandwidth_gbps"] == 1800.0



async def test_auth_rejected():
    """Requests without valid Bearer token must return 401."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/parse", json={"cuda_code": "int main(){}"})
    assert resp.status_code == 401

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            json={"cuda_code": "int main(){}"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401
