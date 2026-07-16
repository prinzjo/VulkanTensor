#include "CPUDevice.h"

namespace tl
{
    CPUDevice::CPUDevice() = default;
    CPUDevice::~CPUDevice() = default;

    void CPUDevice::Add(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        assert(a.Shape() == out.Shape() && b.Shape() == out.Shape());

        const float* __restrict pa = a.Data();
        const float* __restrict pb = b.Data();
        float* __restrict po = out.Data();

        const size_t n = a.Size();
        for (size_t i = 0; i < n; i++)
            po[i] = pa[i] + pb[i];
    }

    void CPUDevice::Sub(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        assert(a.Shape() == out.Shape() && b.Shape() == out.Shape());

        const float* __restrict pa = a.Data();
        const float* __restrict pb = b.Data();
        float* __restrict po = out.Data();

        const size_t n = a.Size();
        for (size_t i = 0; i < n; i++)
            po[i] = pa[i] - pb[i];
    }

    void CPUDevice::ElementMul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        assert(a.Shape() == out.Shape() && b.Shape() == out.Shape());

        const float* __restrict pa = a.Data();
        const float* __restrict pb = b.Data();
        float* __restrict po = out.Data();

        const size_t n = a.Size();
        for (size_t i = 0; i < n; i++)
            po[i] = pa[i] * pb[i];
    }

    void CPUDevice::ElementDiv(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        assert(a.Shape() == out.Shape() && b.Shape() == out.Shape());

        const float* __restrict pa = a.Data();
        const float* __restrict pb = b.Data();
        float* __restrict po = out.Data();

        const size_t n = a.Size();
        for (size_t i = 0; i < n; i++)
            po[i] = pa[i] / pb[i];
    }

    void CPUDevice::MatMul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        const auto& shapeA = a.Shape();
        const auto& shapeB = b.Shape();
        const size_t rank = shapeA.size();

        assert(rank >= 2 && shapeB.size() == rank);

        const size_t M = shapeA[rank - 2];
        const size_t K = shapeA[rank - 1];
        const size_t N = shapeB[rank - 1];

        assert(shapeB[rank - 2] == K && "MatMul: inner dimensions must match");

        size_t batchCount = 1;
        for (size_t i = 0; i + 2 < rank; i++)
        {
            assert(shapeA[i] == shapeB[i] && "MatMul: batch dimensions must match");
            batchCount *= shapeA[i];
        }

        const float* pa = a.Data();
        const float* pb = b.Data();
        float* po = out.Data();

        for (size_t batch = 0; batch < batchCount; batch++)
        {
            const float* A = pa + batch * M * K;
            const float* B = pb + batch * K * N;
            float* C = po + batch * M * N;

            for (size_t m = 0; m < M; m++)
            {
                float* cRow = C + m * N;
                std::fill_n(cRow, N, 0.f);

                for (size_t k = 0; k < K; k++)
                {
                    const float aVal = A[m * K + k];
                    const float* bRow = B + k * N;

                    for (size_t n = 0; n < N; n++)
                        cRow[n] += aVal * bRow[n];
                }
            }
        }
    }

    void CPUDevice::MatMulATB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        const auto& shapeA = a.Shape();
        const auto& shapeB = b.Shape();
        const size_t rank = shapeA.size();

        assert(rank >= 2 && shapeB.size() == rank);

        const size_t K = shapeA[rank - 2];
        const size_t M = shapeA[rank - 1];
        const size_t N = shapeB[rank - 1];

        assert(shapeB[rank - 2] == K && "MatMulATB: contracted dimensions must match");

        size_t batchCount = 1;
        for (size_t i = 0; i + 2 < rank; i++)
        {
            assert(shapeA[i] == shapeB[i] && "MatMulATB: batch dimensions must match");
            batchCount *= shapeA[i];
        }

        const float* pa = a.Data();
        const float* pb = b.Data();
        float* po = out.Data();

        for (size_t batch = 0; batch < batchCount; batch++)
        {
            const float* A = pa + batch * K * M;
            const float* B = pb + batch * K * N;
            float* C = po + batch * M * N;

            std::fill_n(C, M * N, 0.f);

            for (size_t k = 0; k < K; k++)
            {
                const float* aRow = A + k * M;
                const float* bRow = B + k * N;

                for (size_t m = 0; m < M; m++)
                {
                    const float aVal = aRow[m];
                    float* cRow = C + m * N;

                    for (size_t n = 0; n < N; n++)
                        cRow[n] += aVal * bRow[n];
                }
            }
        }
    }

    void CPUDevice::MatMulABT(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out)
    {
        const auto& shapeA = a.Shape();
        const auto& shapeB = b.Shape();
        const size_t rank = shapeA.size();

        assert(rank >= 2 && shapeB.size() == rank);

        const size_t M = shapeA[rank - 2];
        const size_t K = shapeA[rank - 1];
        const size_t N = shapeB[rank - 2];

        assert(shapeB[rank - 1] == K && "MatMulABT: contracted dimensions must match");

        size_t batchCount = 1;
        for (size_t i = 0; i + 2 < rank; i++)
        {
            assert(shapeA[i] == shapeB[i] && "MatMulABT: batch dimensions must match");
            batchCount *= shapeA[i];
        }

        const float* pa = a.Data();
        const float* pb = b.Data();
        float* po = out.Data();

        for (size_t batch = 0; batch < batchCount; batch++)
        {
            const float* A = pa + batch * M * K;
            const float* B = pb + batch * N * K;
            float* C = po + batch * M * N;

            for (size_t m = 0; m < M; m++)
            {
                const float* aRow = A + m * K;
                float* cRow = C + m * N;

                for (size_t n = 0; n < N; n++)
                {
                    const float* bRow = B + n * K;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; k++)
                        sum += aRow[k] * bRow[k];

                    cRow[n] = sum;
                }
            }
        }
    }
}
