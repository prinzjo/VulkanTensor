#pragma once

#include <tensorlib/Device.h>

namespace tl
{
    class CPUDevice : public IDevice
    {
    public:
        CPUDevice();
        ~CPUDevice() override;

        void Add(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;
        void Sub(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;

        void ElementMul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;
        void ElementDiv(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;

        void MatMul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;
        void MatMulATB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;
        void MatMulABT(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) override;
    };
}
