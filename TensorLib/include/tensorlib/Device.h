#pragma once

#include "Tensor.h"
#include <memory>

namespace tl
{
    class IDevice
    {
    public:
        virtual ~IDevice() = default;

        virtual void Add(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0;
        virtual void Sub(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0;

        virtual void ElementMul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0;
        virtual void ElementDiv(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0;

        virtual void MatMul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0;
        virtual void MatMulATB(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0; //Matrix multiplication with transpose of A
        virtual void MatMulABT(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) = 0; //Matrix multiplication with transpose of B

        static std::unique_ptr<IDevice> CreateCPUDevice();
    };
}
