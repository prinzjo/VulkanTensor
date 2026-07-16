#include <tensorlib/Tensor.h>

#include <iostream>

#include "tensorlib/Device.h"

int main()
{
    const auto device = tl::IDevice::CreateCPUDevice();

    tl::Tensor<float> a({2, 2});
    tl::Tensor<float> b({2, 2});

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 2;
    a(1, 1) = 1;

    b(0, 0) = 4;
    b(0, 1) = 1;
    b(1, 0) = 1;
    b(1, 1) = 4;

    tl::Tensor<float> c({2, 2});

    device->MatMul(a, b, c);

    std::cout << c << std::endl;
}
