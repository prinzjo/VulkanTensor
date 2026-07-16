#include <tensorlib/Device.h>

#include "backend/cpu/CPUDevice.h"

namespace tl
{
    std::unique_ptr<IDevice> IDevice::CreateCPUDevice()
    {
        return std::make_unique<CPUDevice>();
    }
}
