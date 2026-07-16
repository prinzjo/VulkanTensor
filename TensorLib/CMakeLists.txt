add_library(TensorLib
        include/tensorlib/Tensor.h
        include/tensorlib/Device.h
        src/backend/cpu/CPUDevice.h
        src/backend/cpu/CPUDevice.cpp
        src/Device.cpp
)

target_include_directories(TensorLib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(TensorLib PUBLIC cxx_std_23)
