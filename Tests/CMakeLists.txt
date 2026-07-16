add_executable(TensorTests
    src/main.cpp
)

target_include_directories(TensorTests
    PUBLIC ../TensorLib/include)

target_link_libraries(TensorTests
    PRIVATE TensorLib)
