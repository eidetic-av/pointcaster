#include "./compiler.h"
#include "../logger.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace pc::kernels {

KernelCompiler::KernelCompiler() {
  pc::logger->info("Loading CUDA Kernel compiler");

  // Initialize CUDA context
  cuInit(0);
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, 0);
  CUcontext cuContext;
  cuCtxCreate(&cuContext, 0, cuDevice);

  // Define program to compile & run

  std::string test_code = R"(

extern "C"
__global__ void squareKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx];
    }
}

    )";

  nvrtcProgram prog;
  const char *src = test_code.c_str();
  nvrtcResult res =
      nvrtcCreateProgram(&prog, src, "squareKernel.cu", 0, NULL, NULL);
  if (res != NVRTC_SUCCESS) {
    pc::logger->error("Failed to create NVRTC program.");
    return;
  }

  // Compile the program
  const char *opts[] = {
      "--gpu-architecture=compute_70",
      "--device-as-default-execution-space"}; // Example options
  nvrtcResult compileRes = nvrtcCompileProgram(prog, 2, opts);

  // Check compilation
  if (compileRes != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char *log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    pc::logger->error("NVRTC Compilation failed:\n{}", log);
    delete[] log;
    nvrtcDestroyProgram(&prog);
    return;
  }

  pc::logger->debug("NVRTC compilation succeeded");

  // Get PTX code
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char *ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);

  pc::logger->debug("Retrieved PTX");

  // Load the module
  CUmodule cuModule;
  cuModuleLoadData(&cuModule, ptx);

  pc::logger->debug("Loaded module");

  // Get function pointer
  CUfunction kernelFunc;
  cuModuleGetFunction(&kernelFunc, cuModule, "squareKernel");

  pc::logger->debug("Retrieved function pointer");

  // Setup data using thrust::device_vector
  thrust::host_vector<float> h_data = {1.0f, 2.0f, 3.0f, 4.0f};
  thrust::device_vector<float> d_data = h_data;

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (h_data.size() + threadsPerBlock - 1) / threadsPerBlock;

  int h_size = static_cast<int>(h_data.size()); // store the size in a variable
  void *args[] = {thrust::raw_pointer_cast(&d_data[0]), &h_size};

  cudaDeviceSynchronize();

  CUresult result;
  result = cuModuleLoadData(&cuModule, ptx); // Example API call
  if (result != CUDA_SUCCESS) {
    const char *errorStr;
    cuGetErrorString(result, &errorStr);
    pc::logger->error("CUDA Error: {}", errorStr);
    return;
  } else {
    pc::logger->debug("No outstanding CUDA errors");
  }

  pc::logger->debug("Launching kernel...");

  cuLaunchKernel(kernelFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0,
                 nullptr, args, nullptr);

  pc::logger->debug("Launched");

  // Copy back the result using thrust::device_vector
  h_data = d_data;

  cudaDeviceSynchronize();

  // Cleanup
  cuModuleUnload(cuModule);
  delete[] ptx;
  nvrtcDestroyProgram(&prog);
  cuCtxDestroy(cuContext); // Destroy the CUDA context

  // Print the input and output
  std::cout << "Results:" << std::endl;
  for (float val : h_data) {
    std::cout << val << std::endl;
  }
}

} // namespace pc::kernels
