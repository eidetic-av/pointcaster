#include "./compiler.h"
#include "../logger.h"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#define NVRTC_SAFE_CALL(Name, x)                                               \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "\nerror: " << Name << " failed with error "                \
		<< nvrtcGetErrorString(result);                                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__device__ void function(int x) { }

struct functor {

  __device__ void operator()(int x) {
    function(x);
  }
};

void for_each() {
  pc::logger->debug("for_each");
  thrust::device_vector<int> d_vec(3);
  d_vec[0] = 0;
  d_vec[0] = 1;
  d_vec[0] = 2;
  thrust::for_each(d_vec.begin(), d_vec.end(), functor());
}

std::string function_source = R"(

__device__ void function(int x) {
  printf("%d\n", x);
}

)";

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
  nvrtcProgram program;
  NVRTC_SAFE_CALL("create_program", nvrtcCreateProgram(&program, function_source.c_str(), "function", 0, nullptr, nullptr));

  // compile the program
  const char *options[] = {"-rdc=true"}; 
  nvrtcResult compile_result = nvrtcCompileProgram(program, 1, options);

  // obtain compilation log
  std::size_t log_size;
  NVRTC_SAFE_CALL("get_log_size", nvrtcGetProgramLogSize(program, &log_size));
  if (log_size > 1) {
    char* log;
    NVRTC_SAFE_CALL("get_log", nvrtcGetProgramLog(program, &log[0]));
    std::cout << &log[0] << "\n";
  }

  if (compile_result != NVRTC_SUCCESS) exit(1);

  // obtain PTX from the program
  std::size_t ptx_size;
  NVRTC_SAFE_CALL("get_ptx_size", nvrtcGetPTXSize(program, &ptx_size));
  char *ptx = new char[ptx_size];
  NVRTC_SAFE_CALL("get_ptx", nvrtcGetPTX(program, ptx));

  // Destroy the program
  NVRTC_SAFE_CALL("destroy_program", nvrtcDestroyProgram(&program));

  // Load precompiled relocatable source with call to external function and link
  // it together with NVRTC-compiled function

  CUlinkState linker;
  CUDA_SAFE_CALL(cuLinkCreate(0, NULL, NULL, &linker));
  CUDA_SAFE_CALL(
      cuLinkAddFile(linker, CU_JIT_INPUT_PTX, "functor.ptx", 0, NULL, NULL));
  CUDA_SAFE_CALL(cuLinkAddData(linker, CU_JIT_INPUT_PTX, (void *)ptx, ptx_size,
			       "function.ptx", 0, NULL, NULL));
  void *cubin;
  CUDA_SAFE_CALL(cuLinkComplete(linker, &cubin, NULL));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, cubin, 0, NULL, NULL));
  CUDA_SAFE_CALL(cuLinkDestroy(linker));

}

} // namespace pc::kernels
