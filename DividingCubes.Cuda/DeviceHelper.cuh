#pragma once

#include "pch.h"

namespace Impacts
{
	namespace Cuda
	{
		class DeviceHelper
		{
		public:
			__host__ static void GetCudaMemInfo(size_t * free, size_t * total)
			{
				gpuErrchk(cudaMemGetInfo(free, total));
			}

			__device__ static int GetThreadId()
			{
				int bId = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
				int tIdx = (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x);
				return (blockDim.x * blockDim.y * blockDim.z) * bId + tIdx;
			}
		};
	}
}
