#pragma once

#include "KernelDimension.hpp"
#include <memory>
#include <thrust/device_vector.h>

namespace Impacts
{
	namespace Cuda
	{
		template <typename TIn, typename TOut>
		class IKernel
		{
		public:
			virtual ~IKernel() {}
			virtual size_t Run(std::shared_ptr<thrust::device_vector<TIn>> const & input,
				KernelDimension const & kernelDim, int currentIter, size_t cubePerIter, size_t liveCubes, size_t totalCubes, int range, short memType) = 0;
			virtual std::shared_ptr<thrust::device_vector<TOut>> GetOutput() const = 0;
		};
	}
}