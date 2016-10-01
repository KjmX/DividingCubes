#pragma once

#include "../IKernel.hpp"
#include "../DeviceHelper.cuh"

#include <thrust/sort.h>

namespace Impacts
{
	namespace Cuda
	{
		namespace Kernels
		{
			struct CubeNotMax
			{
				__host__ __device__ bool operator()(const unsigned int v)
				{
					return v > 0;
				}
			};

			// TODO: Implement the fucking dichotomy method ! ffs !
			template <typename T>
			__global__ void compacted_cubes_size_kernel(T const * cubes, unsigned cubesSize, T limit, unsigned range, size_t * compactedSize)
			{
				unsigned startIdx = DeviceHelper::GetThreadId() * range;

				if ((startIdx == 0) && (cubes[startIdx] == limit))
					*compactedSize = 0;

				for (auto idx = startIdx; (*compactedSize == limit + 1) && (idx < cubesSize) && (idx < startIdx + range); idx++)
				{
					if ((idx < cubesSize - 1) && (cubes[idx] < limit) && (cubes[idx + 1] == limit))
						*compactedSize = idx + 1;
					else if ((idx == cubesSize - 1) && (cubes[idx] < limit))
						*compactedSize = cubesSize;
				}
			}

			template <typename T>
			class CompactCubesKernel : public IKernel < T, size_t >
			{
				std::shared_ptr<thrust::device_vector<size_t>> m_output;
				//std::shared_ptr<thrust::device_vector<unsigned int>> m_compactedCubes;

				T m_limit;

			public:

				CompactCubesKernel(T limit) : m_limit(limit)
				{
					m_output = std::make_shared<thrust::device_vector<size_t>>(1, m_limit + 1);
				}

				size_t Run(std::shared_ptr<thrust::device_vector<T>> const& input, KernelDimension const& kernelDim,
					int currentIter, size_t cubePerIter, size_t liveCubes, size_t totalCubes, int range, short memType) override
				{

					//m_compactedCubes = std::make_shared<thrust::device_vector<unsigned int>>(liveCubes);

					//thrust::copy_if(input->begin(), input->end(), m_compactedCubes->begin(), CubeNotMax());

					// TODO: To change
					thrust::sort(input->begin(), input->end());

					compacted_cubes_size_kernel<T> << <kernelDim.grid, kernelDim.block >> >(thrust::raw_pointer_cast(input->data()),
						liveCubes, m_limit,
						range, thrust::raw_pointer_cast(m_output->data()));

					return m_output->size();
				}

				std::shared_ptr<thrust::device_vector<size_t>> GetOutput() const override
				{
					return m_output;
				}
			};
		}
	}
}