#pragma once

#include "../IKernel.hpp"
#include "../Data/ImageDataSet.cuh"
#include "../Parallel/CubeProcessor.cuh"
#include "../DeviceHelper.cuh"
#include "../Math/PointHelper.cuh"

namespace Impacts
{
	namespace Cuda
	{
		namespace Kernels
		{
			__global__ void classify_cubes_kernel(float * dataset, int datasetSize, int3 datasetDim, float isoValue, int3 grid3dDim,
				int currentIter, size_t cubePerIter, size_t liveCubes, size_t totalCubes, int range, short memType,
				unsigned int * classifiedCubes)
			{
				int startIdx = DeviceHelper::GetThreadId() * range;

				Data::ImageDataSet imgDataSet(dataset, datasetSize);
				imgDataSet.SetDimensions(datasetDim.x, datasetDim.y, datasetDim.z);

				for (int idx = startIdx; (idx < liveCubes) && (idx < startIdx + range); idx++)
				{
					Parallel::CubeProcessor cubeProcessor(&imgDataSet, isoValue, grid3dDim, 0.0f, { 0, 0, 0 }, { 0.0f, 0.0f, 0.0f });

					int3 coords;
					Math::PointHelper::CalculateCubeCoordinates(&coords, idx, grid3dDim);

					cubeProcessor.ClassifyCube(coords.x, coords.y, coords.z, totalCubes, classifiedCubes);
				}
			}

			class ClassifyCubesKernel : public IKernel < float, unsigned int >
			{
				std::shared_ptr<thrust::device_vector<unsigned int>> m_output;

				int m_datasetSize;
				int3 m_datasetDim;
				float m_isoValue;
				int3 m_gridDim;

			public:

				ClassifyCubesKernel(int datasetSize, int3 datasetDim, float isoValue, int3 grid3Dim)
					: m_datasetSize(datasetSize), m_datasetDim(datasetDim), m_isoValue(isoValue), m_gridDim(grid3Dim)
				{
				}

				size_t Run(std::shared_ptr<thrust::device_vector<float>> const& input, KernelDimension const& kernelDim, int currentIter,
					size_t cubePerIter, size_t liveCubes, size_t totalCubes, int range, short memType) override
				{
					m_output = std::make_shared<thrust::device_vector<unsigned int>>(liveCubes);

					classify_cubes_kernel<<<kernelDim.grid, kernelDim.block>>>(thrust::raw_pointer_cast(input->data()),
						m_datasetSize, m_datasetDim, m_isoValue, m_gridDim,
						currentIter, cubePerIter, liveCubes, totalCubes, range, memType,
						thrust::raw_pointer_cast(m_output->data()));

					return m_output->size();
				}

				std::shared_ptr<thrust::device_vector<unsigned>> GetOutput() const override
				{
					return m_output;
				}
			};
		}
	}
}