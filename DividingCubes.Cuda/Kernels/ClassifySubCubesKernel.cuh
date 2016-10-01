#pragma once

#include "../IKernel.hpp"
#include "../DeviceHelper.cuh"
#include "../Data/ImageDataSet.cuh"
#include "../Parallel/CubeProcessor.cuh"
#include "../Math/PointHelper.cuh"
#include "CompactCubesKernel.cuh"

namespace Impacts
{
	namespace Cuda
	{
		namespace Kernels
		{
			__global__ void classify_sub_cubes_kernel(float * dataset, size_t datasetSize, int3 datasetDim, unsigned int * cubes,
				float isoValue, int3 grid3dDim, float subDistance, int3 subDim, float3 subVoxSize,
				int currentIter, size_t subCubePerIter, size_t liveSubCubes, size_t totalSubCubes, int range, short memType,
				unsigned __int64 * classifiedSubCubes)
			{
				int startIdx = DeviceHelper::GetThreadId() * range;

				Data::ImageDataSet imgDataSet(dataset, datasetSize);
				imgDataSet.SetDimensions(datasetDim.x, datasetDim.y, datasetDim.z);

				for (int idx = startIdx; (idx < liveSubCubes) && (idx < startIdx + range); idx++)
				{
					Parallel::CubeProcessor cubeProcessor(&imgDataSet, isoValue, grid3dDim, subDistance, subDim, subVoxSize);

					auto subCubesPerCube = (subDim.x - 1) * (subDim.y - 1) * (subDim.z - 1);
					auto cubeIdx = cubes[(idx + (currentIter * subCubePerIter)) / subCubesPerCube];
					auto subCubeLocalIdx = (idx + (currentIter * subCubePerIter)) % subCubesPerCube;

					int3 cubeCoord;
					Math::PointHelper::CalculateCubeCoordinates(&cubeCoord, cubeIdx, grid3dDim);

					int3 subCubeLocalCoord;
					Math::PointHelper::CalculateCubeCoordinates(&subCubeLocalCoord, subCubeLocalIdx,
					{ subDim.x - 1, subDim.y - 1, subDim.z - 1 });

					// subX, subY, subZ = local sub cube position within a cube (not the in the whole grid)
					// cubeX, cuebY, cubeZ = cube position in the grid
					cubeProcessor.ClassifySubCube(subCubeLocalCoord.x, subCubeLocalCoord.y, subCubeLocalCoord.z,
						cubeCoord.x, cubeCoord.y, cubeCoord.z, idx, subCubeLocalIdx, cubeIdx,
						(grid3dDim.x) * (grid3dDim.y) * (grid3dDim.z) * (subDim.x - 1) * (subDim.y - 1) * (subDim.z - 1),
						classifiedSubCubes);
				}
			}

			class ClassifySubCubesKernel : public IKernel < unsigned int, unsigned __int64 >
			{
				std::shared_ptr<thrust::device_vector<unsigned __int64>> m_output;
				std::shared_ptr<thrust::device_vector<float>> m_dataset;

				size_t m_datasetSize;
				int3 m_datasetDim;
				//size_t m_cubesSize;
				float m_isoValue;
				int3 m_gridDim;
				float m_subDistance;
				int3 m_subDim;
				float3 m_subVoxSize;
				unsigned __int64 m_totalWorldSubCubes;

			public:

				ClassifySubCubesKernel(std::shared_ptr<thrust::device_vector<float>> dataset, size_t datasetSize, int3 datasetDim,
					float isoValue, int3 grid3Dim, float subDistance, int3 subDim, float3 subVoxSize)
					: m_dataset(dataset), m_datasetSize(datasetSize), m_datasetDim(datasetDim), m_isoValue(isoValue),
					m_gridDim(grid3Dim), m_subDistance(subDistance), m_subDim(subDim), m_subVoxSize(subVoxSize)
				{
					m_totalWorldSubCubes = m_gridDim.x * m_gridDim.y * m_gridDim.z * (m_subDim.x - 1) * (m_subDim.y - 1) * (m_subDim.z - 1);
				}

				size_t Run(std::shared_ptr<thrust::device_vector<unsigned>> const& input, KernelDimension const& kernelDim,
					int currentIter, size_t subCubePerIter, size_t liveSubCubes, size_t totalSubCubes, int range, short memType) override
				{
					m_output = std::make_shared<thrust::device_vector<unsigned __int64>>(liveSubCubes);

					classify_sub_cubes_kernel<<<kernelDim.grid, kernelDim.block>>>(thrust::raw_pointer_cast(m_dataset->data()), m_datasetSize, m_datasetDim,
						thrust::raw_pointer_cast(input->data()), m_isoValue, m_gridDim, m_subDistance, m_subDim, m_subVoxSize,
						currentIter, subCubePerIter, liveSubCubes, totalSubCubes, range, memType,
						thrust::raw_pointer_cast(m_output->data()));

					thrust::sort(m_output->begin(), m_output->end());

					thrust::device_vector<size_t> d_size(1, m_totalWorldSubCubes + 1);
					compacted_cubes_size_kernel<unsigned __int64><<<kernelDim.grid, kernelDim.block>>>(
						thrust::raw_pointer_cast(m_output->data()), m_output->size(),
						m_totalWorldSubCubes, range, thrust::raw_pointer_cast(d_size.data()));

					thrust::host_vector<size_t> h_size = d_size;

					return h_size[0];
				}

				std::shared_ptr<thrust::device_vector<unsigned __int64>> GetOutput() const override
				{
					return m_output;
				}
			};
		}
	}
}