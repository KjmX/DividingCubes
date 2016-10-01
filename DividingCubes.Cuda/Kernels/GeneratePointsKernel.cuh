#pragma once

#include "../IKernel.hpp"
#include "../DeviceHelper.cuh"
#include "../Data/ImageDataSet.cuh"
#include "../Parallel/CubeProcessor.cuh"
#include "../Math/PointHelper.cuh"
#include "Geometry/Vertex.hpp"

namespace Impacts
{
	namespace Cuda
	{
		namespace Kernels
		{
			__global__ void generate_points_kernel(float * dataset, size_t datasetSize, int3 datasetDim, unsigned __int64 * subCubes,
				float isoValue, int3 grid3dDim, float3 voxSize, float subDistance, int3 subDim, float3 subVoxSize,
				int currentIter, size_t subCubePerIter, size_t liveSubCubes, size_t totalSubCubes, int range, short memType,
				Geometry::Vertex * points)
			{
				int startIdx = DeviceHelper::GetThreadId() * range;

				Data::ImageDataSet imgDataSet(dataset, datasetSize);
				imgDataSet.SetDimensions(datasetDim.x, datasetDim.y, datasetDim.z);
				imgDataSet.SetVoxelSize(voxSize.x, voxSize.y, voxSize.z);

				for (int idx = startIdx; (idx < liveSubCubes) && (idx < startIdx + range); idx++)
				{
					Parallel::CubeProcessor cubeProcessor(&imgDataSet, isoValue, grid3dDim, subDistance, subDim, subVoxSize);

					auto subCubeWorldIdx = subCubes[idx + (memType * currentIter * subCubePerIter)];
					auto subCubesPerCube = (subDim.x - 1) * (subDim.y - 1) * (subDim.z - 1);

					auto cubeIdx = subCubeWorldIdx / subCubesPerCube;
					auto subCubeLocalIdx = subCubeWorldIdx % subCubesPerCube;

					int3 cubeCoord;
					Math::PointHelper::CalculateCubeCoordinates(&cubeCoord, cubeIdx, grid3dDim);

					int3 subCubeLocalCoord;
					Math::PointHelper::CalculateCubeCoordinates(&subCubeLocalCoord, subCubeLocalIdx,
					{ subDim.x - 1, subDim.y - 1, subDim.z - 1 });

					cubeProcessor.GeneratePoint(subCubeLocalCoord.x, subCubeLocalCoord.y, subCubeLocalCoord.z,
						cubeCoord.x, cubeCoord.y, cubeCoord.z, idx, points);
				}
			}

			class GeneratePointsKernel : public IKernel < unsigned __int64, Geometry::Vertex >
			{
				std::shared_ptr<thrust::device_vector<Geometry::Vertex>> m_output;
				std::shared_ptr<thrust::device_vector<float>> m_dataset;

				size_t m_datasetSize;
				int3 m_datasetDim;
				float m_isoValue;
				int3 m_gridDim;
				float3 m_voxSize;
				float m_subDistance;
				int3 m_subDim;
				float3 m_subVoxSize;

			public:

				GeneratePointsKernel(std::shared_ptr<thrust::device_vector<float>> dataset, size_t datasetSize, int3 datasetDim,
					float isoValue, int3 grid3Dim, float3 voxSize, float subDistance, int3 subDim, float3 subVoxSize)
					: m_dataset(dataset), m_datasetSize(datasetSize), m_datasetDim(datasetDim), m_isoValue(isoValue),
					m_gridDim(grid3Dim), m_voxSize(voxSize), m_subDistance(subDistance), m_subDim(subDim), m_subVoxSize(subVoxSize)
				{

				}

				size_t Run(std::shared_ptr<thrust::device_vector<unsigned long long>> const& input,
					KernelDimension const& kernelDim, int currentIter,
					size_t subCubePerIter, size_t liveSubCubes, size_t totalSubCubes, int range, short memType) override
				{
					m_output = std::make_shared<thrust::device_vector<Geometry::Vertex>>(liveSubCubes);

					generate_points_kernel << <kernelDim.grid, kernelDim.block >> >(thrust::raw_pointer_cast(m_dataset->data()), m_datasetSize, m_datasetDim,
						thrust::raw_pointer_cast(input->data()), m_isoValue, m_gridDim, m_voxSize, m_subDistance, m_subDim, m_subVoxSize,
						currentIter, subCubePerIter, liveSubCubes, totalSubCubes, range, memType,
						thrust::raw_pointer_cast(m_output->data()));

					return m_output->size();
				}

				std::shared_ptr<thrust::device_vector<Geometry::Vertex>> GetOutput() const override
				{
					return m_output;
				}
			};
		}
	}
}