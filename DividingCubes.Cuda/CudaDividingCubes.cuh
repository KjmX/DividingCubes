#pragma once

#include "pch.h"

// Data
#include <Data\ImageDataSet.hpp>
#include <Geometry\Points.hpp>

// Managers
#include "KernelManager.hpp"

// Kernels
#include "Kernels/ClassifyCubesKernel.cuh"
#include "Kernels/CompactCubesKernel.cuh"
#include "Kernels/ClassifySubCubesKernel.cuh"
#include "Kernels/GeneratePointsKernel.cuh"

#include <memory>
#include <stdexcept>

namespace Impacts
{
	namespace Cuda
	{
		class CudaDividingCubes
		{
			std::shared_ptr<Impacts::Data::ImageDataSet> m_imgData;
			float m_isoValue;
			int3 m_dim;
			float3 m_voxSize;
			float m_subDistance;
			int3 m_subDim;
			float3 m_subVoxSize;

			std::shared_ptr<Geometry::Points> m_points;

		public:
			CudaDividingCubes(std::shared_ptr<Impacts::Data::ImageDataSet> const & image, float isoValue, float subDistance = 0.1f)
			{
				ARGUMENT(image != nullptr);

				m_imgData = image;
				m_isoValue = isoValue;
				m_dim = { m_imgData->GetDimensions().x, m_imgData->GetDimensions().y, m_imgData->GetDimensions().z };
				m_voxSize = { m_imgData->GetVoxelSize().x, m_imgData->GetVoxelSize().y, m_imgData->GetVoxelSize().z };
				m_subDistance = subDistance;
				m_subDim = CalculateSubDim();
				m_subVoxSize = CalculateSubSize();
			}
			
			void Start()
			{
				try
				{
					RunKernels();
				}
				catch (thrust::system_error & e)
				{
					throw std::runtime_error(e.what());
				}
			}

			std::shared_ptr<Geometry::Points> GetPoints() const
			{
				return m_points;
			}

		private:
			void RunKernels()
			{
				auto d_dataset = std::make_shared<thrust::device_vector<float>>(m_imgData->GetVector());
				auto datasetMgr = std::make_shared<KernelDataManager<float>>();
				auto datasetId = datasetMgr->WriteToDevice(d_dataset);

				// ======================================= Step 1: Exclude empty voxels ======================================= //
				int3 grid3Dim{ (m_dim.x - 1), (m_dim.y - 1), (m_dim.z - 1) };
				KernelManager<float, unsigned> classifyCubesKernelMgr;
				classifyCubesKernelMgr.Setup(grid3Dim.x * grid3Dim.y * grid3Dim.z, datasetId, datasetMgr);

				auto classifyCubesKernel = std::make_shared<Kernels::ClassifyCubesKernel>(d_dataset->size(), m_dim, m_isoValue, grid3Dim);
				classifyCubesKernelMgr.Run(classifyCubesKernel);
				auto cubesOut = classifyCubesKernelMgr.GetOutput();
				auto cubesId = classifyCubesKernelMgr.GetOutputId();
				auto cubesSize = cubesOut->GetSize(cubesId);

				// ======================================= Step 2: Compact the resulted array ======================================= //
				KernelManager<unsigned, size_t> compactCubesKernelMgr;
				compactCubesKernelMgr.Setup(cubesSize, cubesId, cubesOut);

				auto compactCubesKernel = std::make_shared<Kernels::CompactCubesKernel<unsigned>>(cubesSize);
				compactCubesKernelMgr.Run(compactCubesKernel);
				auto cubeSizeOut = compactCubesKernelMgr.GetOutput();
				auto cubeSizeId = compactCubesKernelMgr.GetOutputId();

				thrust::host_vector<int> h_compactedSize = *cubeSizeOut->ReadToDevice(cubeSizeId, GPU);

				// ======================================= Step 3: Classify sub cubes ======================================= //
				KernelManager<unsigned int, unsigned __int64> classifySubCubesKernelMgr;
				classifySubCubesKernelMgr.Setup((m_subDim.x - 1) * (m_subDim.y - 1) * (m_subDim.z - 1) * h_compactedSize[0], cubesId, cubesOut, true);

				auto classifySubCubesKernel = std::make_shared<Kernels::ClassifySubCubesKernel>(d_dataset, d_dataset->size(), m_dim, m_isoValue,
					grid3Dim, m_subDistance, m_subDim, m_subVoxSize);
				classifySubCubesKernelMgr.Run(classifySubCubesKernel);
				auto subCubesOut = classifySubCubesKernelMgr.GetOutput();
				auto subCubesId = classifySubCubesKernelMgr.GetOutputId();
				auto subCubesSize = subCubesOut->GetSize(subCubesId);

				// ======================================= Step 4: Clear the VRAM ======================================= //
				cubeSizeOut->DeleteFromDevice(cubeSizeId);
				cubesOut->DeleteFromDevice(cubesId);

				// ======================================= Step 5: Generate points ======================================= //
				KernelManager<unsigned __int64, Geometry::Vertex> generatePointsKernelMgr;
				generatePointsKernelMgr.Setup(subCubesSize, subCubesId, subCubesOut, true);

				auto points = std::make_shared<Geometry::Points>(subCubesSize);
				auto generatePointsKernel = std::make_shared<Kernels::GeneratePointsKernel>(d_dataset, d_dataset->size(), m_dim, m_isoValue,
					grid3Dim, m_voxSize, m_subDistance, m_subDim, m_subVoxSize);
				generatePointsKernelMgr.Run(generatePointsKernel, points.get());
			}

			int3 CalculateSubDim()
			{
				return int3{ glm::ceil(m_voxSize.x / m_subDistance) + 1, glm::ceil(m_voxSize.y / m_subDistance) + 1, glm::ceil(m_voxSize.z / m_subDistance) + 1 };
			}

			float3 CalculateSubSize()
			{
				return{ m_voxSize.x / (m_subDim.x - 1), m_voxSize.y / (m_subDim.y - 1), m_voxSize.z / (m_subDim.z - 1) };
			}

		};
	}
}


