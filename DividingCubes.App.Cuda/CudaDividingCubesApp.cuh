#pragma once

#include "Core/App.hpp"
#include "CudaDividingCubes.cuh"
#include "BinaryImageReader.hpp"
#include "Renderer/PointCloudRenderer.hpp"

namespace Impacts
{
	namespace Apps
	{
		class CudaDividingCubesApp : public Core::App < CudaDividingCubesApp >
		{
			std::shared_ptr<Cuda::CudaDividingCubes> m_dividingCubes;
			std::shared_ptr<Data::BinaryImageReader> m_binaryImageReader;
			std::shared_ptr<Data::ImageDataSet> m_imageDataset;
			std::shared_ptr<Geometry::Points> m_points;
			std::shared_ptr<Renderer::PointCloudRenderer> m_renderer;

			std::string m_datasetFile;
			glm::tvec3<int> m_dimensions;
			glm::vec3 m_cubeSize;
			float m_isoValue;
			float m_subDistance;

		public:
			CudaDividingCubesApp(int windowWidth, int windowHight, std::string datasetFile, glm::tvec3<int> dimensions, glm::vec3 cubeSize, float isoValue, float subDistance)
				: App(windowWidth, windowHight, "Cuda Dividing Cubes | OpenGL"), m_datasetFile(datasetFile), m_dimensions(dimensions), m_cubeSize(cubeSize),
				m_isoValue(isoValue), m_subDistance(subDistance)
			{
				m_binaryImageReader = std::make_shared<Data::BinaryImageReader>();
				m_binaryImageReader->SetDimensions(m_dimensions.x, m_dimensions.y, m_dimensions.z);
				m_binaryImageReader->SetVoxelSize(m_cubeSize.x, m_cubeSize.y, m_cubeSize.z);
				int id = m_binaryImageReader->Load(m_datasetFile);
				m_imageDataset = m_binaryImageReader->GetOutput(id);

				m_dividingCubes = std::make_shared<Cuda::CudaDividingCubes>(m_imageDataset, m_isoValue, m_subDistance);
				m_dividingCubes->Start();

				m_points = m_dividingCubes->GetPoints();

				m_renderer = std::make_shared<Renderer::PointCloudRenderer>(m_window->GetRatio(), m_dimensions.z);
				m_renderer->LoadPointCloud(m_points);
			}

			void Draw()
			{
				static float time = 0.0f;
				time += 0.5f;

				m_renderer->Draw(time);
			}

			void KeyDown(int key, int action)
			{
				m_renderer->KeyDown(key, action);
			}
		};
	}
}