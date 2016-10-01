#pragma once

#include "ImageReader.hpp"
#include "glm.hpp"
#include "Macros.hpp"

#include <memory>
#include <map>
#include <fstream>

namespace Impacts
{
	namespace Data
	{
		class BinaryImageReader : public ImageReader
		{
			// Key: simple unique identifier
			std::map<int, std::shared_ptr<ImageDataSet>> m_images;
			glm::tvec3<glm::i32> m_dim;
			glm::vec3 m_voxSize;
			int m_id;

		public:
			BinaryImageReader() : m_id(0)
			{
				m_voxSize.x = m_voxSize.y = m_voxSize.z = 1.0f;
			}

			~BinaryImageReader()
			{

			}

			int Load(std::string const & filename) override
			{
				auto image = Read8BitsImage(filename);
				m_images.insert(std::pair<int, std::shared_ptr<ImageDataSet>>(m_id, move(image)));

				return m_id++;
			}

			std::shared_ptr<ImageDataSet> GetOutput(int id) const override
			{
				ARGUMENT(id < m_id);

				return m_images.at(id);
			}

			void SetDimensions(int width, int height, int depth)
			{
				m_dim.x = width;
				m_dim.y = height;
				m_dim.z = depth;
			}

			void SetVoxelSize(float x, float y, float z)
			{
				m_voxSize.x = x;
				m_voxSize.y = y;
				m_voxSize.z = z;
			}

		private:
			std::shared_ptr<ImageDataSet> Read8BitsImage(std::string const & filename)
			{
				std::ifstream in(filename, std::ifstream::binary);

				VERIFY(m_dim.x > 0 && m_dim.y > 0 && m_dim.z > 0);
				VERIFY(m_voxSize.x > 0.0f && m_voxSize.y > 0.0f && m_voxSize.z > 0.0f);

				int sliceSize = m_dim.x * m_dim.y;
				char * buffer = new char[sliceSize];

				auto image = std::make_shared<ImageDataSet>();

				image->SetDimensions(m_dim.x, m_dim.y, m_dim.z);
				image->SetVoxelSize(m_voxSize.x, m_voxSize.y, m_voxSize.z);

				for (int z = 0; z < m_dim.z; z++)
				{
					in.read(buffer, sliceSize);

					if (!in)
					{
						in.close();
						delete[] buffer;
						throw std::runtime_error("Cannot read from file");
					}

					image->SetSlice(z, buffer);
				}

				return std::move(image);
			}

		};
	}
}
