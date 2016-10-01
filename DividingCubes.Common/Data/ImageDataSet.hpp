#pragma once

#include "../Configuration.hpp"
#include "../glm.hpp"
#include "../Macros.hpp"

#if IMPACTS_COMPILER & IMPACTS_COMPILER_CUDA
#	include <thrust/host_vector.h>
#else
#	include <vector>
#endif


namespace Impacts
{
	namespace Data
	{
		// TODO: This is version 1.0
		class ImageDataSet
		{
#		if IMPACTS_COMPILER & IMPACTS_COMPILER_CUDA
			using DataType = thrust::host_vector<float>;
#		else
			using DataType = std::vector<float>;
#		endif

			DataType m_data;
			glm::tvec3<glm::i32> m_dim;
			glm::vec3 m_voxSize;
			int m_rawSize;

			// No copying is allowed due to heavy data, instead use smart pointers or raw pointers
			ImageDataSet(ImageDataSet const &) = delete;
			ImageDataSet& operator=(ImageDataSet const &) = delete;

		public:
			ImageDataSet()
			{
				m_dim.x = m_dim.y = m_dim.z = 0;
				m_voxSize.x = m_voxSize.y = m_voxSize.z = 1.0f;
				m_rawSize = 0;
			}

			~ImageDataSet()
			{
				m_data.clear();
				DataType().swap(m_data);
			}

			bool SetPixel(unsigned int x, unsigned int y, unsigned int z, float value)
			{
				auto idx = GetOffset(x, y, z);
				if (idx == -1)
					return false;

				if (m_data.size() < static_cast<unsigned int>(m_rawSize))
				{
					Allocate();
				}

				m_data[idx] = value;

				return true;
			}

			float GetPixel(unsigned int x, unsigned int y, unsigned int z) const
			{
				auto idx = GetOffset(x, y, z);
				if (idx == -1)
					return 0.0f;

				return m_data[idx];
			}

			void SetSlice(unsigned int z, char const * buffer)
			{
				ARGUMENT(static_cast<int>(z) < m_dim.z);

				for (auto y = 0; y < m_dim.y; y++)
				{
					for (auto x = 0; x < m_dim.x; x++)
					{
						SetPixel(x, y, z, buffer[y * m_dim.x + x]);
					}
				}
			}

			void SetDimensions(int x, int y, int z)
			{
				m_dim.x = x;
				m_dim.y = y;
				m_dim.z = z;

				m_rawSize = m_dim.x * m_dim.y * m_dim.z;

				// Re-size the data set
				Allocate();
			}

			glm::tvec3<glm::i32> GetDimensions() const
			{
				return m_dim;
			}

			void SetVoxelSize(float x, float y, float z)
			{
				m_voxSize.x = x;
				m_voxSize.y = y;
				m_voxSize.z = z;
			}

			glm::vec3 GetVoxelSize() const
			{
				return m_voxSize;
			}

			const float * GetRawDataConst() const
			{
				return m_data.data();
			}

			float * GetRawData()
			{
				return m_data.data();
			}

			DataType & GetVector()
			{
				return m_data;
			}

			int GetOffset(unsigned int x, unsigned int y, unsigned int z) const
			{
				if ((m_dim.x * m_dim.y * m_dim.z) == 0)
					return -1;

				return (z * m_dim.x * m_dim.y) + (y * m_dim.x) + x;
			}

			// Allocate data space in memory from the dimensions
			void Allocate()
			{
				Allocate(m_dim.x, m_dim.y, m_dim.z);
			}

			// Allocate data space in memory
			void Allocate(int w, int h, int d)
			{
				size_t size = w * h * d;
				if (size <= 0) return;

				m_data.resize(size);
			}
		};
	}
}
