#pragma once

#include "../pch.h"


namespace Impacts
{
	namespace Cuda
	{
		namespace Data
		{
			class ImageDataSet
			{
				float * m_data;
				int m_size;

				int3 m_dim;
				float3 m_voxSize;

			public:
				__device__ ImageDataSet(float * data, int size)
					: m_data(data), m_size(size)
				{
				}

				__device__ int3 const & GetDimensions() const
				{
					return m_dim;
				}

				__device__ void SetDimensions(int x, int y, int z)
				{
					m_dim.x = x;
					m_dim.y = y;
					m_dim.z = z;
				}

				__device__ float3 const & GetVoxelSize() const
				{
					return m_voxSize;
				}

				__device__ void SetVoxelSize(float x, float y, float z)
				{
					m_voxSize.x = x;
					m_voxSize.y = y;
					m_voxSize.z = z;
				}

				__device__ float GetPixel(int x, int y, int z) const
				{
					int idx = z * m_dim.y * m_dim.x + y * m_dim.x + x;
					return m_data[idx];
					//return tex1Dfetch(txt_Dataset, idx);
				}

				__device__ void GetPixel(int x, int y, int z, float * pixel) const
				{
					int idx = z * m_dim.y * m_dim.x + y * m_dim.x + x;
					*pixel = m_data[idx];
				}
			};
		}
	}
}