#pragma once

#include "..\Data\ImageDataSet.cuh"
//#include "Geometry\Voxel.hpp"
#include "..\Math\PointHelper.cuh"

#include <climits>

#include "glm.hpp"

namespace Impacts
{
	namespace Cuda
	{
		namespace Parallel
		{
			class CubeProcessor
			{
				Data::ImageDataSet * m_imgData;
				float m_isoValue;
				int3 m_gridDim;
				int3 m_datasetDim;
				float3 m_voxSize;
				float m_subDistance;
				int3 m_subDim;
				float3 m_subVoxSize;

				__device__ CubeProcessor(CubeProcessor const &) = delete;
				__device__ CubeProcessor& operator=(CubeProcessor const &) = delete;

			public:
				__device__ CubeProcessor(Data::ImageDataSet * imgData, float isoValue, int3 grid3dDim, float subDistance, int3 subDim, float3 subVoxSize)
				{
					m_imgData = imgData;
					m_isoValue = isoValue;
					m_gridDim = grid3dDim;
					m_datasetDim = m_imgData->GetDimensions();
					m_voxSize = m_imgData->GetVoxelSize();
					m_subDistance = subDistance;
					m_subDim = subDim;
					m_subVoxSize = subVoxSize;
				}

				// TODO: remember to add the case where classifiedCubes size is smaller than gridDim (to copy in CPU)
				__device__ void ClassifyCube(int x, int y, int z, size_t classifiedCubesSize, unsigned int * classifiedCubes)
				{
					float s0, s1, s2, s3, s4, s5, s6, s7;

					s0 = m_imgData->GetPixel(x, y, z);
					s1 = m_imgData->GetPixel(x + 1, y, z);
					s2 = m_imgData->GetPixel(x + 1, y + 1, z);
					s3 = m_imgData->GetPixel(x, y + 1, z);
					s4 = m_imgData->GetPixel(x, y, z + 1);
					s5 = m_imgData->GetPixel(x + 1, y, z + 1);
					s6 = m_imgData->GetPixel(x + 1, y + 1, z + 1);
					s7 = m_imgData->GetPixel(x, y + 1, z + 1);

					bool inside = false;
					bool outside = false;

					inside = inside || s0 >= m_isoValue;
					outside = outside || s0 < m_isoValue;

					inside = inside || s1 >= m_isoValue;
					outside = outside || s1 < m_isoValue;

					inside = inside || s2 >= m_isoValue;
					outside = outside || s2 < m_isoValue;

					inside = inside || s3 >= m_isoValue;
					outside = outside || s3 < m_isoValue;

					inside = inside || s4 >= m_isoValue;
					outside = outside || s4 < m_isoValue;

					inside = inside || s5 >= m_isoValue;
					outside = outside || s5 < m_isoValue;

					inside = inside || s6 >= m_isoValue;
					outside = outside || s6 < m_isoValue;

					inside = inside || s7 >= m_isoValue;
					outside = outside || s7 < m_isoValue;

					int idx = z * m_gridDim.y * m_gridDim.x + y * m_gridDim.x + x;

					classifiedCubes[idx] = (inside && outside)
						? idx
						: classifiedCubesSize;
				}

				__device__ void ClassifySubCube(int sx, int sy, int sz, int cx, int cy, int cz,
					int idx, int subCubeLocalIdx, int cubeIdx, unsigned __int64 totalSubCubes, unsigned __int64 * classifiedSubCubes)
				{
					float cs0 = m_imgData->GetPixel(cx, cy, cz);
					float cs1 = m_imgData->GetPixel(cx + 1, cy, cz);
					float cs2 = m_imgData->GetPixel(cx + 1, cy + 1, cz);
					float cs3 = m_imgData->GetPixel(cx, cy + 1, cz);
					float cs4 = m_imgData->GetPixel(cx, cy, cz + 1);
					float cs5 = m_imgData->GetPixel(cx + 1, cy, cz + 1);
					float cs6 = m_imgData->GetPixel(cx + 1, cy + 1, cz + 1);
					float cs7 = m_imgData->GetPixel(cx, cy + 1, cz + 1);

					float s0 = InterpolateSamplePoint(sx, sy, sz, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s1 = InterpolateSamplePoint(sx + 1, sy, sz, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s2 = InterpolateSamplePoint(sx + 1, sy + 1, sz, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s3 = InterpolateSamplePoint(sx, sy + 1, sz, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s4 = InterpolateSamplePoint(sx, sy, sz + 1, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s5 = InterpolateSamplePoint(sx + 1, sy, sz + 1, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s6 = InterpolateSamplePoint(sx + 1, sy + 1, sz + 1, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
					float s7 = InterpolateSamplePoint(sx, sy + 1, sz + 1, cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);

					bool inside = false;
					bool outside = false;

					inside = inside || s0 >= m_isoValue;	inside = inside || s1 >= m_isoValue;
					outside = outside || s0 < m_isoValue;	outside = outside || s1 < m_isoValue;

					inside = inside || s2 >= m_isoValue;	inside = inside || s3 >= m_isoValue;
					outside = outside || s2 < m_isoValue;	outside = outside || s3 < m_isoValue;

					inside = inside || s4 >= m_isoValue;	inside = inside || s5 >= m_isoValue;
					outside = outside || s4 < m_isoValue;	outside = outside || s5 < m_isoValue;

					inside = inside || s6 >= m_isoValue;	inside = inside || s7 >= m_isoValue;
					outside = outside || s6 < m_isoValue;	outside = outside || s7 < m_isoValue;

					classifiedSubCubes[idx] = (inside && outside)
						? cubeIdx * ((m_subDim.x - 1) * (m_subDim.y - 1) * (m_subDim.z - 1)) + subCubeLocalIdx
						: totalSubCubes;
				}

				__device__ void GeneratePoint(int sx, int sy, int sz, int cx, int cy, int cz, int idx, Geometry::Vertex * points)
				{
					glm::vec3 n0 = InterpolateNormal(sx, sy, sz, cx, cy, cz);
					glm::vec3 n1 = InterpolateNormal(sx + 1, sy, sz, cx, cy, cz);
					glm::vec3 n2 = InterpolateNormal(sx + 1, sy + 1, sz, cx, cy, cz);
					glm::vec3 n3 = InterpolateNormal(sx, sy + 1, sz, cx, cy, cz);
					glm::vec3 n4 = InterpolateNormal(sx, sy, sz + 1, cx, cy, cz);
					glm::vec3 n5 = InterpolateNormal(sx + 1, sy, sz + 1, cx, cy, cz);
					glm::vec3 n6 = InterpolateNormal(sx + 1, sy + 1, sz + 1, cx, cy, cz);
					glm::vec3 n7 = InterpolateNormal(sx, sy + 1, sz + 1, cx, cy, cz);

					glm::vec3 n;
					n = n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7;
					n = glm::normalize(n);

					float3 cubeCenter{
						(cx - m_datasetDim.x / 2.0f) * m_voxSize.x,
						(cy - m_datasetDim.y / 2.0f) * m_voxSize.y,
						(cz - m_datasetDim.z / 2.0f) * m_voxSize.z,
					};

					float3 offset{
						(cubeCenter.x - m_voxSize.x / 2.0f) + (m_subVoxSize.x / 2.0f),
						(cubeCenter.y - m_voxSize.y / 2.0f) + (m_subVoxSize.y / 2.0f),
						(cubeCenter.z - m_voxSize.z / 2.0f) + (m_subVoxSize.z / 2.0f)
					};

					points[idx] = {
						glm::vec3(
						offset.x + (sx - m_subDim.x / 2.0f) * m_subVoxSize.x,
						offset.y + (sy - m_subDim.y / 2.0f) * m_subVoxSize.y,
						offset.z + (sz - m_subDim.z / 2.0f) * m_subVoxSize.z
						),
						n
					};
				}

			private:
				__device__ void CheckVertexIntersection(float s, bool & inside, bool & outside)
				{
					inside = inside || s >= m_isoValue;
					outside = outside || s < m_isoValue;
				}

				__device__ float InterpolateSamplePoint(int sx, int sy, int sz, float s0, float s1, float s2, float s3,
					float s4, float s5, float s6, float s7)
				{
					float r = sx * m_subVoxSize.x;
					float s = sy * m_subVoxSize.y;
					float t = sz * m_subVoxSize.z;

					float bf0 = (1.0f - r) * (1.0f - s) * (1.0f - t);
					float bf1 = r * (1.0f - s) * (1.0f - t);
					float bf2 = (1.0f - r) * s * (1.0f - t);
					float bf3 = r * s * (1.0f - t);
					float bf4 = (1.0f - r) * (1.0f - s) * t;
					float bf5 = r * (1.0f - s) * t;
					float bf6 = (1.0f - r) * s * t;
					float bf7 = r * s * t;

					return (s0 * bf0) +
						(s1 * bf1) +
						(s2 * bf2) +
						(s3 * bf3) +
						(s4 * bf4) +
						(s5 * bf5) +
						(s6 * bf6) +
						(s7 * bf7);
				}

				__device__ glm::vec3 InterpolateNormal(int sx, int sy, int sz, int cx, int cy, int cz)
				{
					float r = sx * m_subVoxSize.x;
					float s = sy * m_subVoxSize.y;
					float t = sz * m_subVoxSize.z;

					float bf0 = (1.0f - r) * (1.0f - s) * (1.0f - t);
					float bf1 = r * (1.0f - s) * (1.0f - t);
					float bf2 = (1.0f - r) * s * (1.0f - t);
					float bf3 = r * s * (1.0f - t);
					float bf4 = (1.0f - r) * (1.0f - s) * t;
					float bf5 = r * (1.0f - s) * t;
					float bf6 = (1.0f - r) * s * t;
					float bf7 = r * s * t;

					float3 cn0, cn1, cn2, cn3, cn4, cn5, cn6, cn7;

					Math::PointHelper::GetGradientAtPoint(&cn0, cx, cy, cz, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn1, cx + 1, cy, cz, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn2, cx + 1, cy + 1, cz, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn3, cx, cy + 1, cz, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn4, cx, cy, cz + 1, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn5, cx + 1, cy, cz + 1, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn6, cx + 1, cy + 1, cz + 1, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);
					Math::PointHelper::GetGradientAtPoint(&cn7, cx, cy + 1, cz + 1, m_voxSize.x, m_voxSize.y, m_voxSize.z, m_imgData);

					glm::vec3 n;

					n.x = (cn0.x * bf0) + (cn1.x * bf1) + (cn2.x * bf2) + (cn3.x * bf3)
						+ (cn4.x * bf4) + (cn5.x * bf5) + (cn6.x * bf6) + (cn7.x * bf7);
					n.y = (cn0.y * bf0) + (cn1.y * bf1) + (cn2.y * bf2) + (cn3.y * bf3)
						+ (cn4.y * bf4) + (cn5.y * bf5) + (cn6.y * bf6) + (cn7.y * bf7);
					n.z = (cn0.z * bf0) + (cn1.z * bf1) + (cn2.z * bf2) + (cn3.z * bf3)
						+ (cn4.z * bf4) + (cn5.z * bf5) + (cn6.z * bf6) + (cn7.z * bf7);

					return n;
				}
			};
		}
	}
}