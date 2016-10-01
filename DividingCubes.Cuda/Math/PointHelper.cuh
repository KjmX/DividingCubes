#pragma once

#define GLM_FORCE_CUDA
#include "glm\glm.hpp"

#include "..\Data\ImageDataSet.cuh"

#define CubeCoordinates(x, y, z, index, dim)\
		int offset2D = index % (dim.x * dim.y);\
		z = index / (dim.x * dim.y);\
		y = offset2D / dim.x;\
		x = offset2D % dim.x;\

namespace Impacts
{
	namespace Cuda
	{
		namespace Math
		{
			class PointHelper
			{
			public:

				__device__ static void CalculateCubeCoordinates(int3 * coord, int index, int3 dim)
				{
					int offset2D = index % (dim.x * dim.y);
					coord->z = index / (dim.x * dim.y);
					coord->y = offset2D / dim.x;
					coord->x = offset2D % dim.x;
				}

				__device__ static void GetGradientAtPoint(float3 * gradient, int x, int y, int z, float sizeX, float sizeY, float sizeZ,
					Data::ImageDataSet const * imageData)
				{
					// Get the dimensions of the cell
					auto dims = imageData->GetDimensions();

					float forward;
					float backward;

					// x-axis
					if (dims.x == 1)
						gradient->x = 0.0f;
					else if (x == 0) // the first cell in the x-axis, means no backward neighbor...
					{
						forward = imageData->GetPixel(x + 1, y, z);
						backward = imageData->GetPixel(x, y, z);
						gradient->x = (backward - forward) / sizeX;
					}
					else if (x == (dims.x - 1))	// the last cell in the x-axis, no forward neighbor...
					{
						forward = imageData->GetPixel(x, y, z);
						backward = imageData->GetPixel(x - 1, y, z);
						gradient->x = (backward - forward) / sizeX;
					}
					else
					{
						forward = imageData->GetPixel(x + 1, y, z);
						backward = imageData->GetPixel(x - 1, y, z);
						gradient->x = (backward - forward) / (2 * sizeX);
					}

					// y-axis
					if (dims.y == 1)
						gradient->y = 0.0f;
					else if (y == 0) // the first cell in the y-axis
					{
						forward = imageData->GetPixel(x, y + 1, z);
						backward = imageData->GetPixel(x, y, z);
						gradient->y = (backward - forward) / sizeY;
					}
					else if (y == (dims.y - 1)) // the last cell in the y-axis
					{
						forward = imageData->GetPixel(x, y, z);
						backward = imageData->GetPixel(x, y - 1, z);
						gradient->y = (backward - forward) / sizeY;
					}
					else
					{
						forward = imageData->GetPixel(x, y + 1, z);
						backward = imageData->GetPixel(x, y - 1, z);
						gradient->y = (backward - forward) / (2 * sizeY);
					}

					// z-axis
					if (dims.z == 1)
						gradient->z = 0.0f;
					else if (z == 0) // the first cell in the z-axis
					{
						forward = imageData->GetPixel(x, y, z + 1);
						backward = imageData->GetPixel(x, y, z);
						gradient->z = (backward - forward) / sizeZ;
					}
					else if (z == (dims.z - 1)) // the last cell in the z-axis
					{
						forward = imageData->GetPixel(x, y, z);
						backward = imageData->GetPixel(x, y, z - 1);
						gradient->z = (backward - forward) / sizeZ;
					}
					else
					{
						forward = imageData->GetPixel(x, y, z + 1);
						backward = imageData->GetPixel(x, y, z - 1);
						gradient->z = (backward - forward) / (2 * sizeZ);
					}
				}


				__device__ static void HexahedronBasisFunctions(glm::vec3 p, float * bf0, float * bf1, float * bf2, float * bf3,
					float * bf4, float * bf5, float * bf6, float * bf7)
				{
					float r = p.x;
					float s = p.y;
					float t = p.z;

					*bf0 = (1.0f - r) * (1.0f - s) * (1.0f - t);
					*bf1 = r * (1.0f - s) * (1.0f - t);
					*bf2 = r * s * (1.0f - t);
					*bf3 = (1.0f - r) * s * (1.0f - t);
					*bf4 = (1.0f - r) * (1.0f - s) * t;
					*bf5 = r * (1.0f - s) * t;
					*bf6 = r * s * t;
					*bf7 = (1.0f - r) * s * t;
				}

			};
		}
	}
}