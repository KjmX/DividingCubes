#pragma once

#include "Vertex.hpp"

#include <memory>

namespace Impacts
{
	namespace Geometry
	{
		struct Voxel
		{
			glm::vec3 position;
			Vertex vertices[8];
			float scalars[8];

			static std::shared_ptr<Voxel> CreateVoxelAt(glm::vec3 const & center, glm::vec3 const & size)
			{
				auto voxel = std::make_shared<Voxel>();

				Voxel::CreateVoxel(center, size, voxel.get());

				return voxel;
			}

			static IMPACTS_FUNC_DECL void CreateVoxel(glm::vec3 const & center, glm::vec3 const & size, Voxel * voxel)
			{
				voxel->vertices[0] = { glm::vec3(-1 * size.x / 2.0f + center.x, -1 * size.y / 2.0f + center.y, -1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[1] = { glm::vec3( 1 * size.x / 2.0f + center.x, -1 * size.y / 2.0f + center.y, -1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[2] = { glm::vec3( 1 * size.x / 2.0f + center.x,  1 * size.y / 2.0f + center.y, -1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[3] = { glm::vec3(-1 * size.x / 2.0f + center.x,  1 * size.y / 2.0f + center.y, -1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[4] = { glm::vec3(-1 * size.x / 2.0f + center.x, -1 * size.y / 2.0f + center.y,  1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[5] = { glm::vec3( 1 * size.x / 2.0f + center.x, -1 * size.y / 2.0f + center.y,  1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[6] = { glm::vec3( 1 * size.x / 2.0f + center.x,  1 * size.y / 2.0f + center.y,  1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };
				voxel->vertices[7] = { glm::vec3(-1 * size.x / 2.0f + center.x,  1 * size.y / 2.0f + center.y,  1 * size.z / 2.0f + center.z), glm::vec3(0.0f, 0.0f, 0.0f) };

				voxel->position = center;
			}
		};
	}
}