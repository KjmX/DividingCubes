#pragma once

#include "Vertex.hpp"

//#	if IMPACTS_COMPILER & IMPACTS_COMPILER_CUDA
//#		include <thrust/host_vector.h>
//		using Points = thrust::host_vector<Vertex>;
//#	else
//#		include <vector>
//		using Points = std::vector<Vertex>;
//#	endif

#include <vector>

namespace Impacts
{
	namespace Geometry
	{
		using Points = std::vector<Vertex>;
	}
}