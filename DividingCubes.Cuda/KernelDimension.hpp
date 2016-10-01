#pragma once

#include "pch.h"

namespace Impacts
{
	namespace Cuda
	{
		struct KernelDimension
		{
			dim3 grid;
			dim3 block;
		};
	}
}