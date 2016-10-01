#pragma once

#include "Configuration.hpp"

#if IMPACTS_COMPILER && IMPACTS_COMPILER_CUDA
#	define GLM_FORCE_CUDA
#endif

#include <glm/glm.hpp>