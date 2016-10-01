#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <sstream>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")

#ifndef GLSL
#define GLSL(version, A) "#version " #version "\n" #A
#endif

inline void GLEW_VERIFY(GLenum glewRes)
{
	if (glewRes != GLEW_OK)
	{
		std::ostringstream ss;
		ss << "glew expression error" << std::endl << glewGetErrorString(glewRes) << std::endl;
		throw std::runtime_error(ss.str());
	}
}
