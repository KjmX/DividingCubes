#pragma once

#include "../pch.h"
#include "Macros.hpp"
#include <string>

namespace Impacts
{
	namespace Core
	{
		struct IInput
		{
			static void * app;

			template <typename APP>
			static void OnKeyDown(GLFWwindow * window, int key, int, int action, int mods)
			{
				static_cast<APP *>(app)->OnKeyDown(key, action);
			}

			template <typename APP>
			static void OnMouseMove(GLFWwindow * window, double x, double y)
			{
				static_cast<APP *>(app)->OnMouseMove(x, y);
			}

			template <typename APP>
			static void OnMouseDown(GLFWwindow * window, int button, int action, int mods)
			{
				static_cast<APP *>(app)->OnMouseDown(button, action);
			}
		};

		void * IInput::app;


		class AppWindow
		{
			int m_width;
			int m_height;
			std::string m_title;
			GLFWwindow* m_window;
			IInput m_input;

		public:

			AppWindow()
			{}

			~AppWindow()
			{
				Destroy();
			}

			template <typename APP>
			void Create(APP * app, int width, int height, std::string const & title)
			{
				ARGUMENT(app);
				ARGUMENT(width > 0);
				ARGUMENT(height > 0);

				m_input.app = app;
				m_width = width;
				m_height = height;
				m_title = title;

				m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);

				VERIFY(m_window);

				glfwMakeContextCurrent(m_window);
				glfwSwapInterval(1); // number of screens updated before swap the framebuffer

				glfwSetKeyCallback(m_window, reinterpret_cast<GLFWkeyfun>(IInput::OnKeyDown<APP>));	// key down
				glfwSetCursorPosCallback(m_window, reinterpret_cast<GLFWcursorposfun>(IInput::OnMouseMove<APP>));	// mouse move
				glfwSetMouseButtonCallback(m_window, reinterpret_cast<GLFWmousebuttonfun>(IInput::OnMouseDown<APP>));	// mouse down
			}

			void SetViewport()
			{
				ASSERT(m_window);

				glfwGetFramebufferSize(m_window, &m_width, &m_height);
				glViewport(0, 0, m_width, m_height);
			}

			int ShouldClose()
			{
				ASSERT(m_window);

				return glfwWindowShouldClose(m_window);
			}

			void SwapBuffers()
			{
				ASSERT(m_window);

				glfwSwapBuffers(m_window);
			}

			void Destroy()
			{
				ASSERT(m_window);

				glfwDestroyWindow(m_window);
			}

			int GetWidth() const
			{
				return m_width;
			}

			int GetHeight() const
			{
				return m_height;
			}

			float GetRatio() const
			{
				return static_cast<float>(m_width) / m_height;
			}

			GLFWwindow * Get() const
			{
				return m_window;
			}
		};
	}
}