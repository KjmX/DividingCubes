#pragma once

#include "AppWindow.hpp"
#include <memory>

#ifdef _DEBUG
#include <iostream>
#endif


namespace Impacts
{
	namespace Core
	{
		template <typename T>
		class App
		{
		protected:
			std::shared_ptr<AppWindow> m_window;

		public:
			App(int width, int height, std::string const & title)
			{
				VERIFY(glfwInit());

				m_window = std::make_shared<AppWindow>();

				m_window->Create(this, width, height, title);

				glewExperimental = true;

				// GLEW_VERIFY is just an inline function which checks the result of a glew operation
				GLEW_VERIFY(glewInit());

				ASSERT(GLEW_VERSION_2_1);

#ifdef _DEBUG
				if (GLEW_ARB_vertex_array_object)
					std::cout << "genVertexArrays supported" << std::endl;

				if (GLEW_APPLE_vertex_array_object)
					std::cout << "genVertexArrayAPPLE supported" << std::endl;
#endif

				//glEnable(GL_ALPHA_TEST);
				//glAlphaFunc(GL_GREATER, 0.5f);

				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_LESS);

				glDepthMask(GL_TRUE);

				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

				//glEnable(GL_POINT_SMOOTH);

				//glEnable(GL_PROGRAM_POINT_SIZE);
			}

			~App()
			{
				glfwTerminate();
			}

			void Run()
			{
				while (!m_window->ShouldClose())
				{
					m_window->SetViewport();

					glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

					static_cast<T *>(this)->Draw();

					m_window->SwapBuffers();
					glfwPollEvents();	// Trait and pass system messages.
				}
			}

			void Draw() { }
			void KeyDown(int, int) { }
			void MouseMove(double, double) { }
			void MouseDown(int, int) { }

			void OnKeyDown(int key, int action)
			{
				static_cast<T *>(this)->KeyDown(key, action);
			}

			void OnMouseMove(double x, double y)
			{
				static_cast<T *>(this)->MouseMove(x, y);
			}

			void OnMouseDown(int button, int action)
			{
				static_cast<T *>(this)->MouseDown(button, action);
			}
		};
	}
}
