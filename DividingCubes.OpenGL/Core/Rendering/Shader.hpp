#pragma once

#include "..\..\pch.h"
#include "Macros.hpp"
#include "..\..\Helpers\FileSystem.hpp"

#include <vector>
#include <memory>

namespace Impacts
{
	namespace Core
	{
		namespace Rendering
		{
			struct ShaderSource
			{
			private:
				std::shared_ptr<FileIOSystem::FileData> m_vertex;
				std::shared_ptr<FileIOSystem::FileData> m_fragment;

			public:
				ShaderSource() { }
				ShaderSource(std::shared_ptr<FileIOSystem::FileData> vertex, std::shared_ptr<FileIOSystem::FileData> fragment)
					: m_vertex(vertex), m_fragment(fragment)
				{ }

				std::shared_ptr<FileIOSystem::FileData> const & GetVertexShader() const
				{
					return m_vertex;
				}

				std::shared_ptr<FileIOSystem::FileData> const & GetFragmentShader() const
				{
					return m_fragment;
				}
			};

			class Shader
			{
				GLuint m_sId;
				std::vector<std::shared_ptr<ShaderSource>> m_sources;

			public:
				Shader(const char* vertexShader, const char* fragmentShader)
				{
					Load(vertexShader, fragmentShader);
				}

				Shader(std::string const & vertShaderPath, std::string const & fragShaderPath)
				{
					ARGUMENT(vertShaderPath.size() > 0);
					ARGUMENT(fragShaderPath.size() > 0);

					FileIOSystem::FileSystem fs;
					auto vertFileData = fs.ReadFile(vertShaderPath);
					auto fragFileData = fs.ReadFile(fragShaderPath);

					auto shaderSource = std::make_shared<ShaderSource>(vertFileData, fragFileData);

					Load(vertFileData->GetData(), fragFileData->GetData(), vertFileData->GetSize(), fragFileData->GetSize());

					m_sources.push_back(std::move(shaderSource));
				}

				~Shader()
				{
					Unbind();
				}

				GLuint GetId() const
				{
					return m_sId;
				}

				void Bind()
				{
					glUseProgram(m_sId);
				}

				void Unbind()
				{
					glUseProgram(0);
				}

				void CompilerCheck(GLuint id)
				{
					GLint comp;
					glGetShaderiv(id, GL_COMPILE_STATUS, &comp);

					if (comp == GL_FALSE)
					{
						ThrowError(id, "Shader compilation FAILED");
					}
				}

				void LinkCheck(GLuint id)
				{
					GLint linkStatus, validateStatus;
					glGetProgramiv(id, GL_LINK_STATUS, &linkStatus);

					if (linkStatus == GL_FALSE)
					{
						ThrowError(id, "Shader linking FAILED");
					}

					glValidateProgram(id);
					glGetProgramiv(id, GL_VALIDATE_STATUS, &validateStatus);

					if (validateStatus == GL_FALSE)
					{
						ThrowError(id, "Shader validation FAILED");
					}
				}

			private:
				void ThrowError(GLuint id, std::string msg)
				{
					GLchar messages[256];
					glGetShaderInfoLog(id, sizeof(messages), nullptr, &messages[0]);
					throw std::runtime_error(msg + "=> " + std::string(messages));
				}

				void Load(const char * vertexShader, const char * fragmentShader, GLint vSize = 0, GLint fSize = 0)
				{
					// Create shader program
					m_sId = glCreateProgram();
					auto vId = glCreateShader(GL_VERTEX_SHADER);
					auto fId = glCreateShader(GL_FRAGMENT_SHADER);

					// Loader shader source code
					glShaderSource(vId, 1, &vertexShader, (vSize > 0 ? &vSize : nullptr));
					glShaderSource(fId, 1, &fragmentShader, (fSize > 0 ? &fSize : nullptr));

					// Compile
					glCompileShader(vId);
					glCompileShader(fId);

					// Check for compile errors
					CompilerCheck(vId);
					CompilerCheck(fId);

					// Attach shaders to program
					glAttachShader(m_sId, vId);
					glAttachShader(m_sId, fId);

					// Link program
					glLinkProgram(m_sId);

					// Check for linking errors
					LinkCheck(m_sId);

					//Use program
					glUseProgram(m_sId);
				}
			};
		}
	}
}
