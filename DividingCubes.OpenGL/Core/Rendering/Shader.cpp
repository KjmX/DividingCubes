#include "pch.h"
/*#include "Macros.h"
#include "Shader.hpp"

using namespace Impacts::Core::Rendering;
using namespace Impacts::FileIOSystem;
using namespace std;

Shader::Shader(const char * vertexShader, const char * fragmentShader)
{
	Load(vertexShader, fragmentShader);
}

Shader::Shader(string const & vertShaderPath, string const & fragShaderPath)
{
	ARGUMENT(vertShaderPath.size() > 0);
	ARGUMENT(fragShaderPath.size() > 0);

	FileSystem fs;
	auto vertFileData = fs.ReadFile(vertShaderPath);
	auto fragFileData = fs.ReadFile(fragShaderPath);

	auto shaderSource = make_shared<ShaderSource>(vertFileData, fragFileData);

	Load(vertFileData->GetData(), fragFileData->GetData(), vertFileData->GetSize(), fragFileData->GetSize());

	m_sources.push_back(move(shaderSource));
}

Shader::~Shader()
{
	Unbind();
}

GLuint Shader::GetId() const
{
	return m_sId;
}

void Shader::Bind()
{
	glUseProgram(m_sId);
}

void Shader::Unbind()
{
	glUseProgram(0);
}

void Shader::CompilerCheck(GLuint id)
{
	GLint comp;
	glGetShaderiv(id, GL_COMPILE_STATUS, &comp);

	if (comp == GL_FALSE)
	{
		ThrowError(id, "Shader compilation FAILED");
	}
}

void Shader::LinkCheck(GLuint id)
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

void Shader::ThrowError(GLuint id, std::string msg)
{
	GLchar messages[256];
	glGetShaderInfoLog(id, sizeof(messages), nullptr, &messages[0]);
	throw runtime_error(msg + "=> " + string(messages));
}

void Shader::Load(const char * vertexShader, const char * fragmentShader, GLint vSize, GLint fSize)
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
*/