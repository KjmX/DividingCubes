#include "pch.h"
/*#include "PointCloudRenderer.hpp"

using namespace Impacts::Renderer;
using namespace Impacts::Core::Rendering;
using namespace Impacts::Geometry;
using namespace std;

string Impacts::Renderer::ShaderSource::vert = GLSL(120,

	attribute vec4 position;
	attribute vec4 color;

	varying vec4 dstColor;

	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 projection;

	void main(void)
	{
		dstColor = color;
		gl_Position = projection * view * model * position;
	}

);

string Impacts::Renderer::ShaderSource::frag = GLSL(120,

	varying vec4 dstColor;

	void main(void)
	{
		gl_FragColor = dstColor;
	}

);

PointCloudRenderer::PointCloudRenderer(float windowRatio, int depth) : m_windowRatio(windowRatio), m_depth(depth)
{
	m_eyeZ = m_depth * 4;
}

PointCloudRenderer::~PointCloudRenderer()
{
}

void PointCloudRenderer::LoadPointCloud(shared_ptr<Points> const & points)
{
	ARGUMENT(points);
	VERIFY(points->size() > 0);

	m_points = points;

	Init();
}

void PointCloudRenderer::Draw(float time)
{
	m_shader->Bind();

	glBindVertexArray(m_arrayId);

	auto eyeDirection = glm::vec3(0.0f, 0.0f, m_eyeZ);

	glm::mat4 view = glm::lookAt(eyeDirection, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	glm::mat4 proj = glm::perspective<float>(glm::radians(45.0f), m_windowRatio, 0.1f, -10.0f);

	glUniformMatrix4fv(m_viewId, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(m_projectionId, 1, GL_FALSE, glm::value_ptr(proj));

	auto rotateZ = glm::rotate<float>(glm::mat4(), glm::radians(90.0f), glm::vec3(0, 0, 1));
	auto rotate = glm::rotate<float>(rotateZ, time / 100.f, glm::vec3(0, 1, 0));
	glm::mat4 model = rotate;
	glUniformMatrix4fv(m_modelId, 1, GL_FALSE, glm::value_ptr(model));

	glm::mat4 modelview = glm::inverse(view);
	glUniformMatrix4fv(m_modelviewId, 1, GL_FALSE, glm::value_ptr(modelview));

	glm::mat3 normalMatrix = glm::transpose<float>(glm::inverse(glm::mat3(view * model)));
	glUniformMatrix3fv(m_normalMatrixId, 1, GL_FALSE, glm::value_ptr(normalMatrix));

	m_lightShader->Draw(eyeDirection, time, m_depth);

	glDrawArrays(GL_POINTS, 0, m_points->size());

	glBindVertexArray(0);

	m_shader->Unbind();
}

void PointCloudRenderer::KeyDown(int key, int action)
{
	/*
	*	Oi oi ! If u want to play with the light position just use:
	*		z: for making the light going forward	(-Z axis)
	*		s: for making the light going backward	(+Z axis)
	*		d: for making the light going right		(+X axis)
	*		q: for making the light going left		(-X axis)
	*		e: for making the light going up		(+Y axis)
	*		a: for making the light going down		(-Y axis)
	*/

	/*
	*	There is also two keys, I and K, it's for making the object
	*	more or less shiny ! I => more, K => less
	*/

	/*
	*	Also u can go forward by pressing the UP arrow key, and backward
	*	by pressing the DOWN arrow key. (Ofc later we will create a camera class)
	*/

	if (key == GLFW_KEY_Z)
	{
		if(action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustLightPosition(glm::vec3(0.0f, 0.0f, -1.0f));
	}
	else if (key == GLFW_KEY_S)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustLightPosition(glm::vec3(0.0f, 0.0f, 1.0f));
	}

	if (key == GLFW_KEY_D)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustLightPosition(glm::vec3(1.0f, 0.0f, 0.0f));
	}
	else if (key == GLFW_KEY_Q)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustLightPosition(glm::vec3(-1.0f, 0.0f, 0.0f));
	}

	if (key == GLFW_KEY_E)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustLightPosition(glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else if (key == GLFW_KEY_A)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustLightPosition(glm::vec3(0.0f, -1.0, 0.0f));
	}

	// Material Shininess
	if (key == GLFW_KEY_I)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustMaterialShininess(1.0f);
	}
	else if (key == GLFW_KEY_K)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_lightShader->AdjustMaterialShininess(-1.0f);
	}


	// camera Z
	if(key == GLFW_KEY_UP)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_eyeZ -= 3.0f;
	}
	else if (key == GLFW_KEY_DOWN)
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
			m_eyeZ += 3.0f;
	}
}

void PointCloudRenderer::Init()
{
	InitShader();

	glGenVertexArrays(1, &m_arrayId);
	glBindVertexArray(m_arrayId);

	glGenBuffers(1, &m_bufferId);
	glBindBuffer(GL_ARRAY_BUFFER, m_bufferId);
	glBufferData(GL_ARRAY_BUFFER, m_points->size() * sizeof(Vertex), m_points->data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(m_positionId);
	glEnableVertexAttribArray(m_normalId);
	glEnableVertexAttribArray(m_colorId);

	glVertexAttribPointer(m_positionId, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(0));
	glVertexAttribPointer(m_normalId, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, normal)));
	glVertexAttribPointer(m_colorId, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, color)));

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PointCloudRenderer::InitShader()
{
	//m_shader = make_shared<Shader>(ShaderSource::vert.c_str(), ShaderSource::frag.c_str());
	m_shader = make_shared<Shader>(string("..\\Data\\Shaders\\light_vs.glsl"), string("..\\Data\\Shaders\\light_fs.glsl"));

	m_positionId = glGetAttribLocation(m_shader->GetId(), "position");
	m_normalId = glGetAttribLocation(m_shader->GetId(), "normal");
	m_colorId = glGetAttribLocation(m_shader->GetId(), "color");
	m_normalMatrixId = glGetUniformLocation(m_shader->GetId(), "normalMatrix");
	m_modelId = glGetUniformLocation(m_shader->GetId(), "model");
	m_viewId = glGetUniformLocation(m_shader->GetId(), "view");
	m_projectionId = glGetUniformLocation(m_shader->GetId(), "projection");
	m_modelviewId = glGetUniformLocation(m_shader->GetId(), "modelview");

	m_lightShader = make_shared<LightShader>(m_shader);
	m_lightShader->Initialize();

	m_shader->Unbind();

	//glPointSize(2.0f);
}
*/