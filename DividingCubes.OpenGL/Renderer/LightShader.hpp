#pragma once

#include "..\pch.h"
#include "..\Core\Rendering\Shader.hpp"
#include "Macros.hpp"

#include "glm\glm.hpp"
#include "glm\gtc\type_ptr.hpp"

#include <memory>

namespace Impacts
{
	namespace Renderer
	{
		struct LightProperties
		{
			glm::vec4 position;
			glm::vec3 color;
			glm::vec4 ambient;
			glm::vec4 diffuse;
			glm::vec4 specular;
			float constantAttenuation;
			float linearAttenuation;
			float quadraticAttenuation;
		};

		struct MaterialProperties
		{
			glm::vec4 emission;
			glm::vec4 ambient;
			glm::vec4 diffuse;
			glm::vec4 specular;
			float shininess;
		};

		class LightShader
		{
			struct LightIDs
			{
				GLuint positionId;
				GLuint colorId;
				GLuint ambientId;
				GLuint diffuseId;
				GLuint specularId;
				GLuint constantAttenuationId;
				GLuint linearAttenuationId;
				GLuint quadraticAttenuationId;
			};

			struct MaterialIDs
			{
				GLuint emissionId;
				GLuint ambientId;
				GLuint diffuseId;
				GLuint specularId;
				GLuint shininessId;
			};

			std::shared_ptr<Core::Rendering::Shader> m_shader;
			LightIDs m_lightIds;
			MaterialIDs m_materialIds;
			GLuint m_eyeDirectionId;
			LightProperties m_lightProp;
			MaterialProperties m_materialProp;

		public:
			LightShader(std::shared_ptr<Core::Rendering::Shader> const & shader)
				: m_shader(shader)
			{
			}

			void Initialize()
			{
				VERIFY(m_shader != nullptr);

				m_lightIds.positionId = glGetUniformLocation(m_shader->GetId(), "light.position");
				m_lightIds.colorId = glGetUniformLocation(m_shader->GetId(), "light.color");
				m_lightIds.ambientId = glGetUniformLocation(m_shader->GetId(), "light.ambient");
				m_lightIds.diffuseId = glGetUniformLocation(m_shader->GetId(), "light.diffuse");
				m_lightIds.specularId = glGetUniformLocation(m_shader->GetId(), "light.specular");
				m_lightIds.constantAttenuationId = glGetUniformLocation(m_shader->GetId(), "light.constantAttenuation");
				m_lightIds.linearAttenuationId = glGetUniformLocation(m_shader->GetId(), "light.linearAttenuation");
				m_lightIds.quadraticAttenuationId = glGetUniformLocation(m_shader->GetId(), "light.quadraticAttenuation");

				m_materialIds.emissionId = glGetUniformLocation(m_shader->GetId(), "material.emission");
				m_materialIds.ambientId = glGetUniformLocation(m_shader->GetId(), "material.ambient");
				m_materialIds.diffuseId = glGetUniformLocation(m_shader->GetId(), "material.diffuse");
				m_materialIds.specularId = glGetUniformLocation(m_shader->GetId(), "material.specular");
				m_materialIds.shininessId = glGetUniformLocation(m_shader->GetId(), "material.shininess");

				m_eyeDirectionId = glGetUniformLocation(m_shader->GetId(), "eyeDirection");

				m_lightProp.position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
				m_lightProp.color = glm::vec3(1.0f, 1.0f, 1.0f);
				m_lightProp.ambient = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
				m_lightProp.diffuse = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				m_lightProp.specular = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				m_lightProp.constantAttenuation = 0.5f;
				m_lightProp.linearAttenuation = 0.1f;
				m_lightProp.quadraticAttenuation = 0.0f;

				m_materialProp.emission = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
				m_materialProp.ambient = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);
				m_materialProp.diffuse = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				m_materialProp.specular = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				m_materialProp.shininess = 20.0f;
			}

			void Draw(glm::vec3 const & eyeDirection, float time, float depth)
			{
				glUniform4fv(m_lightIds.positionId, 1, glm::value_ptr(m_lightProp.position));
				glUniform3fv(m_lightIds.colorId, 1, glm::value_ptr(m_lightProp.color));
				glUniform4fv(m_lightIds.ambientId, 1, glm::value_ptr(m_lightProp.ambient));
				glUniform4fv(m_lightIds.diffuseId, 1, glm::value_ptr(m_lightProp.diffuse));
				glUniform4fv(m_lightIds.specularId, 1, glm::value_ptr(m_lightProp.specular));
				glUniform1f(m_lightIds.constantAttenuationId, m_lightProp.constantAttenuation);
				glUniform1f(m_lightIds.linearAttenuationId, m_lightProp.linearAttenuation);
				glUniform1f(m_lightIds.quadraticAttenuationId, m_lightProp.quadraticAttenuation);

				glUniform4fv(m_materialIds.emissionId, 1, glm::value_ptr(m_materialProp.emission));
				glUniform4fv(m_materialIds.ambientId, 1, glm::value_ptr(m_materialProp.ambient));
				glUniform4fv(m_materialIds.diffuseId, 1, glm::value_ptr(m_materialProp.diffuse));
				glUniform4fv(m_materialIds.specularId, 1, glm::value_ptr(m_materialProp.specular));
				glUniform1f(m_materialIds.shininessId, m_materialProp.shininess);

				glUniform3fv(m_eyeDirectionId, 1, glm::value_ptr(eyeDirection));
			}

			void AdjustLightPosition(glm::vec3 acc)
			{
				m_lightProp.position += glm::vec4(acc, 1.0f);
			}
			
			void AdjustMaterialShininess(float shininess)
			{
				m_materialProp.shininess += shininess;
			}
		};
	}
}
