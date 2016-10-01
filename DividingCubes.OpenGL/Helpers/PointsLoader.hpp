#pragma once

#include "Geometry/Points.hpp"
#include "FileSystem.hpp"

#include <memory>
#include <sstream>
#include <iomanip>

namespace Impacts
{
	namespace Helpers
	{
		class PointsLoader
		{
			FileIOSystem::FileSystem m_fileSys;

		public:

			PointsLoader() {}

			void Save(std::shared_ptr<Geometry::Points> const & points, std::string const & path)
			{
				for (auto const & v : *points)
				{
					std::string pos = to_string_with_precision<float>(v.position.x) + "," + to_string_with_precision<float>(v.position.y) + "," + to_string_with_precision<float>(v.position.z);
					std::string nor = to_string_with_precision<float>(v.normal.x) + "," + to_string_with_precision<float>(v.normal.y) + "," + to_string_with_precision<float>(v.normal.z);

					std::string vertex = pos + " " + nor + "\n";

					m_fileSys.SaveFile(path, vertex.c_str(), vertex.size(), true);
				}

				char eof = '\0';
				m_fileSys.SaveFile(path, &eof, 1, true, true);
			}

			std::shared_ptr<Geometry::Points> Load(std::string const & path)
			{
				auto data = m_fileSys.ReadFile(path, true);

				std::shared_ptr<Geometry::Points> points = std::make_shared<Geometry::Points>();

				char* next_token = nullptr;

				auto vertex = strtok_s(data->GetRawData(), "\n", &next_token);

				while (vertex != nullptr)
				{
					char* next_tokenI = nullptr;

					auto info = strtok_s(vertex, " ", &next_tokenI);
					auto pos = GetVec3(info);

					info = strtok_s(nullptr, " ", &next_tokenI);
					auto nor = GetVec3(info);

					points->push_back({ pos, nor });

					vertex = strtok_s(nullptr, "\n", &next_token);
				}

				return std::move(points);
			}

		private:
			template <typename T>
			std::string to_string_with_precision(const T a_value, const int n = 9)
			{
				std::ostringstream out;
				out << std::setprecision(n) /*<< std::fixed*/ << a_value;
				auto s = out.str();

				return s;
			}

			glm::vec3 GetVec3(char * data)
			{
				char* next_token = nullptr;
				float f[3];

				auto v = strtok_s(data, ",", &next_token);

				for (int i = 0; (i < 3) && (v != nullptr); i++)
				{
					f[i] = stof(v);
					v = strtok_s(nullptr, ",", &next_token);
				}

				return glm::vec3(f[0], f[1], f[2]);
			}

			glm::vec4 GetVec4(char * data)
			{
				char* next_token = nullptr;
				float f[4];

				auto v = strtok_s(data, ",", &next_token);

				for (int i = 0; (i < 4) && (v != nullptr); i++)
				{
					f[i] = stof(v);
					v = strtok_s(nullptr, ",", &next_token);
				}

				return glm::vec4(f[0], f[1], f[2], f[3]);
			}
		};
	}
}
