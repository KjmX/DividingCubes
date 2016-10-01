#include "pch.h"
/*#include "PointsLoader.hpp"

#include <iostream>

using namespace Impacts::Helpers;
using namespace Impacts::FileIOSystem;
using namespace Impacts::Geometry;
using namespace std;

void PointsLoader::Save(shared_ptr<Points> const & points, string const & path)
{
	for(auto const & v : *points)
	{
		string pos = to_string_with_precision<float>(v.position.x) + "," + to_string_with_precision<float>(v.position.y) + "," + to_string_with_precision<float>(v.position.z);
		string nor = to_string_with_precision<float>(v.normal.x) + "," + to_string_with_precision<float>(v.normal.y) + "," + to_string_with_precision<float>(v.normal.z);
		string col = to_string_with_precision<float>(v.color.r) + "," + to_string_with_precision<float>(v.color.g) + "," + to_string_with_precision<float>(v.color.b)
			+ "," + to_string_with_precision<float>(v.color.a);

		string vertex = pos + " " + nor + " " + col + "\n";

		m_fileSys.SaveFile(path, vertex.c_str(), vertex.size(), true);
	}

	char eof = '\0';
	m_fileSys.SaveFile(path, &eof, 1, true, true);
}

shared_ptr<Points> PointsLoader::Load(string const & path)
{
	auto data = m_fileSys.ReadFile(path, true);

	shared_ptr<Points> points = make_shared<Points>();

	char* next_token = nullptr;
	
	auto vertex = strtok_s(data->GetRawData(), "\n", &next_token);

	while (vertex != nullptr)
	{
		char* next_tokenI = nullptr;

		auto info = strtok_s(vertex, " ", &next_tokenI);
		auto pos = GetVec3(info);

		info = strtok_s(nullptr, " ", &next_tokenI);
		auto nor = GetVec3(info);

		//info = strtok_s(nullptr, " ", &next_tokenI);
		//auto col = GetVec4(info);

		points->push_back({ pos, nor, glm::vec4(0.4f, 0.4f, 0.4f, 1.0f) });

		vertex = strtok_s(nullptr, "\n", &next_token);
	}

	return move(points);
}

glm::vec3 PointsLoader::GetVec3(char * data)
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

glm::vec4 PointsLoader::GetVec4(char * data)
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
*/