#include "ImageDataSet.hpp"

/*using namespace Impacts::Data;

int ImageDataSet::GetOffset(unsigned int x, unsigned int y, unsigned int z) const
{
	if ((m_dim.x * m_dim.y * m_dim.z) == 0)
		return -1;

	return (z * m_dim.x * m_dim.y) + (y * m_dim.x) + x;
}

void ImageDataSet::Allocate()
{
	Allocate(m_dim.x, m_dim.y, m_dim.z);
}

void ImageDataSet::Allocate(int w, int h, int d)
{
	size_t size = w * h * d;
	if (size <= 0) return;

	m_data.resize(size);
}

bool ImageDataSet::SetPixel(unsigned int x, unsigned int y, unsigned int z, float value)
{
	auto idx = GetOffset(x, y, z);
	if (idx == -1)
		return false;

	if (m_data.size() < static_cast<unsigned int>(m_rawSize))
	{
		Allocate();
	}

	m_data[idx] = value;

	return true;
}

float ImageDataSet::GetPixel(unsigned int x, unsigned int y, unsigned int z) const
{
	auto idx = GetOffset(x, y, z);
	if (idx == -1)
		return 0.0f;

	return m_data[idx];
}

void ImageDataSet::SetSlice(unsigned int z, char const * buffer)
{
	ARGUMENT(static_cast<int>(z) < m_dim.z);

	for (auto y = 0; y < m_dim.y; y++)
	{
		for (auto x = 0; x < m_dim.x; x++)
		{
			SetPixel(x, y, z, buffer[y * m_dim.x + x]);
		}
	}
}

void ImageDataSet::SetDimensions(int x, int y, int z)
{
	m_dim.x = x;
	m_dim.y = y;
	m_dim.z = z;

	m_rawSize = m_dim.x * m_dim.y * m_dim.z;

	// Re-size the data set
	Allocate();
}

glm::tvec3<glm::i32> ImageDataSet::GetDimensions() const
{
	return m_dim;
}

void ImageDataSet::SetVoxelSize(float x, float y, float z)
{
	m_voxSize.x = x;
	m_voxSize.y = y;
	m_voxSize.z = z;
}

glm::vec3 ImageDataSet::GetVoxelSize() const
{
	return m_voxSize;
}

const float * ImageDataSet::GetRawDataConst() const
{
	return m_data.data();
}

float * ImageDataSet::GetRawData()
{
	return m_data.data();
}

ImageDataSet::DataType & ImageDataSet::GetVector()
{
	return m_data;
}
*/