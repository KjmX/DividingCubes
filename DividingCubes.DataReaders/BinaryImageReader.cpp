#include "BinaryImageReader.hpp"

/*using namespace Impacts::Data;
using namespace std;

int BinaryImageReader::Load(std::string const & filename)
{
	auto image = Read8BitsImage(filename);
	m_images.insert(pair<int, shared_ptr<ImageDataSet>>(m_id, move(image)));

	return m_id++;
}

shared_ptr<ImageDataSet> BinaryImageReader::GetOutput(int id) const
{
	ARGUMENT(id < m_id);

	return m_images.at(id);
}

void BinaryImageReader::SetDimensions(int width, int height, int depth)
{
	m_dim.x = width;
	m_dim.y = height;
	m_dim.z = depth;
}

void BinaryImageReader::SetVoxelSize(float x, float y, float z)
{
	m_voxSize.x = x;
	m_voxSize.y = y;
	m_voxSize.z = z;
}

shared_ptr<ImageDataSet> BinaryImageReader::Read8BitsImage(std::string const & filename)
{
	ifstream in(filename, ifstream::binary);
	
	VERIFY(m_dim.x > 0 && m_dim.y > 0 && m_dim.z > 0);
	VERIFY(m_voxSize.x > 0.0f && m_voxSize.y > 0.0f && m_voxSize.z > 0.0f);

	int sliceSize = m_dim.x * m_dim.y;
	char * buffer = new char[sliceSize];

	auto image = make_shared<ImageDataSet>();

	image->SetDimensions(m_dim.x, m_dim.y, m_dim.z);
	image->SetVoxelSize(m_voxSize.x, m_voxSize.y, m_voxSize.z);

	for (int z = 0; z < m_dim.z; z++)
	{
		in.read(buffer, sliceSize);

		if (!in)
		{
			in.close();
			delete[] buffer;
			throw std::runtime_error("Cannot read from file");
		}

		image->SetSlice(z, buffer);
	}

	return move(image);
}
*/