#include "pch.h"
#include "FileSystem.hpp"

/*
using namespace Impacts::FileIOSystem;
using namespace std;

shared_ptr<FileData> FileSystem::ReadFile(string const & path, bool binary)
{
	int mode = binary ? ios::in | ios::binary : ios::in;
	ifstream in(path, mode);

	if (!in.is_open())
		throw runtime_error("Unable to open file");

	in.tellg();
	in.seekg(0, ios::end);
	unsigned long size = in.tellg();
	in.seekg(ios::beg);

	char * data = new char[size + 1];

	int i = 0;
	while (in.good())
	{
		data[i] = in.get();
		if (!in.eof())
			++i;
	}

	//in.read(data, size);

	data[size] = 0;

	in.close();

	shared_ptr<FileData> file = make_shared<FileData>(data, size);

	return move(file);
}

void FileSystem::SaveFile(string const & path, char const * data, size_t size, bool append, bool binary)
{
	int mode = binary ? ios::out | ios::binary : ios::out;
	if (append)
		mode |= ios::app;

	ofstream out(path, mode);

	if(!out.is_open())
		throw runtime_error("Unable to open file");

	out.write(data, size);

	out.close();
}
*/