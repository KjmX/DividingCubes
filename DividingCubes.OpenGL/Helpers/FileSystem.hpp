#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>

namespace Impacts
{
	namespace FileIOSystem
	{
		class FileData
		{
			char * m_data;
			size_t m_size;

			FileData(FileData const &) = delete;
			FileData & operator=(FileData const &) = delete;

		public:
			FileData(char * data, size_t size) : m_data(data), m_size(size)
			{ }

			~FileData()
			{
				if(m_data != nullptr)
					delete[] m_data;
			}

			const char * GetData() const { return m_data; }

			char * GetRawData() const { return m_data; }

			size_t GetSize() const { return m_size; }
		};

		class FileSystem
		{
		public:
			std::shared_ptr<FileData> ReadFile(std::string const & path, bool binary = false)
			{
				int mode = binary ? std::ios::in | std::ios::binary : std::ios::in;
				std::ifstream in(path, mode);

				if (!in.is_open())
					throw std::runtime_error("Unable to open file");

				in.tellg();
				in.seekg(0, std::ios::end);
				unsigned long size = in.tellg();
				in.seekg(std::ios::beg);

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

				std::shared_ptr<FileData> file = std::make_shared<FileData>(data, size);

				return std::move(file);
			}

			void SaveFile(std::string const & path, char const * data, size_t size, bool append = false, bool binary = false)
			{
				int mode = binary ? std::ios::out | std::ios::binary : std::ios::out;
				if (append)
					mode |= std::ios::app;

				std::ofstream out(path, mode);

				if (!out.is_open())
					throw std::runtime_error("Unable to open file");

				out.write(data, size);

				out.close();
			}
		};
	}
}
