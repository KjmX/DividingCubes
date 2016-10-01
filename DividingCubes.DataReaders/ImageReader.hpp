#pragma once

#include "Data/ImageDataSet.hpp"
#include <string>
#include <memory>

namespace Impacts
{
	namespace Data
	{
		class ImageReader
		{
		public:
			virtual ~ImageReader() = 0 {}

			virtual int Load(std::string const & filename) = 0;
			virtual std::shared_ptr<ImageDataSet> GetOutput(int id) const = 0;
		};
	}
}
