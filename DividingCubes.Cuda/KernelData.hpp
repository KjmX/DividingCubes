#pragma once

#include "Data/chunk_vector.hpp"
#include <thrust/device_vector.h>
#include <map>

namespace Impacts
{
	namespace Cuda
	{
		template <typename T>
		struct KernelData
		{
			using HostDataType = Data::chunk_vector < T > ;
			using DeviceDataType = thrust::device_vector < T > ;

			std::map<int, std::shared_ptr<HostDataType>> hostData;
			std::map<int, std::shared_ptr<DeviceDataType>> deviceData;
		};
	}
}