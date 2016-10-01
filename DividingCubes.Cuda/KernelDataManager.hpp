#pragma once

#include "KernelData.hpp"

namespace Impacts
{
	namespace Cuda
	{
		enum MemoryType
		{
			CPU = 100,
			GPU
		};

		template <typename T>
		class KernelDataManager
		{
			KernelData<T> m_data;
			int m_ids;
			std::map<int, MemoryType> m_memoryTypes;

		public:
			KernelDataManager() : m_ids(0)
			{
			}

			std::shared_ptr<typename KernelData<T>::DeviceDataType> ReadToDevice(int id, MemoryType type, unsigned index = 0, size_t size = 0) const
			{
				std::shared_ptr<typename KernelData<T>::DeviceDataType> d_data;

				if (type == CPU)
				{
					auto const h_data = m_data.hostData.at(id);
					d_data = h_data->copy_to_device(index, size);
				}
				else if (type == GPU)
				{
					d_data = m_data.deviceData.at(id);
				}

				return d_data;
			}

			int WriteToHost(std::shared_ptr<thrust::host_vector<T>> const & data, int id = -1)
			{
				if (m_data.hostData.find(id) != m_data.hostData.end())
					m_data.hostData.at(id)->push_back(data);
				else
				{
					id = GetId();
					auto chunkVec = std::make_shared<typename KernelData<T>::HostDataType>();
					chunkVec->push_back(data);
					m_data.hostData.insert(std::pair<int, std::shared_ptr<typename KernelData<T>::HostDataType>>(id, chunkVec));
					AddMemoryType(id, CPU);
				}

				return id;
			}

			int WriteToDevice(std::shared_ptr<thrust::device_vector<T>> const & data)
			{
				auto id = GetId();
				m_data.deviceData.insert(std::pair<int, std::shared_ptr<typename KernelData<T>::DeviceDataType>>(id, data));
				AddMemoryType(id, GPU);

				return id;
			}

			void DeleteFromDevice(int id)
			{
				auto iter = m_data.deviceData.find(id);
				if (iter == m_data.deviceData.end())
					return;

				RemoveDeviceVector(iter->second.get());
				m_data.deviceData.erase(iter);
			}

			MemoryType GetMemoryType(int id)
			{
				return m_memoryTypes[id];
			}

			size_t GetSize(int id)
			{
				auto type = GetMemoryType(id);

				if (type == CPU)
				{
					if (m_data.hostData.find(id) != m_data.hostData.end())
						return m_data.hostData.at(id)->size();
				}
				if (type == GPU)
					if (m_data.deviceData.find(id) != m_data.deviceData.end())
						return m_data.deviceData.at(id)->size();
				return 0;
			}

		private:
			int GetId()
			{
				return m_ids++;
			}

			void AddMemoryType(int id, MemoryType memType)
			{
				if (m_memoryTypes.find(id) == m_memoryTypes.end())
					m_memoryTypes.insert(std::pair<int, MemoryType>(id, memType));
			}

			template <typename T>
			void RemoveDeviceVector(thrust::device_vector<T> * vec)
			{
				vec->clear();
				thrust::device_vector<T>().swap(*vec);
			}
		};
	}
}