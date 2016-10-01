#pragma once

#include "../pch.h"
#include "Macros.hpp"
#include <vector>
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_malloc_allocator.h>

namespace Impacts
{
	namespace Cuda
	{
		namespace Data
		{
			template <class T>
			class chunk_vector
			{
				std::vector<std::shared_ptr<thrust::host_vector<T>>> m_data;
				size_t m_size;

			public:

				chunk_vector() : m_size(0)
				{

				}

				void push_back(std::shared_ptr<thrust::host_vector<T>> chunk)
				{
					m_size += chunk->size();
					m_data.push_back(std::move(chunk));
				}

				size_t size() const
				{
					return m_size;
				}

				std::shared_ptr<thrust::device_vector<T, thrust::device_malloc_allocator<T>>> copy_to_device(unsigned int index, size_t size)
				{
					ARGUMENT(index < m_size);
					ARGUMENT(size <= (m_size - index));

					auto tIdx = index;
					int startChunkIdx = -1;
					unsigned int startChunkEleIdx;

					for (auto i = 0; i < m_data.size(); i++)
					{
						std::shared_ptr<thrust::host_vector<T>> const & chunk = m_data[i];
						if (tIdx < chunk->size())
						{
							startChunkIdx = i;
							startChunkEleIdx = tIdx;
							break;
						}
						tIdx -= chunk->size();
					}

					ASSERT(startChunkIdx > -1);

					auto tSize = size;
					auto offset = 0;

					auto d_data = std::make_shared<thrust::device_vector<T, thrust::device_malloc_allocator<T>>>(size);

					for (auto i = startChunkIdx; i < m_data.size(); i++)
					{
						std::shared_ptr<thrust::host_vector<T>> const & chunk = m_data[i];

						size_t endChunkEleIdx = chunk->size() < tSize ? chunk->size() :
							(chunk->size() > (tSize + startChunkEleIdx) ? (tSize + startChunkEleIdx) : chunk->size());

						auto ptr = thrust::raw_pointer_cast(chunk->data());
						auto sizeToCopy = endChunkEleIdx - startChunkEleIdx;

						auto raw_ptr = raw_pointer_cast(d_data->data());

						gpuErrchk(cudaMemcpy(raw_ptr + offset, ptr + startChunkEleIdx, sizeToCopy * sizeof(T), cudaMemcpyHostToDevice));

						offset += sizeToCopy;
						tSize -= sizeToCopy;
						if (tSize <= 0 && offset == size)
							break;

						startChunkEleIdx = 0;
					}
					return std::move(d_data);
				}
			};
		}
	}
}