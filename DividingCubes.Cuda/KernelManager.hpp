#pragma once

#include "Macros.hpp"

#include "KernelDimension.hpp"
#include "KernelDataManager.hpp"
#include "IKernel.hpp"
#include "DeviceHelper.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cmath>

namespace Impacts
{
	namespace Cuda
	{
		template <typename TIn, typename TOut>
		class KernelManager
		{
			std::shared_ptr<KernelDataManager<TOut>> m_outputDataMgr;

			int m_ids;
			int m_threadCount;
			int m_warpSize;
			KernelDimension m_kernelDim;

			int m_totalIter;
			int m_cubesPerIter;
			int m_totalCubes;
			int m_totalBytes;
			MemoryType m_inputMemType;
			MemoryType m_outputMemType;
			size_t m_outputElementSize;
			int m_inputId;
			int m_outputId;
			std::shared_ptr<KernelDataManager<TIn>> m_inputDataMgr;

		public:
			KernelManager()
			{
				m_outputDataMgr = std::make_shared <KernelDataManager<TOut>>();
				m_ids = 0;
				GetDeviceInfo();
				m_kernelDim.block = dim3(m_warpSize, 1, 1);
				m_kernelDim.grid = dim3(m_threadCount / m_warpSize, 1, 1);
				m_outputId = -1;
			}

			void Setup(size_t totalCubes, int inputId, std::shared_ptr<KernelDataManager<TIn>> const & inputDataMgr, bool forceCpu = false)
			{
				size_t free, total;
				DeviceHelper::GetCudaMemInfo(&free, &total);

				auto outputElementSize = sizeof(TOut);

				int totalBytes = totalCubes * outputElementSize;
				auto cubeSize = outputElementSize;

				m_inputMemType = inputDataMgr->GetMemoryType(inputId);
				if (m_inputMemType == CPU)
				{
					totalBytes += totalCubes * sizeof(TIn);
					cubeSize += sizeof(TIn);
				}

				m_inputId = inputId;
				m_inputDataMgr = inputDataMgr;
				m_outputElementSize = outputElementSize;
				m_totalIter = ceil(static_cast<float>(totalBytes) / static_cast<float>(free));
				m_cubesPerIter = free / cubeSize;
				m_totalCubes = totalCubes;
				m_totalBytes = totalBytes;
				m_outputMemType = (m_totalIter > 1) || forceCpu ? CPU : GPU;
			}

			void Run(std::shared_ptr<IKernel<TIn, TOut>> const & kernel)
			{
				RunKernel(kernel);
			}

			void Run(std::shared_ptr<IKernel<TIn, TOut>> const & kernel, std::vector<TOut> * customCpuSpace)
			{
				RunKernel(kernel, customCpuSpace);
			}

			std::shared_ptr<KernelDataManager<TOut>> GetOutput() const
			{
				return m_outputDataMgr;
			}

			int GetOutputId() const
			{
				return m_outputId;
			}

			size_t GetOutputSize(int outputId) const
			{
				return m_outputDataMgr->GetSize(outputId);
			}

			/*static void GetCudaMemInfo(size_t * free, size_t * total)
			{
			gpuErrchk(cudaMemGetInfo(free, total));
			}*/

		private:

			void RunKernel(std::shared_ptr<IKernel<TIn, TOut>> const & kernel,
				std::vector<TOut> * customCpuSpace = nullptr)
			{
				auto totalCubes = m_totalCubes;
				m_outputId = -1;

				for (auto i = 0; i < m_totalIter; i++)
				{
					auto liveCubes = totalCubes > m_cubesPerIter ? m_cubesPerIter : totalCubes;
					auto range = liveCubes <= m_threadCount ? 1 : ceil(static_cast<float>(liveCubes) / static_cast<float>(m_threadCount));

					std::shared_ptr<typename KernelData<TIn>::DeviceDataType> d_input;

					// Read input
					if (m_inputMemType == CPU)
						d_input = m_inputDataMgr->ReadToDevice(m_inputId, CPU, i * m_cubesPerIter, liveCubes);
					else if (m_inputMemType == GPU)
						d_input = m_inputDataMgr->ReadToDevice(m_inputId, GPU);

					// Run kernel
					size_t sizeToCopy = kernel->Run(d_input, m_kernelDim, i, m_cubesPerIter, liveCubes,
						m_totalCubes, range, m_inputMemType == CPU ? 0 : 1);

					// Write output
					if (m_outputMemType == CPU)
					{
						if (customCpuSpace == nullptr)
						{
							auto h_output = std::make_shared<thrust::host_vector<TOut>>(sizeToCopy);
							auto d_output = kernel->GetOutput();
							thrust::copy(d_output->begin(), d_output->begin() + sizeToCopy, h_output->begin());
							m_outputId = m_outputDataMgr->WriteToHost(h_output, m_outputId);
						}
						else
						{
							// TODO: remove this and add copy to one global space feature
							auto d_output = kernel->GetOutput();
							gpuErrchk(cudaMemcpy(customCpuSpace->data() + i * m_cubesPerIter,
								thrust::raw_pointer_cast(d_output->data()), sizeof(TOut) * sizeToCopy, cudaMemcpyDeviceToHost));
							m_outputId = -1;
						}

						RemoveDeviceVector(kernel->GetOutput().get());
					}
					else if (m_outputMemType == GPU)
					{
						m_outputId = m_outputDataMgr->WriteToDevice(kernel->GetOutput());
					}

					totalCubes -= m_cubesPerIter;
				}
			}

			void GetDeviceInfo()
			{
				int deviceCount = 0;
				gpuErrchk(cudaGetDeviceCount(&deviceCount));

				ASSERT(deviceCount > 0);

				cudaSetDevice(0);
				cudaDeviceProp deviceProp;
				cudaGetDeviceProperties(&deviceProp, 0);

				m_threadCount = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
				m_warpSize = deviceProp.warpSize;
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

