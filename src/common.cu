/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   common.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */


#include <tiny-cuda-nn/common.h>

#include <algorithm>
#include <cctype>


TCNN_NAMESPACE_BEGIN

static_assert(
	__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2),
	"tiny-cuda-nn requires at least CUDA 10.2"
);

uint32_t cuda_compute_capability(int device) {
	cudaDeviceProp props;
	CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
	return props.major * 10 + props.minor;
}

std::string to_lower(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::tolower(c); });
	return str;
}

std::string to_upper(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::toupper(c); });
	return str;
}

TCNN_NAMESPACE_END
