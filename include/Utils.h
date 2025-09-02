//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_UTILS_H
#define SPECHLS_INCLUDED_UTILS_H

#include <cstddef>
#include <cstdint>

namespace utils {

static inline size_t getMinBitwidth(uint64_t maxValue) { return 64 - __builtin_clzll(maxValue); }

} // namespace utils

#endif // SPECHLS_INCLUDED_UTILS_H
