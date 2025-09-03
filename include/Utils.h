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

static inline size_t getMinBitwidth(uint64_t maxValue) {
  // return 64 - __builtin_clzll(maxValue);
  if (maxValue == 0)
    return 0;

  size_t n = 1;

  // clang-format off
  if ((maxValue & 0xffffffff00000000ull) == 0) { n += 32; maxValue <<= 32; }
  if ((maxValue & 0xffff000000000000ull) == 0) { n += 16; maxValue <<= 16; }
  if ((maxValue & 0xff00000000000000ull) == 0) { n +=  8; maxValue <<=  8; }
  if ((maxValue & 0xf000000000000000ull) == 0) { n +=  4; maxValue <<=  4; }
  if ((maxValue & 0xc000000000000000ull) == 0) { n +=  2; maxValue <<=  2; }
  // clang-format on

  return 64 - (n - (maxValue >> 63));
}

} // namespace utils

#endif // SPECHLS_INCLUDED_UTILS_H
