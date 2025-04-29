//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
#define CODEGEN_INCLUDED_SPECHLS_SUPPORT_H

template <typename T, unsigned int N>
T delay(T (&buffer)[N], T value, bool enable = true) {
  T result = buffer[0];
  if (enable) {
    for (unsigned int i = 0; i < N - 1; ++i)
      buffer[i] = buffer[i + 1];
    buffer[N - 1] = value;
  }
  return result;
}

template <typename T, unsigned int... Depths>
T rewind() {}

#endif // CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
