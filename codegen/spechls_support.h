//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
#define CODEGEN_INCLUDED_SPECHLS_SUPPORT_H

#include "ap_int.h"

template <typename T>
constexpr T mu(bool first, T init, T loop) {
  return first ? init : loop;
}

template <typename T, int N>
constexpr void delay_init(T (&buffer)[N], T value) {
  for (int i = 0; i < N; ++i)
    buffer[i] = value;
}

template <typename T, int N>
constexpr T delay(T (&buffer)[N], T value, bool enable = true, [[maybe_unused]] T init = {}) {
  T result = buffer[0];
  if (enable) {
    for (int i = 0; i < N - 1; ++i)
      buffer[i] = buffer[i + 1];
    buffer[N - 1] = value;
  }
  return result;
}

template <typename T, int Min, int Max>
constexpr ap_int<Max - Min> extract(T input) {
  return input.range(Min, Max);
}

template <int N, int M>
constexpr ap_int<N * M> replicate(ap_int<M> input) {
  ap_int<N * M> result = 0;
  for (int i = 0; i < N; ++i)
    result = result.concat(input);
  return result;
}

template <int N, int M>
constexpr ap_int<N + M> concat(ap_int<N> lhs, ap_int<M> rhs) {
  return lhs.concat(rhs);
}

#endif // CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
