//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
#define CODEGEN_INCLUDED_SPECHLS_SUPPORT_H

#include <algorithm>

#include "ap_int.h"

template <typename T>
constexpr T mu(bool first, T init, T loop) {
  return first ? init : loop;
}

template <typename T, unsigned int N>
constexpr void delay_init(T (&buffer)[N], T value) {
  for (int i = 0; i < N; ++i)
    buffer[i] = value;
}

template <typename T, unsigned int Depth>
constexpr T delay(T *buffer, T value, bool enable = true, [[maybe_unused]] T init = {}) {
  T result = buffer[0];
  if (enable) {
    for (int i = 0; i < Depth - 1; ++i)
      buffer[i] = buffer[i + 1];
    buffer[Depth - 1] = value;
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
constexpr ap_uint<N * M> replicate(ap_uint<M> input) {
  ap_uint<N * M> result = 0;
  for (int i = 0; i < N; ++i)
    result = result.concat(input);
  return result;
}

template <int N, int M>
constexpr ap_int<N + M> concat(ap_int<N> lhs, ap_int<M> rhs) {
  return lhs.concat(rhs);
}

template <typename T, unsigned int... Depths>
constexpr T resolve_offset(unsigned int offset, T *values, T default_value) {
  T result = default_value;
  auto update = [&](unsigned int d) {
    if (offset == d)
      result = values[d];
  };
  (update(Depths), ...);
  return result;
}

template <typename T, unsigned int Offset, unsigned int... Depths>
constexpr T rollback(T *buffer, T value, unsigned int offset, bool next_input) {
  constexpr unsigned int max_depth = std::max({0u, Depths...}) + 1;
  unsigned int off = offset - Offset;
  if (next_input) {
    for (int i = 0; i < max_depth - 1; ++i)
      buffer[i] = buffer[i + 1];
  }
  T result = resolve_offset<T, Depths...>(off, buffer, value);

  return result;
}

template <typename T, typename... Ts>
constexpr T gamma(unsigned int select, Ts... values) {
  T result = T{};
  unsigned int idx = 0;
  auto update = [&](T value) {
    if (select == idx++)
      result = value;
  };
  (update(values), ...);
  return result;
}

template <typename T>
constexpr T *alpha(T *array, unsigned int index, T value, bool we) {
  if (we) {
    array[index] = value;
  }
  return array;
}

template <typename Result, typename Arg, int Depth>
constexpr Result fifo(Arg arg) {
  // TODO
  return Result{};
}

#endif // CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
