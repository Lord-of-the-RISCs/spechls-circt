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

template <typename T, int ID>
T mu(T init, T loop) {
  static bool f = true;
  T result = f ? init : loop;
  f = false;
  return result;
}

template <typename T, unsigned int N>
void delay_init(T (&buffer)[N], T value) {
  for (int i = 0; i < N; ++i)
    buffer[i] = value;
}

template <typename T, unsigned int N>
T delay_pop(T (&buffer)[N]) {
  return buffer[0];
}

template <typename T, unsigned int N>
void delay_push(T (&buffer)[N], T value, bool enable = true) {
  if (enable) {
    for (int i = 0; i < N - 1; ++i)
      buffer[i] = buffer[i + 1];
    buffer[N - 1] = value;
  }
}

template <typename T, int Min, int Max>
ap_int<Max - Min> extract(T input) {
  return input.range(Max - 1, Min);
}

template <int N, int M>
ap_int<N * M> replicate(ap_int<M> input) {
  ap_int<N * M> result = 0;
  for (int i = 0; i < N; ++i)
    result = result.concat(input);
  return result;
}

template <int N, int M>
ap_uint<N * M> replicate(ap_uint<M> input) {
  ap_uint<N * M> result = 0;
  for (int i = 0; i < N; ++i)
    result = result.concat(input);
  return result;
}

template <int N, int M>
ap_int<N + M> concat(ap_int<N> lhs, ap_int<M> rhs) {
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
T rollback(T *buffer, T value, unsigned int offset, bool next_input) {
  constexpr unsigned int max_depth = std::max({0u, Depths...});
  unsigned int off = offset - Offset;
  if (next_input) {
    for (unsigned int i = max_depth; i > 0; --i)
      buffer[i] = buffer[i - 1];
  }
  T result = resolve_offset<T, Depths...>(off, buffer, value);
  if (next_input)
    buffer[0] = result;

  return result;
}

template <unsigned int Offset>
ap_uint<1> cancel(ap_uint<1> *buffer, ap_uint<1> value, unsigned int offset, bool next_input) {
  if (next_input)
    *buffer = value;
  if (offset >= Offset)
    *buffer = false;
  return *buffer;
}

template <typename T, unsigned int... Depths>
T rewind(T *buffer, T value, int offset, bool next_input) {
  constexpr unsigned int max_depth = std::max({0u, Depths...});
  if (next_input) {
    for (unsigned int i = max_depth; i > 0; --i) {
      buffer[i] = buffer[i - 1];
    }
    buffer[0] = value;
  }
  return buffer[offset > 0 ? offset - 1 : 0];
}

template <typename T, typename... Ts>
T gamma(unsigned int select, Ts... values) {
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
T *alpha(T *array, unsigned int index, T value, bool we) {
  if (we) {
    array[index] = value;
  }
  return array;
}

template <typename T>
struct FifoType {
  T data{};
  bool full = false;
  bool empty = true;
};

template <typename T>
struct FifoInputType {
  bool write = false;
  T data{};
};

template <typename T>
void fifo_read(FifoType<T> &fifo) {
  fifo.empty = true;
  fifo.full = false;
}

template <typename Arg, typename T>
void fifo_write(FifoType<T> &fifo, const Arg &input) {
  if (input.write) {
    fifo.empty = false;
    fifo.full = true;
    fifo.data = input.data;
  }
}

template <typename OutType, typename InType, int... Values>
OutType lut(InType index) {
  OutType result{};
  int idx = 0;
  auto update = [&](int value) {
    if (index == idx++)
      result = value;
  };
  (update(Values), ...);
  return result;
}

#endif // CODEGEN_INCLUDED_SPECHLS_SUPPORT_H
