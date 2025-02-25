//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_DIALECT_DELAY_H
#define SPECHLS_DIALECT_DELAY_H

namespace spechls {

template <typename T>
mux<T>::DelayLine(int s) {
  ptr = new T[s];
  size = s;
  pos = 0;
}

template <typename T, int N>
void mux(T args...) {
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0') {
    if (*fmt == 'd') {
      int i = va_arg(args, int);
      std::cout << i << '\n';
    } else if (*fmt == 'c') {
      // note automatic conversion to integral type
      int c = va_arg(args, int);
      std::cout << static_cast<char>(c) << '\n';
    } else if (*fmt == 'f') {
      double d = va_arg(args, double);
      std::cout << d << '\n';
    }
    ++fmt;
  }

  va_end(args);
}

template <typename T>
T DelayLine<T>::pop() {
  return ptr[pos];
}

#endif // SPECHLS_DIALECT_DELAY_H
