//===- InitAllTranslations.h - CIRCT Translations Registration --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all translations
// in and out of CIRCT to the system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLTRANSLATIONS_H
#define CIRCT_INITALLTRANSLATIONS_H

namespace SpecHLS {

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllTranslations() {
  static bool initOnce = []() { return true; }();
  (void)initOnce;
}
} // namespace SpecHLS

#endif // CIRCT_INITALLTRANSLATIONS_H
