//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_INIT_ALL_TRANSLATIONS_H
#define SPECHLS_INCLUDED_INIT_ALL_TRANSLATIONS_H

namespace SpecHLS {

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllTranslations() {
  static bool initOnce = []() { return true; }();
  (void)initOnce;
}
} // namespace SpecHLS

#endif // SPECHLS_INCLUDED_INIT_ALL_TRANSLATIONS_H
