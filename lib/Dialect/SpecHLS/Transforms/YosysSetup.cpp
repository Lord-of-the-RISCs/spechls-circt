//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/Transforms/YosysSetup.h"

#define _YOSYS_
// push because yosys generate warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <kernel/log.h>
#include <kernel/register.h>
#include <kernel/yosys.h>
#pragma GCC diagnostic pop
#undef _YOSYS_

void spechls::setupYosys() {
  static bool isSetup = false;
  if (!isSetup) {
    isSetup = true;
    Yosys::yosys_setup();
  }
}