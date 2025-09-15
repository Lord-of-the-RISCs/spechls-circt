// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel() {
  %true = hw.constant 1 : i1
  spechls.exit if %true
}
