// RUN: spechls-opt --spechls-to-hw %s | spechls-opt | FileCheck %s

// CHECK-LABEL: hw.module @kernel
spechls.kernel @kernel() {
  %true = hw.constant 1 : i1
  spechls.exit if %true
}
