// RUN: spechls-opt %s | spechls-opt --simplify-gamma-nodes | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel(%cond: i1, %x: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.gamma <"x">(%cond, %x, %x) : i1, i32
  %1 = spechls.gamma <"x">(%cond, %0, %x) : i1, i32
  spechls.exit if %true with %1 : i32
}
