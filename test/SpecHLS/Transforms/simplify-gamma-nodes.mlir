// RUN: spechls-opt -split-input-file --simplify-gamma-nodes %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @trivial
spechls.kernel @trivial(%cond: i1, %x: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.gamma <"x">(%cond, %x, %x) : i1, i32
  %1 = spechls.gamma <"x">(%cond, %0, %x) : i1, i32
  // CHECK: spechls.exit if %true with %arg1
  spechls.exit if %true with %1 : i32
}

//---

// CHECK-LABEL: @constant_inputs
spechls.kernel @constant_inputs(%x: i32, %y: i32) -> i32 {
  %true = hw.constant 1 : i1
  %idx = hw.constant 4 : i32
  %0 = spechls.gamma <"x">(%idx, %x, %y, %x, %x, %y) : i32, i32
  // CHECK: spechls.exit if %true with %arg1
  spechls.exit if %true with %0 : i32
}
