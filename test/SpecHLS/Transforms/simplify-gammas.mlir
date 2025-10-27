// RUN: spechls-opt -split-input-file --simplify-gammas %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @trivial
spechls.kernel @trivial(%cond: i1, %x: i32) -> i32 {
  %true = hw.constant 1 : i1
  %0 = spechls.gamma<"x">(%cond, %x, %x) : i1, i32
  %1 = spechls.gamma<"x">(%cond, %0, %x) : i1, i32
  // CHECK: spechls.exit if %true with %arg1
  spechls.exit if %true with %1 : i32
}

//---

// CHECK-LABEL: @constant_inputs
spechls.kernel @constant_inputs(%x: i32, %y: i32) -> i32 {
  %true = hw.constant 1 : i1
  %idx = hw.constant 4 : i32
  %0 = spechls.gamma<"x">(%idx, %x, %y, %x, %x, %y) : i32, i32
  // CHECK: spechls.exit if %true with %arg1
  spechls.exit if %true with %0 : i32
}

//--

// CHECK-LABEL: @merge_binary
spechls.kernel @merge_binary(%c1: i32, %c2: i32, %0: i32, %1: i32, %2: i32) -> i32 {
  %true = hw.constant 1 : i1
  %g = spechls.gamma<"x">(%c1, %0, %1) : i32, i32
  // CHECK: %lut = spechls.lut %2 [0, 1, 2, 2] : (i2) -> i2
  // CHECK: %gamma = spechls.gamma<"x">(%lut, %arg2, %arg3, %arg4) {}: i2, i32
  %result = spechls.gamma<"x">(%c2, %g, %2) : i32, i32
  spechls.exit if %true with %result : i32
}

//---

// CHECK-LABEL: @merge_tree
spechls.kernel @merge_tree(%c1: i32, %c2: i32, %c3: i32, %c4: i32, %0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32) -> i32 {
  %true = hw.constant 1 : i1
  %g1 = spechls.gamma<"x">(%c1, %2, %3) : i32, i32
  %g2 = spechls.gamma<"x">(%c2, %g1, %4, %5) : i32, i32
  %g3 = spechls.gamma<"x">(%c3, %0, %1) : i32, i32
  // CHECK: %{{.*}} = spechls.lut %{{.*}} [0, 1, 2, 2] : (i2) -> i2
  // CHECK: %{{.*}} = spechls.lut %{{.*}} [0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 4, 0, 0, 0, 0, 0] : (i4) -> i3
  // CHECK: %{{.*}} = spechls.lut %{{.*}} [0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0] : (i4) -> i3
  // CHECK: %{{.*}} = spechls.gamma<"x">(%{{.*}}, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) {}: i3, i32
  %result = spechls.gamma<"x">(%c4, %g3, %g2) : i32, i32
  spechls.exit if %true with %result : i32
}
