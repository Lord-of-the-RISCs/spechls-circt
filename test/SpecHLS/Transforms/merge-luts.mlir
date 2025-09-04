// RUN: spechls-opt -split-input-file --merge-luts %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @constant
spechls.kernel @constant() -> i32 {
  %true = hw.constant 1 : i1
  %0 = hw.constant 2 : i2
  // CHECK: %[[c:.*]] = hw.constant 4 : i32
  // CHECK: spechls.exit if %true with %[[c]] : i32
  %1 = spechls.lut %0 [0, 2, 4, 6] : (i2) -> i32
  spechls.exit if %true with %1 : i32
}

//---

// CHECK-LABEL: @consecutive
spechls.kernel @consecutive(%x: i32) -> i3 {
  %true = hw.constant 1 : i1
  %0 = spechls.lut %x [1, 3, 2, 1] : (i32) -> i2
  // CHECK: %{{.*}} = spechls.lut %arg0 [2, 6, 4, 2] : (i32) -> i3
  %1 = spechls.lut %0 [0, 2, 4, 6] : (i2) -> i3
  spechls.exit if %true with %1 : i3
}
