// RUN: spechls-opt -split-input-file --simplify-luts %s | spechls-opt | FileCheck %s

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

//---

// CHECK-LABEL: @concat
spechls.kernel @concat(%arg0: i32, %arg1: i32) -> i2 {
  %true = hw.constant true
  %idx0 = comb.extract %arg0 from 0 : (i32) -> i1
  %idx1 = comb.extract %arg1 from 0 : (i32) -> i1
  %0 = spechls.lut %idx0 [0, 1] : (i1) -> i1
  %1 = spechls.lut %idx1 [1, 0] : (i1) -> i1
  %2 = comb.concat %0, %1 : i1, i1
  // CHECK: %[[idx:.*]] = comb.concat %{{.*}}, %{{.*}} : i1, i1
  // CHECK: %{{.*}} = spechls.lut %[[idx]] [2, 3, 0, 1] : (i2) -> i2
  %3 = spechls.lut %2 [3, 2, 1, 0] : (i2) -> i2
  spechls.exit if %true with %3 : i2
}
