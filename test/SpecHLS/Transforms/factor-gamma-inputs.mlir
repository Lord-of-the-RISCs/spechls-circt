// RUN: spechls-opt -split-input-file --factor-gamma-inputs %s | FileCheck %s
// (If your pass flag differs, replace --factor-gamma-inputs accordingly)

//===----------------------------------------------------------------------===//
// Two matched comb.add arms among 4 inputs (positions 0 and 2).
// Expect:  (i)  a LUT reindexing %arg0 over {0,2} => [0,0,1,0] : (i2)->i1
//          (ii) two inner gammas (one per operand) driven by the LUT
//          (iii) one shared comb.add using those inner gammas
//          (iv) outer gamma rewired to use the shared result at 0 and 2
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @merge_add_two_arms
spechls.kernel @merge_add_two_arms(%sel: i2, %a0: i32, %b0: i32, %a1: i32, %b1: i32, %z: i32) -> i32 {
  %true = hw.constant 1 : i1

  %p0 = comb.add %a0, %b0 : i32
  %p1 = comb.add %a1, %b1 : i32

  %y  = spechls.gamma<"x">(%sel, %p0, %z, %p1, %z) : i2, i32

  // CHECK: %[[LUT:.*]] = spechls.lut %arg0 [0, 0, 1, 0] : (i2) -> i1
  // CHECK: %[[GA:.*]] = spechls.gamma<"x">(%[[LUT]], %arg1, %arg3) : i1, i32
  // CHECK: %[[GB:.*]] = spechls.gamma<"x">(%[[LUT]], %arg2, %arg4) : i1, i32
  // CHECK: %[[ADD:.*]] = comb.add %[[GA]], %[[GB]] : i32
  // CHECK: %[[OUT:.*]] = spechls.gamma<"x">(%arg0, %[[ADD]], %arg5, %[[ADD]], %arg5) : i2, i32
  // CHECK: spechls.exit if {{.*}} with %[[OUT]] : i32

  spechls.exit if %true with %y : i32
}

//---

//===----------------------------------------------------------------------===//
// Three matched comb.mul arms among 5 inputs (positions 1, 2, and 4).
// Select is i3 (8-entry LUT). Expect a LUT: [0,0,1,0,2,0,0,0] : (i3)->i2,
// two inner gammas (per operand), one shared comb.mul, and outer gamma
// uses shared result at positions 1,2,4.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @merge_mul_three_arms
spechls.kernel @merge_mul_three_arms(%sel: i3,
                                     %a0: i32, %b0: i32,
                                     %a1: i32, %b1: i32,
                                     %z:  i32,
                                     %a2: i32, %b2: i32) -> i32 {
  %true = hw.constant 1 : i1

  %m0 = comb.mul %a0, %b0 : i32
  %m1 = comb.mul %a1, %b1 : i32
  %m2 = comb.mul %a2, %b2 : i32

  // Arms: [ %z, %m0, %m1, %z, %m2 ]
  %y  = spechls.gamma<"x">(%sel, %z, %m0, %m1, %z, %m2) : i3, i32

  // CHECK: %[[LUT:.*]] = spechls.lut %arg0 [0, 0, 1, 0, 2, 0, 0, 0] : (i3) -> i2
  // CHECK: %[[GA:.*]] = spechls.gamma<"x">(%[[LUT]], %arg1, %arg3, %arg6) : i2, i32
  // CHECK: %[[GB:.*]] = spechls.gamma<"x">(%[[LUT]], %arg2, %arg4, %arg7) : i2, i32
  // CHECK: %[[MUL:.*]] = comb.mul %[[GA]], %[[GB]] : i32
  // CHECK: %[[OUT:.*]] = spechls.gamma<"x">(%arg0, %arg5, %[[MUL]], %[[MUL]], %arg5, %[[MUL]]) : i3, i32
  // CHECK: spechls.exit if {{.*}} with %[[OUT]] : i32

  spechls.exit if %true with %y : i32
}

//---

//===----------------------------------------------------------------------===//
// Identical constants behind different arms (zero-operand producers).
// Expect: NO inner LUT/gammas; the pass simply rewires matched arms to refer
// to the same constant SSA value (chosen root). We only assert the absence
// of LUT and the duplicated use of the same constant in the outer gamma.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @merge_constants
spechls.kernel @merge_constants(%sel: i2, %x: i32) -> i32 {
  %true = hw.constant 1 : i1

  %c0 = hw.constant 42 : i32
  %c1 = hw.constant 42 : i32
  %y  = spechls.gamma<"x">(%sel, %c0, %x, %c1, %x) : i2, i32

  // CHECK: %[[C:.*]] = hw.constant 42 : i32
  // CHECK-NOT: spechls.lut
  // CHECK: %[[OUT:.*]] = spechls.gamma<"x">(%arg0, %[[C]], %arg1, %[[C]], %arg1) : i2, i32
  // CHECK: spechls.exit if {{.*}} with %[[OUT]] : i32

  spechls.exit if %true with %y : i32
}
