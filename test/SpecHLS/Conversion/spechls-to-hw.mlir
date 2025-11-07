// RUN: spechls-opt --spechls-to-hw %s | FileCheck %s

//===----------------------------------------------------------------------===//
// 1) Trivial kernel + 2-way gamma → comb.mux
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @k1(
// CHECK-SAME:   %sel: i1, %a: i32, %b: i32,
// CHECK-SAME:   %clk: !seq.clock, %rst: i1, %ce: i1
// CHECK-SAME: ) -> (i32)
spechls.kernel @k1(%sel: i1, %a: i32, %b: i32) -> i32 {
  %v = spechls.gamma<"x">(%sel, %a, %b) : i1, i32
  spechls.exit with %v : i32
}

// Inside body we expect a comb.mux and a single hw.output.
// CHECK:         %[[MUX:.+]] = comb.mux %sel, %b, %a : i32
// CHECK-NEXT:    hw.output %[[MUX]] : i32


//===----------------------------------------------------------------------===//
// 2) Kernel with a child task → child hw.module (+clk/rst/ce) and instance
//    Parent forwards its (%clk,%rst,%ce) to the child instance (same order)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @k2(
// CHECK-SAME:   %x: i32, %y: i32,
// CHECK-SAME:   %clk: !seq.clock, %rst: i1, %ce: i1
// CHECK-SAME: ) -> (i32)
spechls.kernel @k2(%x: i32, %y: i32) -> i32 {
  %r = spechls.task @child(%x: i32, %y: i32) -> i32 {
    // trivial body: return first arg
    spechls.commit %arg0 : i32
  }
  spechls.exit with %r : i32
}

// Child module must exist with (args..., clk, rst, ce) and one i32 result.
// CHECK:       hw.module @child(
// CHECK-SAME:    %arg0: i32, %arg1: i32,
// CHECK-SAME:    %clk: !seq.clock, %rst: i1, %ce: i1
// CHECK-SAME:  ) -> (i32)
// CHECK:         hw.output %arg0 : i32

// Parent body must instantiate child and append (%clk, %rst, %ce) *after* functional args.
// CHECK:       %[[INST:.+]] = hw.instance "child" @child(%x, %y, %clk, %rst, %ce)
// CHECK:       hw.output %[[INST]] : i32


//===----------------------------------------------------------------------===//
// 3) N-way gamma → hw.array_create + hw.array_get
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @k3(
// CHECK-SAME:   %idx: i2, %a0: i32, %a1: i32, %a2: i32, %a3: i32,
// CHECK-SAME:   %clk: !seq.clock, %rst: i1, %ce: i1
// CHECK-SAME: ) -> (i32)
spechls.kernel @k3(%idx: i2, %a0: i32, %a1: i32, %a2: i32, %a3: i32) -> i32 {
  %v = spechls.gamma<"x">(%idx, %a0, %a1, %a2, %a3) : i2, i32
  spechls.exit with %v : i32
}

// CHECK:         %[[ARR:.+]] = hw.array_create %a0, %a1, %a2, %a3 : array<4xi32>
// CHECK-NEXT:    %[[GET:.+]] = hw.array_get %[[ARR]][%idx] : i32
// CHECK-NEXT:    hw.output %[[GET]] : i32


//===----------------------------------------------------------------------===//
// 4) Multi-result kernel → hw.module with multiple outputs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @k4(
// CHECK-SAME:   %a: i32,
// CHECK-SAME:   %clk: !seq.clock, %rst: i1, %ce: i1
// CHECK-SAME: ) -> (i32, i32)
spechls.kernel @k4(%a: i32) -> (i32, i32) {
  spechls.exit with %a, %a : i32, i32
}

// CHECK:         hw.output {{%.*}}, {{%.*}} : i32, i32
