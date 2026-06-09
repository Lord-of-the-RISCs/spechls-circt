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

   // CHECK: %c1_i2 = hw.constant 1 : i2
   // CHECK: %c0_i2 = hw.constant 0 : i2
   // CHECK: %true = hw.constant true
   // CHECK: %0 = comb.extract %arg1 from 0 : (i32) -> i2
   // CHECK: %1 = comb.extract %arg0 from 0 : (i32) -> i2
   // CHECK: %2 = comb.icmp ult %0, %c0_i2 : i2
   // CHECK: %3 = comb.icmp eq %0, %c0_i2 : i2
   // CHECK: %4 = comb.add %0, %1 : i2
   // CHECK: %5 = comb.add %0, %c1_i2 : i2
   // CHECK: %6 = comb.mux %3, %4, %5 : i2
   // CHECK: %7 = comb.mux %2, %0, %6 : i2
   // CHECK: %gamma = spechls.gamma<"x">(%7, %arg2, %arg3, %arg4) {}: i2, i32
   // CHECK: spechls.exit if %true with %gamma : i32


  %result = spechls.gamma<"x">(%c2, %g, %2) : i32, i32
  spechls.exit if %true with %result : i32
}

//---

// CHECK-LABEL: @merge_tree
spechls.kernel @merge_tree(%c1: i32, %c2: i32, %c3: i32, %c4: i32, %0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32) -> i32 {
    // CHECK: %c1_i3 = hw.constant 1 : i3
    // CHECK: %c2_i3 = hw.constant 2 : i3
    // CHECK: %false = hw.constant false
    // CHECK: %c1_i2 = hw.constant 1 : i2
    // CHECK: %c0_i2 = hw.constant 0 : i2
    // CHECK: %true = hw.constant true
    // CHECK: %0 = comb.extract %arg3 from 0 : (i32) -> i2
    // CHECK: %1 = comb.extract %arg2 from 0 : (i32) -> i2
    // CHECK: %2 = comb.icmp ult %0, %c0_i2 : i2
    // CHECK: %3 = comb.icmp eq %0, %c0_i2 : i2
    // CHECK: %4 = comb.add %0, %1 : i2
    // CHECK: %5 = comb.add %0, %c1_i2 : i2
    // CHECK: %6 = comb.mux %3, %4, %5 : i2
    // CHECK: %7 = comb.mux %2, %0, %6 : i2
    // CHECK: %8 = comb.concat %false, %7 : i1, i2
    // CHECK: %9 = comb.extract %arg1 from 0 : (i32) -> i3
    // CHECK: %10 = comb.icmp ult %8, %c2_i3 : i3
    // CHECK: %11 = comb.icmp eq %8, %c2_i3 : i3
    // CHECK: %12 = comb.add %8, %9 : i3
    // CHECK: %13 = comb.add %8, %c2_i3 : i3
    // CHECK: %14 = comb.mux %11, %12, %13 : i3
    // CHECK: %15 = comb.mux %10, %8, %14 : i3
    // CHECK: %16 = comb.extract %arg0 from 0 : (i32) -> i3
    // CHECK: %17 = comb.icmp ult %15, %c2_i3 : i3
    // CHECK: %18 = comb.icmp eq %15, %c2_i3 : i3
    // CHECK: %19 = comb.add %15, %16 : i3
    // CHECK: %20 = comb.add %15, %c1_i3 : i3
    // CHECK: %21 = comb.mux %18, %19, %20 : i3
    // CHECK: %22 = comb.mux %17, %15, %21 : i3
    // CHECK: %gamma = spechls.gamma<"x">(%22, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) {}: i3, i32
    // CHECK: spechls.exit if %true with %gamma : i32

  %true = hw.constant 1 : i1
  %g1 = spechls.gamma<"x">(%c1, %2, %3) : i32, i32
  %g2 = spechls.gamma<"x">(%c2, %g1, %4, %5) : i32, i32
  %g3 = spechls.gamma<"x">(%c3, %0, %1) : i32, i32
  %result = spechls.gamma<"x">(%c4, %g3, %g2) : i32, i32
  spechls.exit if %true with %result : i32
}
