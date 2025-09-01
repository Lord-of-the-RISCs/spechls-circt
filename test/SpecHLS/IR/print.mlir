// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel(%state : i32, %enable : i1, %arg : i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %io_state = spechls.print %arg0, %arg1, "Hello, world!\0A"
  %new_state = spechls.print %state, %enable, "Hello, world!\n"
  // CHECK: %io_state_0 = spechls.print %io_state, %arg1, "Number = %d\0A", %arg2 : i32
  %new_state_2 = spechls.print %new_state, %enable, "Number = %d\n", %arg : i32
  spechls.exit if %true with %new_state_2 : i32
}
