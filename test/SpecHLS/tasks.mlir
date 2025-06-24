// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @toplevel
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: i32, %[[arg2:[a-zA-Z0-9]+]]: i1
spechls.kernel @toplevel(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %[[res:.+]] = spechls.task "task1" -> i32
  %0 = spechls.task "task1" -> i32 {
    %true = hw.constant 1 : i1
    // CHECK: spechls.commit %true, %[[arg1]] : i32
    spechls.commit %true, %in1 : i32
  }
  // CHECK: spechls.exit if %[[arg2]] with %[[res]] : i32
  spechls.exit if %in2 with %0 : i32
}
