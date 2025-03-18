// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @task
spechls.htask @task(%state : i32, %enable : i1, %arg : i32) -> i32 {
  // CHECK: %0 = spechls.print %arg0, %arg1, "Hello, world!\0A"
  %new_state = spechls.print %state, %enable, "Hello, world!\n"
  // CHECK: %1 = spechls.print %0, %arg1, "Number = %d\0A", %arg2 : i32
  %new_state_2 = spechls.print %new_state, %enable, "Number = %d\n", %arg : i32
  spechls.commit %new_state_2 : i32
}
