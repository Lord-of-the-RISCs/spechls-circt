// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @toplevel
spechls.hkernel @toplevel(%in1 : i32) -> i32 {
  %0 = spechls.launch @task(%in1) : (i32) -> i32
  spechls.exit %0 : i32
}

// CHECK: spechls.htask @task
spechls.htask @task(%in1 : i32) -> i32 {
  spechls.commit %in1 : i32
}
