// RUN: spechls-opt %s | FileCheck %s

spechls.fsm.machine @fsm_x0 : (i1, i1) -> (i1, i1) {
}

spechls.kernel @kernel(%input: i1) -> i1 {
  %true = hw.constant 1 : i1
  %result = spechls.task "task"(%arg = %input) : (i1) -> !spechls.struct<"commit_type" {"enable": i1, "value": i1}> {
    spechls.fsm.instance "control" @fsm_x0
    %state, %command = spechls.fsm.trigger "control"(%arg, %arg) : (i1, i1) -> (i1, i1)
    spechls.commit %command, %state : i1, i1
  }
  %value = spechls.field<"value"> %result : !spechls.struct<"commit_type" {"enable": i1, "value": i1}>
  spechls.exit if %true with %value : i1
}

// CHECK: spechls.fsm.machine @fsm_x0 : (i1, i1) -> (i1, i1)
// CHECK: spechls.fsm.instance "control" @fsm_x0
// CHECK: spechls.fsm.trigger "control"(%arg1, %arg1) : (i1, i1) -> (i1, i1)
