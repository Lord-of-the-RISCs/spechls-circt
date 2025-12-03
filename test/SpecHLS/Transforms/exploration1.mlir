// RUN: spechls-opt -split-input-file --speculation-exploration="targetTask=task targetClock=10" %s | spechls-opt | FileCheck %s

spechls.kernel @kernel(%x_init: i32) -> i32 {
  %true0 = hw.constant 1 : i1
  %0 = spechls.task "task"(%in = %x_init): (i32) -> !spechls.struct<"commit_type_0" {"enable": i1, "commit_val_0": i32 }> {
    %true1 = hw.constant 1 : i1
    %mu = spechls.mu<"x">(%in, %gamma) : i32
    %cond = spechls.call @cond(%mu) {spechls.combDelay = 20.0: f64}: (i32) -> i1
    %fast = spechls.call @fast(%mu) : (i32) -> i32
    %slow = spechls.call @slow(%mu) {spechls.combDelay = 70.0: f64}: (i32) -> i32
    // CHECK: %gamma = spechls.gamma<"x">(%cond, %fast, %slow) {spechls.profilingId = 0 : i64, spechls.speculation = 1 : i32}: i1, i32
    %gamma = spechls.gamma<"x">(%cond, %fast, %slow) {spechls.profilingId=0}: i1, i32
    spechls.commit %true1, %gamma : i1, i32
  }
  %1 = spechls.field<"commit_val_0"> %0 : !spechls.struct<"commit_type_0" {"enable": i1, "commit_val_0": i32 }>
  spechls.exit if %true0 with %1 : i32
}