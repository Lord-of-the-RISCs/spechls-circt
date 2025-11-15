// RUN: spechls-opt --spechls-to-hw %s | FileCheck %s

// CHECK-LABEL: @task
spechls.kernel @task_comb(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %task1.result = hw.instance "task2" @task1(arg0: %arg0: i32) -> (result: i32)
  %0 = spechls.task "task2"(%arg1 = %in1) : (i32) -> i32 {
    %true = hw.constant 1 : i1
    // CHECK: comb.mux
    spechls.commit %true, %arg1 : i32
  }
  // CHECK: hw.output %task1.result : i32
  spechls.exit if %in2 with %0 : i32
}

// CHECK-LABEL: @task
spechls.kernel @task_seq(%in1 : i32, %in2 : i1) -> i32 {
  // CHECK: %task1.result = hw.instance "task1" @task1(arg0: %arg0: i32) -> (result: i32)
  %0 = spechls.task "task1"(%arg1 = %in1) : (i32) -> i32 {
    %true = hw.constant 1 : i1
    %m = spechls.mu<"x">(%arg1, %arg1) : i32
    // CHECK: comb.mux
    spechls.commit %true, %arg1 : i32
  }
  // CHECK: hw.output %task1.result : i32
  spechls.exit if %in2 with %0 : i32
}


//---

// CHECK-LABEL: @gamma
spechls.kernel @gamma(%cond: i1, %x: i32, %y: i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: comb.mux %arg0, %arg2, %arg1 : i32
  %0 = spechls.gamma<"x">(%cond, %x, %y) : i1, i32
  spechls.exit if %true with %0 : i32
}

spechls.kernel @array_raw(%init :!spechls.array<i32, 4>, %index: i32, %value: i32) -> i32 {
  %true = hw.constant true
    %mu = spechls.mu<"x">(%1, %init) : !spechls.array<i32, 4>
    %1 = spechls.alpha %mu[%index: i32], %value if %true : !spechls.array<i32, 4>
    %0 = spechls.load %1[%index : i32] : <i32, 4>
     spechls.exit if %true with %0 : i32
  }
//---

// CHECK-LABEL: @nary_gamma
spechls.kernel @nary_gamma(%cond: i2, %x: i32, %y: i32, %z: i32) -> i32 {
  %true = hw.constant 1 : i1
  // CHECK: %[[array:.+]] = hw.array_create %arg1, %arg2, %arg3 : i32
  // CHECK: hw.array_get %[[array]][%arg0] : !hw.array<3xi32>, i2
  %0 = spechls.gamma<"x">(%cond, %x, %y, %z) : i2, i32
  spechls.exit if %true with %0 : i32
}


