// RUN: spechls-opt -split-input-file --outline-core %s | spechls-opt |FileCheck %s

//CHECK: wcet.core @core_test(%arg0: i32, %arg1: i1, %arg2: !spechls.array<i32, 32>, %arg3: i32 {wcet.instrs}, %arg4: i32, %arg5: i32 {wcet.pred = 0 : i32}) -> !spechls.struct<"out" { "delay_0" : i32, "delay_1" : i32, "result" : i32 }>
//CHECK-NOT: spechls.delay
//CHECK: wcet.commit

spechls.kernel @simpl(%in0 : i32, %in1 : !spechls.array<i32, 32>) -> i32 {
    %true = hw.constant 1 : i1
    %result_struct = spechls.task "test" (%0 = %in0, %1 = %true, %arr = %in1) : (i32, i1, !spechls.array<i32, 32>)  -> 
    !spechls.struct<"out_task" {"result" : i32 }> attributes {spechls.speculative} {
        %2 = comb.add %0, %0 : i32
        %3 = spechls.load %arr[%2 : i32] {wcet.fetch} : <i32, 32>
        %4 = spechls.delay %3 by 1 : i32
        %5 = spechls.delay %4 by 1 : i32
        spechls.commit %5 : i32
      }
    %result = spechls.field <"result"> %result_struct : !spechls.struct<"out_task" {"result" : i32 }>
      spechls.exit if %true with %result : i32 
  }
