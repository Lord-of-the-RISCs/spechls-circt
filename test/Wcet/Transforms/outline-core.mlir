// RUN: spechls-opt -split-input-file --outline-core %s | spechls-opt |FileCheck %s

//CHECK: wcet.core @core_test(%arg0: i32 {wcet.instrNb = 0 : i32}, %arg1: i32, %arg2: i32 {wcet.nbPred = 0 : i32}, %arg3: i32 {wcet.nbPred = 1 : i32}) -> (i32, i32, i32)
//CHECK-NOT: spechls.delay
//CHECK-NOT: spechls.mu
//CHECK: wcet.commit

spechls.kernel @simpl(%in0 : i32, %in1 : !spechls.array<i32, 32>) -> i32 {
    %true = hw.constant 1 : i1
    %result_struct = spechls.task "test" () : ()  -> 
    !spechls.struct<"out_task" {"result" : i32 }> attributes {spechls.speculative} {
        %0 = wcet.init @x : i32
        %mu = spechls.mu<"x">(%0, %result): i32
        %1 = hw.constant 1 : i1
        %2 = wcet.init @y: i32
        %arr = spechls.call @getMem() : () -> !spechls.array<i32, 32>
        %3 = spechls.load %arr[%2 : i32] {wcet.fetch} : <i32, 32>
        %4 = spechls.delay %3 by 1 : i32
        %5 = spechls.delay %4 by 1 : i32
        %result = comb.add %mu, %5 : i32
        spechls.commit %result : i32
      }
    %result = spechls.field <"result"> %result_struct : !spechls.struct<"out_task" {"result" : i32 }>
      spechls.exit if %true with %result : i32 
  }
