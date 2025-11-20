//RUN: spechls-opt --inline-core %s | spechls-opt | FileCheck %s


wcet.core @mult(%op1 : i32, %op2 : i32) -> i32 {
    %result = comb.mul %op1, %op2 : i32
    wcet.commit %result : i32
  }

wcet.core @adder(%op1 : i32, %op2 : i32) -> i32 {
  %mul = wcet.core_instance @mult(%op1, %op2) : (i32, i32) -> i32
  %result = comb.add %op1, %mul : i32
    wcet.commit %result : i32
}

//CHECK-LABEL:  @simple
spechls.kernel @simple(%in0 : i32, %in1 : i32) -> i32 {
    %true = hw.constant 1 : i1
    // CHECK: comb.mul
    // CHECK: comb.add
    %result = wcet.core_instance @adder(%in0, %in1) : (i32, i32) -> i32
    spechls.exit if %true with %result : i32 
  }
