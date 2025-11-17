//RUN: spechls-opt -split-input-file --insert-dummy %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @simple
spechls.kernel @simple(%in0 : i32, %in1 : i32) -> i32 {
  //CHECK: %0:2 = wcet.dummy %arg0, %arg1 : i32, i32 -> i32, i32
   %true = hw.constant 1 : i1 
   %pack = spechls.pack %in0, %in1 : (i32, i32) -> !spechls.struct< "packed" { "in0" : i32, "in1" : i32 } >
   %unpack:2 = spechls.unpack %pack : !spechls.struct< "packed" { "in0" : i32, "in1" : i32 }>
   %result = comb.add %unpack#0, %unpack#1 : i32
   spechls.exit if %true with %result : i32
}
