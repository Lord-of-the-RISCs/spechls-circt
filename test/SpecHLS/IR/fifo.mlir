// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK-LABEL: @kernel
spechls.kernel @kernel(%in1 : !spechls.struct<"s" { "val0" : i1, "val1" : i32 }>) -> !spechls.struct<"s" { "val0" : i1, "val1" : i32 }> {
  %true = hw.constant 1 : i1
  // CHECK: spechls.fifo<4> %arg0 : <"s" { "val0" : i1, "val1" : i32 }>
  %0 = spechls.fifo<4> %in1 : !spechls.struct<"s" { "val0" : i1, "val1" : i32 }>
  spechls.exit if %true with %0 : !spechls.struct<"s" { "val0" : i1, "val1" : i32 }>
}
