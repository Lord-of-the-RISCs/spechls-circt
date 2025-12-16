"builtin.module"() ({
  "spechls.kernel"() <{function_type = (i32, i1) -> i32, sym_name = "kernel"}> ({
  ^bb0(%arg0: i32, %arg1: i1):
    %0 = "hw.constant"() <{value = true}> : () -> i1
    %1 = "spechls.delay"(%arg0) <{depth = 1 : ui32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> i32
    %2 = "spechls.delay"(%1, %arg1) <{depth = 2 : ui32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (i32, i1) -> i32
    %3 = "spechls.delay"(%2, %arg0) <{depth = 1 : ui32, operandSegmentSizes = array<i32: 1, 0, 1>}> : (i32, i32) -> i32
    %4 = "spechls.delay"(%3, %arg1, %arg0) <{depth = 4 : ui32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (i32, i1, i32) -> i32
    %5 = "spechls.delay"(%3, %arg1, %5) <{depth = 4 : ui32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (i32, i1, i32) -> i32
    "spechls.exit"(%arg1, %5) : (i1, i32) -> ()
  }) : () -> ()
}) : () -> ()

