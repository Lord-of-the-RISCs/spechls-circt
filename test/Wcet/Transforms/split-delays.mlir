// RUN: spechls-opt -split-input-file --split-delays %s | spechls-opt | FileCheck %s

//CHECK-LABEL: @simple
spechls.kernel @simple(%in1 : i32, %in2 : i1) -> i32 {
  //CHECK: %0 = spechls.delay %arg0 by 1 if %true {} : i32
  //CHECK: %1 = spechls.delay %0 by 1 if %true {} : i32
  //CHECK: %2 = spechls.delay %1 by 1 if %true {} : i32
  %true = hw.constant 1 : i1
  %d0 = spechls.delay %in1 by 3 if %true : i32
  spechls.exit if %in2 with %d0 : i32
}

//CHECK-LABEL: @rollbackable
spechls.kernel @rollbackable(%in1 : i32, %in2 : i1, %rollback : i1) -> i32 {
  //CHECK: %0 = spechls.rollbackableDelay %arg0 by 1 rollback %arg2 at 2 [1, 2] if %true {} : i32
  //CHECK: %1 = spechls.rollbackableDelay %0 by 1 rollback %arg2 at 1 [1, 2] if %true {} : i32
  //CHECK: %2 = spechls.rollbackableDelay %1 by 1 rollback %arg2 at 0 [1, 2] if %true {} : i32
  %true = hw.constant 1 : i1
  %d0 = spechls.rollbackableDelay %in1 by 3 rollback %rollback at 0 [1,2] if %true : i32
  spechls.exit if %in2 with %d0 : i32
}

//CHECK-LABEL: @cancel
spechls.kernel @cancel(%in1 : i1, %in2 : i1, %cancel : i1) -> i1 {
  //CHECK: %0 = spechls.cancellableDelay %arg0 by 1 cancel %arg2 at 2 if %true {} : i1
  //CHECK: %1 = spechls.cancellableDelay %0 by 1 cancel %arg2 at 1 if %true {} : i1
  //CHECK: %2 = spechls.cancellableDelay %1 by 1 cancel %arg2 at 0 if %true {} : i1
  %true = hw.constant 1 : i1
  %d0 = spechls.cancellableDelay %in1 by 3 cancel %cancel at 0 if %true : i1
  spechls.exit if %in2 with %d0 : i1
}

//CHECK-LABEL: @doubleUse
spechls.kernel @doubleUse(%in1 : i32, %in2 : i1) -> i32 {
  //CHECK: %0 = spechls.delay %arg0 by 1 if %true {} : i32
  //CHECK: %1 = spechls.delay %0 by 1 if %true {} : i32
  //CHECK: %2 = spechls.delay %1 by 1 if %true {} : i32
  //CHECK: %3 = spechls.delay %2 by 1 {} : i32
  //CHECK: %4 = spechls.delay %3 by 1 {} : i32
  %true = hw.constant 1 : i1
  %d0 = spechls.delay %in1 by 3 if %true : i32
  %d1 = spechls.delay %d0 by 2 : i32
  %result = comb.add %d0, %d1 : i32
  spechls.exit if %in2 with %result : i32
}
