// RUN: spechls-opt %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @kernel
spechls.kernel @kernel(%in1 : i32, %in2 : i1, %rollback : i1) -> i32 {
  %true = hw.constant 1 : i1
  %false = hw.constant false
  // CHECK: spechls.rollbackableDelay %arg0 by 1 rollback %arg2 at 0 %false [] {} : i32
  // CHECK: spechls.rollbackableDelay %0 by 2 rollback %arg2 at 1 %false [] if %arg1 {} : i32
  // CHECK: spechls.rollbackableDelay %1 by 1 rollback %arg2 at 0 %false [] init %arg0 {} : i32
  // CHECK: spechls.rollbackableDelay %2 by 4 rollback %arg2 at 3 %false [] if %arg1 init %arg0 {} : i32
  // CHECK: spechls.rollbackableDelay %2 by 4 rollback %arg2 at 2 %false [] if %arg1 init %4 {} : i32
  %d0 = spechls.rollbackableDelay %in1 by 1 rollback %rollback at 0 %false [] : i32
  %d1 = spechls.rollbackableDelay %d0 by 2 rollback %rollback at 1 %false []  if %in2 : i32
  %d2 = spechls.rollbackableDelay %d1 by 1 rollback %rollback at 0 %false []  init %in1 : i32
  %d3 = spechls.rollbackableDelay %d2 by 4 rollback %rollback at 3 %false []  if %in2 init %in1 : i32
  %d4 = spechls.rollbackableDelay %d2 by 4 rollback %rollback at 2 %false []  if %in2 init %d4 : i32
  spechls.exit if %in2 with %d4 : i32
}
