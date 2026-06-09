// RUN: spechls-opt %s | spechls-opt | FileCheck %s --check-prefixes=COMMON,ROUNDTRIP
// RUN: spechls-opt --inline-optimized-func-body %s | FileCheck %s --check-prefixes=COMMON,INLINEBODY
// RUN: spechls-opt --inline-optimized-func-opt-body %s | FileCheck %s --check-prefixes=COMMON,INLINEOPTBODY

// COMMON-LABEL: @kernel
// COMMON-SAME: %[[arg0:[a-zA-Z0-9]+]]: i3
spechls.kernel @kernel(%a : i3) -> i3 {
  // COMMON: %[[true:.+]] = hw.constant true
  %true = hw.constant true
  // INLINEOPTBODY: %[[c2:.+]] = hw.constant 2 : i3
  // ROUNDTRIP: %[[func:.+]] = spechls.optimizedFunc (%[[arg0]] : i3) : i3 (%[[bodyArg:.+]]){
  %result = spechls.optimizedFunc (%a : i3) : i3  (%arg0){
    // ROUNDTRIP: spechls.yield %[[bodyArg]] : i3
    spechls.yield %arg0 : i3
    // ROUNDTRIP: }(%[[optBodyArg:.+]]){
  }(%arg0){
    // ROUNDTRIP: %[[cst:.+]] = hw.constant 2 : i3
    %cst = hw.constant 2 : i3
    // ROUNDTRIP: spechls.yield %[[cst]] : i3
    spechls.yield %cst : i3
    // ROUNDTRIP: }
  }
  // ROUNDTRIP: spechls.exit if %[[true]] with %[[func]] : i3
  // INLINEBODY: spechls.exit if %[[true]] with %[[arg0]] : i3
  // INLINEOPTBODY: spechls.exit if %[[true]] with %[[c2]] : i3
  spechls.exit if %true with %result : i3
}
