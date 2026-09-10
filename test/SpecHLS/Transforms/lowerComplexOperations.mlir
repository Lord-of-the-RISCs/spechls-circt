// RUN: spechls-opt -split-input-file --lower-complex-operation %s | spechls-opt | FileCheck %s

// CHECK: spechls.kernel @mul1(%arg0: i12) -> i12
spechls.kernel @mul1(%arg0: i12) -> i12 {
  %true = hw.constant true
  %4 = hw.constant 4 : i12
  %0 = comb.mul %arg0, %4 : i12
  // CHECK: %[[c:.*]] = hw.constant 2 : i12
  // CHECK: %[[s:.*]] = comb.shl %arg0, %[[c]] : i12
  // CHECK: spechls.exit if %true with %[[s]] : i12
  spechls.exit if %true with %0 : i12 
}

// CHECK: spechls.kernel @mul2(%arg0: i3, %arg1: i3) -> i3
spechls.kernel @mul2(%arg0: i3, %arg1: i3) -> i3 {
  // CHECK-DAG: %[[true:.*]] = hw.constant true
  // CHECK-DAG: %[[c0:.*]] = hw.constant 0 : i3
  // CHECK-DAG: %[[c1:.*]] = hw.constant 1 : i3
  // CHECK-DAG: %[[c2:.*]] = hw.constant 2 : i3
  // CHECK-DAG: %[[b0:.*]] = comb.extract %arg1 from 0 : (i3) -> i1
  // CHECK-DAG: %[[b1:.*]] = comb.extract %arg1 from 1 : (i3) -> i1
  // CHECK-DAG: %[[b2:.*]] = comb.extract %arg1 from 2 : (i3) -> i1
  // CHECK-DAG: %[[s1:.*]] = comb.shl %arg0, %[[c1]] : i3
  // CHECK-DAG: %[[s2:.*]] = comb.shl %arg0, %[[c2]] : i3
  // CHECK-DAG: %[[m0:.*]] = comb.mux %[[b0]], %arg0, %[[c0]] : i3
  // CHECK-DAG: %[[m1:.*]] = comb.mux %[[b1]], %[[s1]], %[[c0]] : i3
  // CHECK-DAG: %[[m2:.*]] = comb.mux %[[b2]], %[[s2]], %[[c0]] : i3
  // CHECK-DAG: %[[a1:.*]] = comb.add %[[m0]], %[[m1]] : i3
  // CHECK-DAG: %[[a2:.*]] = comb.add %[[a1]], %[[m2]] : i3
  // CHECK: spechls.exit if %[[true]] with %[[a2]] : i3
  %true = hw.constant true
  %0 = comb.mul %arg0, %arg1 : i3
  spechls.exit if %true with %0 : i3 
}

// CHECK: spechls.kernel @div1(%arg0: i12) -> i12
spechls.kernel @div1(%arg0: i12) -> i12 {
  %true = hw.constant true
  %4 = hw.constant 4 : i12
  %0 = comb.divu %arg0, %4 : i12

  // CHECK-DAG: %[[true:.*]] = hw.constant true
  // CHECK-DAG: %[[c2:.*]] = hw.constant 2 : i12
  // CHECK-DAG: %[[div:.*]] = comb.shru %arg0, %[[c2]] : i12
  // CHECK: spechls.exit if %[[true]] with %[[div]] : i12 
  spechls.exit if %true with %0 : i12 
}

// CHECK: spechls.kernel @div2(%arg0: i3, %arg1: i3) -> i3
spechls.kernel @div2(%arg0: i3, %arg1: i3) -> i3 {
  %true = hw.constant true
  %0 = comb.divu %arg0, %arg1 : i3
  // CHECK-DAG: %[[true:.*]] = hw.constant true
  // CHECK-DAG: %[[c0_2b:.*]] = hw.constant 0 : i2
  // CHECK-DAG: %[[c0:.*]] = hw.constant 0 : i3
  // CHECK-DAG: %[[c1:.*]] = hw.constant 1 : i3
  // CHECK-DAG: %[[c2:.*]] = hw.constant 2 : i3
  // CHECK-DAG: %[[c4:.*]] = hw.constant -4 : i3
  // CHECK-DAG: %[[in_b0:.*]] = comb.extract %arg0 from 0 : (i3) -> i1
  // CHECK-DAG: %[[in_b1:.*]] = comb.extract %arg0 from 1 : (i3) -> i1
  // CHECK-DAG: %[[in_b2:.*]] = comb.extract %arg0 from 2 : (i3) -> i1
  // CHECK-DAG: %[[concat_0:.*]] = comb.concat %[[c0_2b]], %[[in_b2]] : i2, i1
  // CHECK-DAG: %[[comp_0:.*]] = comb.icmp uge %[[concat_0]], %arg1 : i3
  // CHECK-DAG: %[[sub_0:.*]] = comb.sub %[[concat_0]], %arg1 : i3
  // CHECK-DAG: %[[mux_0:.*]] = comb.mux %[[comp_0]], %[[sub_0]], %[[concat_0]] : i3
  // CHECK-DAG: %[[mux_1:.*]] = comb.mux %[[comp_0]], %[[c4]], %[[c0]] : i3
  // CHECK-DAG: %[[e_mux_0:.*]] = comb.extract %[[mux_0]] from 0 : (i3) -> i2
  // CHECK-DAG: %[[concat_1:.*]] = comb.concat %[[e_mux_0]], %[[in_b1]] : i2, i1
  // CHECK-DAG: %[[comp_1:.*]] = comb.icmp uge %[[concat_1]], %arg1 : i3
  // CHECK-DAG: %[[sub_1:.*]] = comb.sub %[[concat_1]], %arg1 : i3
  // CHECK-DAG: %[[or_0:.*]] = comb.or %[[mux_1]], %[[c2]] : i3
  // CHECK-DAG: %[[mux_2:.*]] = comb.mux %[[comp_1]], %[[sub_1]], %[[concat_1]] : i3
  // CHECK-DAG: %[[mux_3:.*]] = comb.mux %[[comp_1]], %[[or_0]], %[[mux_1]] : i3
  // CHECK-DAG: %[[e_mux_2:.*]] = comb.extract %[[mux_2]] from 0 : (i3) -> i2
  // CHECK-DAG: %[[concat_2:.*]] = comb.concat %[[e_mux_2]], %[[in_b0]] : i2, i1
  // CHECK-DAG: %[[comp_2:.*]] = comb.icmp uge %[[concat_2]], %arg1 : i3
  // CHECK-DAG: %[[or_1:.*]] = comb.or %[[mux_3]], %[[c1]] : i3
  // CHECK-DAG: %[[div:.*]] = comb.mux %[[comp_2]], %[[or_1]], %[[mux_3]] : i3
  // CHECK: spechls.exit if %[[true]] with %[[div]] : i3
  spechls.exit if %true with %0 : i3 
}