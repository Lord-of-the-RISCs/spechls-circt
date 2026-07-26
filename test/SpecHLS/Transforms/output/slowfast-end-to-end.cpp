#include <ap_int.h>
#include <io_printf.h>

#include "spechls_support.h"

static void fsm_fsm_x0_trigger(unsigned int state, bool input1, unsigned int &nextState, bool &output0, bool &output1, unsigned int &output2, unsigned int &output3, unsigned int &output4, bool &output5, unsigned int &output6, unsigned int &output7, unsigned int &output8, unsigned int &output9) {
  output0 = {};
  output1 = {};
  output2 = {};
  output3 = {};
  output4 = {};
  output5 = {};
  output6 = {};
  output7 = {};
  output8 = {};
  output9 = {};
  switch (state) {
    case 0:
      output0 = 0;
      output1 = 0;
      output2 = 0;
      output3 = 0;
      output4 = 0;
      output5 = 0;
      output6 = 0;
      output7 = 0;
      output8 = 0;
      output9 = 0;
      break;
    case 1:
      output0 = 0;
      output1 = 0;
      output2 = 0;
      output3 = 0;
      output4 = 0;
      output5 = 0;
      output6 = 0;
      output7 = 0;
      output8 = 0;
      output9 = 0;
      break;
    case 2:
      output0 = 1;
      output1 = 1;
      output2 = 0;
      output3 = 0;
      output4 = 0;
      output5 = 0;
      output6 = 0;
      output7 = 0;
      output8 = 0;
      output9 = 0;
      break;
    case 3:
      output0 = 0;
      output1 = 0;
      output2 = 0;
      output3 = 0;
      output4 = 0;
      output5 = 1;
      output6 = 1;
      output7 = 1;
      output8 = 0;
      output9 = 0;
      break;
    case 4:
      output0 = 0;
      output1 = 0;
      output2 = 0;
      output3 = 0;
      output4 = 0;
      output5 = 0;
      output6 = 0;
      output7 = 1;
      output8 = 0;
      output9 = 0;
      break;
    default: break;
  }
  nextState = state;
  switch (nextState) {
    case 0:
      if (true) nextState = 1;
      break;
    case 1:
      if (true) nextState = 2;
      break;
    case 2:
      if (input1 == 1) nextState = 3;
      else if (true) nextState = 2;
      break;
    case 3:
      if (true) nextState = 4;
      break;
    case 4:
      if (true) nextState = 2;
      break;
    default: nextState = 0; break;
  }
}

void slowfast(unsigned int &);


void slowfast(unsigned int &arg_1) {
  commit_type v_2{};
  unsigned int arg_3{};
  unsigned int fsm_x0State_4{};
  unsigned int fsm_fsm_x0_result_5{};
  bool fsm_fsm_x0_result_6{};
  bool fsm_fsm_x0_result_7{};
  unsigned int fsm_fsm_x0_result_8{};
  unsigned int fsm_fsm_x0_result_9{};
  unsigned int fsm_fsm_x0_result_10{};
  bool fsm_fsm_x0_result_11{};
  unsigned int fsm_fsm_x0_result_12{};
  unsigned int fsm_fsm_x0_result_13{};
  unsigned int fsm_fsm_x0_result_14{};
  unsigned int fsm_fsm_x0_result_15{};
  unsigned int rollback_16{};
  unsigned int rollback_16_buffer[3]{};
  unsigned int rollback_17{};
  unsigned int rollback_17_buffer[3]{};
  unsigned int delay_18{};
  unsigned int delay_18_buffer[2]{};
  bool eq_19{};
  bool eq_20{};
  unsigned int delay_21{};
  unsigned int delay_21_buffer[1]{};
  unsigned int delay_22{};
  unsigned int delay_22_buffer[1]{};
  bool delay_23{};
  bool delay_23_buffer[1]{};
  unsigned int x_24{};
  bool cond_25{};
  bool v_26{};
  unsigned int fast_27{};
  unsigned int slow_28{};
  unsigned int v_29{};
  unsigned int v_30{};
  unsigned int x_31{};
  arg_3 = arg_1;
  fsm_x0State_4 = (unsigned int){0};
  x_24 = arg_3;
  bool exit_ = false;
  while (!exit_) {
    ;
    delay_18 = delay_18_buffer[0];
    delay_21 = delay_21_buffer[0];
    delay_22 = delay_22_buffer[0];
    delay_23 = delay_23_buffer[0];
    fsm_fsm_x0_trigger(fsm_x0State_4, delay_23, fsm_fsm_x0_result_5, fsm_fsm_x0_result_6, fsm_fsm_x0_result_7, fsm_fsm_x0_result_8, fsm_fsm_x0_result_9, fsm_fsm_x0_result_10, fsm_fsm_x0_result_11, fsm_fsm_x0_result_12, fsm_fsm_x0_result_13, fsm_fsm_x0_result_14, fsm_fsm_x0_result_15);
    rollback_17 = rollback<unsigned int, 0, 2>(rollback_17_buffer, x_24, fsm_fsm_x0_result_8, fsm_fsm_x0_result_11);
    eq_19 = fsm_fsm_x0_result_15 == (unsigned int){0};
    eq_20 = fsm_fsm_x0_result_15 == (unsigned int){0};
    cond_25 = cond(rollback_17);
    fast_27 = fast(rollback_17);
    slow_28 = slow(rollback_17);
    x_31 = gamma<unsigned int, 0>(fsm_fsm_x0_result_13, fast_27, delay_22);
    rollback_16 = rollback<unsigned int, 0, 2>(rollback_16_buffer, x_31, fsm_fsm_x0_result_12, fsm_fsm_x0_result_11);
    delay_push<unsigned int, 2>(delay_18_buffer, rollback_16, (bool){true});
    delay_push<unsigned int, 1>(delay_21_buffer, slow_28, eq_19);
    delay_push<unsigned int, 1>(delay_22_buffer, delay_21, eq_20);
    delay_push<bool, 1>(delay_23_buffer, cond_25, (bool){true});
    v_2 = (commit_type){fsm_fsm_x0_result_7, delay_18};
    fsm_x0State_4 = fsm_fsm_x0_result_5;
    x_24 = rollback_16;
    exit_ = (bool){true};
  }
}
