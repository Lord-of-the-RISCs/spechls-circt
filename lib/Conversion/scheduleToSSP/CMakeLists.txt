add_circt_conversion_library(SPECHLSscheduleToSSP
  scheduleToSSP.cpp

  DEPENDS
  SPECHLSConversionPassIncGen

  LINK_LIBS PUBLIC
  Schedule
)