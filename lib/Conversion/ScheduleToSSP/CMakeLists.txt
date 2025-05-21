add_circt_conversion_library(ScheduleToSSP
  ScheduleToSSP.cpp

  DEPENDS
  SPECHLSConversionPassIncGen

  LINK_LIBS PUBLIC
  Schedule
)
