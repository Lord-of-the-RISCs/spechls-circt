add_circt_conversion_library(ScheduleToSSP
  ScheduleToSSP.cpp

  DEPENDS
  ScheduleConversionPassIncGen

  LINK_LIBS PUBLIC
  Schedule
)
