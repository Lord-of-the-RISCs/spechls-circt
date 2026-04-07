add_circt_library(SpecHLSAnalysis
  ConfigurationExcluderAnalysis.cpp
  MobilityAnalysis.cpp
  SchedulingAnalysis.cpp
  SpeculationExplorationAnalysis.cpp
  OperationDelayAnalysis.cpp

  LINK_LIBS PUBLIC
  SpecHLS
)
