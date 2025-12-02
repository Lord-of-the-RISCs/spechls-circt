add_circt_library(SpecHLSAnalysis
  ConfigurationExcluderAnalysis.cpp
  MobilityAnalysis.cpp
  SchedulingAnalysis.cpp
  SpeculationExplorationAnalysis.cpp

  LINK_LIBS PUBLIC
  CIRCT
  SpecHLS
)
