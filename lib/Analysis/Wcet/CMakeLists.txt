add_circt_library(WcetAnalysis
  GraphWcetAnalysis.cpp
  LinearWcetAnalysis.cpp
  PenaltyGraphAnalysis.cpp

  LINK_LIBS PUBLIC
  CIRCT
  SpecHLS
  Wcet
  WcetUtils
)
