add_circt_library(WcetAnalysis
  GraphWcetAnalysis.cpp
  LinearWcetAnalysis.cpp

  LINK_LIBS PUBLIC
  CIRCT
  SpecHLS
  Wcet
  WcetUtils
)
