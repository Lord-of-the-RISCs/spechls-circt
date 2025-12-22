add_circt_library(WcetUtils
  WcetUtils.cpp

  LINK_LIBS PUBLIC
  CIRCT
  SpecHLS
  Wcet
)
