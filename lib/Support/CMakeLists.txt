add_circt_library(SpecHLSSupport
  TransitiveClosure.cpp

  LINK_LIBS PUBLIC
  MLIR
  SpecHLS
)
