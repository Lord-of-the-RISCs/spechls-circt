add_circt_library(SpecHLSSupport
  DelayUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  )
