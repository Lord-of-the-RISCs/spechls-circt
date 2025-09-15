add_circt_conversion_library(SpecHLSToHW
  SpecHLSToHW.cpp

  DEPENDS
  SpecHLSConversionPassIncGen

  LINK_LIBS PUBLIC
  SpecHLS
)
