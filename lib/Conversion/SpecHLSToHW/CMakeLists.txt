add_circt_conversion_library(SpecHLSToHW
        common.cpp
        SpecHLSToHW.cpp
  SpecHLSToSeq.cpp

  DEPENDS
  SpecHLSConversionPassIncGen

  LINK_LIBS PUBLIC
  SpecHLS
)
