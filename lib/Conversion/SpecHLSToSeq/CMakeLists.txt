add_circt_conversion_library(SpecHLSToSeq
        SpecHLSToSeq.cpp

  DEPENDS
  SpecHLSConversionPassIncGen

  LINK_LIBS PUBLIC
  SpecHLS
)
