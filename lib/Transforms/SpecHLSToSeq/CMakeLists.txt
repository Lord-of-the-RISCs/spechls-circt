add_mlir_library(SpecHLSToSeq
        SpecHLSToSeq.cpp
        DelayConversion.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSConversionIncGen

        LINK_LIBS PUBLIC
        CIRCT
)
