add_circt_translation_library(SpecHLSTargetCpp
    TranslateToCpp.cpp

    LINK_LIBS PUBLIC
    SpecHLS
    CIRCTComb
    CIRCTHW
    MLIRTranslateLib
)
