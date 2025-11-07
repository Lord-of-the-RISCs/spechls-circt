add_circt_translation_library(SpecHLSTargetUclid
    TranslateToUclid.cpp
    LINK_LIBS PUBLIC
    SpecHLS
    MLIRTranslateLib
)
