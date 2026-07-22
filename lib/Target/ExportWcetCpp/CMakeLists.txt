add_circt_translation_library(SpecHLSExportWcetCpp
  ExportWcetCpp.cpp

  LINK_LIBS PUBLIC

  MLIRTranslateLib


  SpecHLS
  Wcet
)
