add_executable(spechls-lsp spechls-lsp.cpp)

if(${SPECHLS_LINK_DYLIB})
  set(LIBS MLIR CIRCT)
else()
  set(LIBS
    MLIRIR
    MLIRAnalysis
    MLIRArithDialect
    MLIRLspServerLib
    MLIRRegisterAllDialects
    MLIRSMT
    CIRCTHW
    CIRCTComb
    CIRCTSV
    CIRCTSeq
    CIRCTSSP
    CIRCTMoore
    CIRCTSim
    CIRCTDebug
    CIRCTSynth
    CIRCTVerif
    CIRCTLLHD
    CIRCTHandshake
    CIRCTKanagawa
    CIRCTDatapath
    CIRCTCalyx
    CIRCTSystemC
    CIRCTPipelineOps
    CIRCTLoopSchedule
    CIRCTOM
    CIRCTHWArith
    CIRCTArc
    CIRCTRTGDialect
    CIRCTFIRRTL
    CIRCTInteropDialect
  )
endif()

target_link_libraries(spechls-lsp PRIVATE
    SpecHLS
    Schedule
    Wcet
    ${LIBS}
)
install(TARGETS spechls-lsp DESTINATION bin)
set_property(TARGET spechls-lsp PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
