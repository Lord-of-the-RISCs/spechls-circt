add_executable(spechls-reduce spechls-reduce.cpp)

if(${SPECHLS_LINK_DYLIB})
  set(LIBS MLIR CIRCT)
else()
  set(LIBS
    MLIRIR
    MLIRAnalysis
    MLIRArithDialect
    MLIRReduceLib
    CIRCTHW
    CIRCTComb
    CIRCTSV
    CIRCTSeq
    CIRCTSSP
    CIRCTSSPTransforms
    CIRCTMoore
    CIRCTMooreTransforms
    CIRCTSim
    CIRCTDebug
    CIRCTSynth
    CIRCTVerif
    CIRCTLLHD
    CIRCTLLHDTransforms
    CIRCTImportVerilog
    CIRCTExportVerilog
    CIRCTMooreToCore
  )
endif()

target_link_libraries(spechls-reduce PRIVATE
    SpecHLS
    Schedule
    ScheduleTransforms
    ScheduleToSSP
    ${LIBS}
)
install(TARGETS spechls-reduce DESTINATION bin)
set_property(TARGET spechls-reduce PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
