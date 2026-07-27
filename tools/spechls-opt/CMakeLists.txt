add_executable(spechls-opt spechls-opt.cpp)

if(${SPECHLS_LINK_DYLIB})
  set(LIBS MLIR CIRCT)
else()
  set(LIBS
    MLIRIR
    MLIRAnalysis
    MLIRArithDialect
    MLIRMlirOptMain
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

target_link_libraries(spechls-opt PRIVATE
    SpecHLS
    SpecHLSTransforms
    SpecHLSToHW
    Schedule
    Wcet
    ScheduleTransforms
    WcetTransforms
    ScheduleToSSP
    ${LIBS}

    Yosys::Yosys
)
install(TARGETS spechls-opt DESTINATION bin)
set_property(TARGET spechls-opt PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
