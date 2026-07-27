add_library(heptaneapi SHARED
  Heptane.cpp
)

if(${SPECHLS_LINK_DYLIB})
  set(LIBS MLIR CIRCT)
else()
  set(LIBS
    MLIRIR
    CIRCTHW
    CIRCTCAPIComb
    CIRCTCAPIHW
    CIRCTCAPIDebug
    CIRCTCAPIFIRRTL
    CIRCTCAPIFSM
    CIRCTCAPISynth
    CIRCTCAPISim
    CIRCTCAPIVerif
    CIRCTCAPISV
    CIRCTCAPIArc
    CIRCTCAPILTL
    CIRCTCAPILLHD
    CIRCTCAPIMoore
    CIRCTCAPISeq
    CIRCTCAPISSP
    CIRCTCAPIHWArith
    CIRCTImportVerilog
    CIRCTExportVerilog
    CIRCTMooreToCore
    CIRCTLLHDTransforms
    CIRCTMooreTransforms
  )
endif()

target_link_libraries(heptaneapi PUBLIC
    ScheduleCAPI
    SpecHLSCAPI
    WcetCAPI
    ScheduleToSSP
    SpecHLSTransforms
    SpecHLSToHW
    MLIR-C
    SpecHLS
    SpecHLSToHW
    Schedule
    ScheduleTransforms
    Wcet
    WcetTransforms
    WcetAnalysis
    WcetUtils
    Yosys::Yosys
    ${LIBS}
)

install(TARGETS heptaneapi)
set_property(TARGET heptaneapi PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
