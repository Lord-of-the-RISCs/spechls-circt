add_executable(wcet-translate wcet-translate.cpp)

if(${SPECHLS_LINK_DYLIB})
  set(LIBS MLIR CIRCT)
else()
  set(LIBS
    MLIRIR
    MLIRAnalysis
    MLIRArithDialect
    CIRCTHW
    CIRCTComb
  )
endif()

target_link_libraries(wcet-translate PRIVATE
    SpecHLS
    Wcet
    SpecHLSTargetCpp
    SpecHLSExportWcetCpp
    ${LIBS}
)
install(TARGETS wcet-translate DESTINATION bin)
set_property(TARGET wcet-translate PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
