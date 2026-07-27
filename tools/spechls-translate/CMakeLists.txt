add_executable(spechls-translate spechls-translate.cpp)

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

target_link_libraries(spechls-translate PRIVATE
    SpecHLS
    SpecHLSTargetCpp
    ${LIBS}
)
install(TARGETS spechls-translate DESTINATION bin)
set_property(TARGET spechls-translate PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
