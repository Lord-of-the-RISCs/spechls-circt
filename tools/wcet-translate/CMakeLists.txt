add_executable(wcet-translate wcet-translate.cpp)

target_link_libraries(wcet-translate PRIVATE
    SpecHLS
    Wcet
    SpecHLSTargetCpp
    SpecHLSExportWcetCpp
    MLIR
    CIRCT
)
install(TARGETS wcet-translate DESTINATION bin)
set_property(TARGET wcet-translate PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
