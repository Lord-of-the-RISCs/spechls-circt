add_executable(spechls-translate spechls-translate.cpp)

target_link_libraries(spechls-translate PRIVATE
    SpecHLS
    SpecHLSTargetCpp
    SpecHLSTargetUclid
    MLIR
    CIRCT
)
install(TARGETS spechls-translate DESTINATION bin)
set_property(TARGET spechls-translate PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
