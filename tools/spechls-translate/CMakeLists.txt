add_executable(spechls-translate spechls-translate.cpp)

target_link_libraries(spechls-translate PRIVATE
    SpecHLS
    SpecHLSTargetCpp
    MLIR
    CIRCT
)
# Match the locally built MLIR/CIRCT libraries, which were compiled without RTTI.
target_compile_options(spechls-translate PRIVATE -fno-rtti)
install(TARGETS spechls-translate DESTINATION bin)
set_property(TARGET spechls-translate PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
