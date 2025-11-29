add_executable(spechls-lsp spechls-lsp.cpp)

target_link_libraries(spechls-lsp PRIVATE
    SpecHLS
    Schedule
    Wcet
    MLIR
    CIRCT
)
install(TARGETS spechls-lsp DESTINATION bin)
set_property(TARGET spechls-lsp PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
