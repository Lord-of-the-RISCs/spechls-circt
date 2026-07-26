add_mlir_pdll_library(SpecHLSSimplifyGammasPatternsIncGen
    SimplifyGammas.pdll
    SimplifyGammas.h.inc
)

add_mlir_pdll_library(SpecHLSSimplifyLUTsPatternsIncGen
    SimplifyLUTs.pdll
    SimplifyLUTs.h.inc
)

add_mlir_pdll_library(SpecHLSRemoveIntraRAWPatternsIncGen
    RemoveIntraRAW.pdll
    RemoveIntraRAW.h.inc
)

add_mlir_pdll_library(SpecHLSExposeMemorySpeculationPatternsIncGen
    ExposeMemorySpeculation.pdll
    ExposeMemorySpeculation.h.inc
)

add_mlir_pdll_library(SpecHLSInferLSQPatternsIncGen
    InferLSQ.pdll
    InferLSQ.h.inc
)

add_circt_dialect_library(SpecHLSTransforms
    Outlining.cpp
    YosysSetup.cpp
    SimplifyLUTs.cpp
    SimplifyGammas.cpp
    OptimizeControlLogic.cpp
    RemoveIntraRAW.cpp
    ExposeMemorySpeculation.cpp
    SpecHLSTopoSort.cpp
    SimplifyComb.cpp
    InferLSQ.cpp
    SpeculationExplorationPass.cpp
    LowerDelays.cpp
    AnnotateTiming.cpp
    LowerComplexOperations.cpp
    ExposeControlFlowSpeculation.cpp
    InferSpeculationFSM.cpp
    LinearizeTask.cpp
    MergeTasks.cpp
    SplitWithSyncPass.cpp
    AffectProfilingID.cpp
    InlineOptimizedFuncPass.cpp

    DEPENDS
    SpecHLSTransformsIncGen
    SpecHLSSimplifyLUTsPatternsIncGen
    SpecHLSSimplifyGammasPatternsIncGen
    SpecHLSRemoveIntraRAWPatternsIncGen
    SpecHLSExposeMemorySpeculationPatternsIncGen
    SpecHLSInferLSQPatternsIncGen

    LINK_LIBS PUBLIC
    SpecHLSAnalysis
)
target_include_directories(SpecHLSTransforms
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Yosys headers use exceptions; keep that requirement limited to translation
# units that include them instead of changing LLVM-wide compile settings.
set_source_files_properties(
    YosysSetup.cpp
    OptimizeControlLogic.cpp
    PROPERTIES
    COMPILE_OPTIONS "-std=gnu++20;-fexceptions"
)
