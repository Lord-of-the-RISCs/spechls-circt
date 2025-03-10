add_compile_definitions(_YOSYS_)

add_mlir_library(SpecHLSTransforms
        Analysis/DataflowDominatorAnalysis.cpp
        Analysis/PostDataflowDominatorAnalysis.cpp
        Analysis/TransitiveClosure.cpp
        Common/OutliningUtils.cpp
        Common/UnrollInstr.cpp
        Common/LongestPath.cpp
        Simplifier/YosysOptimizer.cpp
        Simplifier/RTLILImporter.cpp
        Simplifier/FactorGammaInputs.cpp
        Simplifier/EliminateRedundantGammaInputs.cpp
        Simplifier/InlineOptimizedModules.cpp
        Simplifier/MergeGammas.cpp
        Simplifier/MergeLUTs.cpp
        SpecHLSToComb/SpecHLSToComb.cpp
        SpecHLSToComb/SpecHLSLUTToComb.cpp
        SpecHLSToComb/GammaConversion.cpp
        SpecHLSToComb/RollbackConversion.cpp
        SpecHLSToComb/LUTConversion.cpp
        SpecHLSToSeq/SpecHLSToSeq.cpp
        SpecHLSToSeq/DelayConversion.cpp
        Simplifier/GroupControlLogic.cpp
        Scheduling/MobilityPass.cpp
        Scheduling/ConfigurationExcluder.cpp
        Scheduling/Schedule.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSTransformsPassIncGen

        LINK_LIBS PUBLIC
        MLIRSpecHLS
)
target_compile_options(SpecHLSTransforms PUBLIC -fexceptions)
