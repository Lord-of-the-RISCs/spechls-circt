add_circt_dialect_library(ScheduleTransforms
    ConfigurationExcluder.cpp
    Mobility.cpp
    Schedule.cpp
    ExtractCriticalPath.cpp

    DEPENDS
    ScheduleTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTSSP
    CIRCTSSPTransforms
)
