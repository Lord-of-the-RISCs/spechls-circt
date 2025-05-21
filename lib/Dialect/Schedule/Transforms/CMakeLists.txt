add_circt_dialect_library(ScheduleTransforms
    CheckSchedule.cpp
    ConfigurationExcluder.cpp
    Mobility.cpp
    Schedule.cpp

    DEPENDS
    ScheduleTransformsIncGen
)
