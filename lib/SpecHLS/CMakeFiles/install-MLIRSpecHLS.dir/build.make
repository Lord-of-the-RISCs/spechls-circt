# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Applications/CLion 2.app/Contents/bin/cmake/mac/bin/cmake"

# The command to remove a file.
RM = "/Applications/CLion 2.app/Contents/bin/cmake/mac/bin/cmake" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template

# Utility rule file for install-MLIRSpecHLS.

# Include any custom commands dependencies for this target.
include lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/progress.make

lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS:
	cd /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/lib/SpecHLS && "/Applications/CLion 2.app/Contents/bin/cmake/mac/bin/cmake" -DCMAKE_INSTALL_COMPONENT="MLIRSpecHLS" -P /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/cmake_install.cmake

install-MLIRSpecHLS: lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS
install-MLIRSpecHLS: lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/build.make
.PHONY : install-MLIRSpecHLS

# Rule to build all files generated by this target.
lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/build: install-MLIRSpecHLS
.PHONY : lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/build

lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/clean:
	cd /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/lib/SpecHLS && $(CMAKE_COMMAND) -P CMakeFiles/install-MLIRSpecHLS.dir/cmake_clean.cmake
.PHONY : lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/clean

lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/depend:
	cd /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/lib/SpecHLS /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/lib/SpecHLS /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/SpecHLS/CMakeFiles/install-MLIRSpecHLS.dir/depend

