#######################################################################
#                                                                     #
# CMake script for configuration of incremental_fe library            #
#                                                                     #
# Usage: cmake -DCMAKE_INSTALL_PREFIX=/installation/path /source/path #
#                                                                     #
#######################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(incremental_fe VERSION 1.0.0)

#locate deal.II (this is only needed to provide with the testing facilities)
FIND_PACKAGE(deal.II 9.1.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#locate galerkin_tools
FIND_PACKAGE(galerkin_tools 1.0.0 QUIET HINTS $ENV{GALERKIN_TOOLS_DIR} ${GALERKIN_TOOLS_DIR})
IF(NOT ${galerkin_tools_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of galerkin_tools. ***\n\n"
    "You may want to either pass a flag -DGALERKIN_TOOLS_DIR=/path/to/galerkin_tools to cmake\n"
    "or set an environment variable \"GALERKIN_TOOLS_DIR\" that contains this path."
    )
ENDIF()

#define paths for installation of libraries and headers of library incremental_fe
INCLUDE(GNUInstallDirs)
SET(INCREMENTAL_FE_INSTALL_PATH_LIBRARIES ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
SET(INCREMENTAL_FE_INSTALL_PATH_HEADERS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

#set library names for debug and release versions of incremental_fe library, respectively
SET(INCREMENTAL_FE_TARGET_DEBUG ${PROJECT_NAME}.debug)
SET(INCREMENTAL_FE_TARGET_RELEASE ${PROJECT_NAME}.release)

#this does the actual definition of the target, link libraries, etc.
ADD_SUBDIRECTORY(source)

#get the properties for compiling, linking and headers (these are written into the config file)
GET_PROPERTY(INCREMENTAL_FE_LIBRARIES_DEBUG GLOBAL PROPERTY INCREMENTAL_FE_LIBRARIES_DEBUG)
GET_PROPERTY(INCREMENTAL_FE_LIBRARIES_RELEASE GLOBAL PROPERTY INCREMENTAL_FE_LIBRARIES_RELEASE)
GET_PROPERTY(INCREMENTAL_FE_INCLUDE_DIRECTORIES GLOBAL PROPERTY INCREMENTAL_FE_INCLUDE_DIRECTORIES)
GET_PROPERTY(INCREMENTAL_FE_COMPILE_FLAGS_DEBUG GLOBAL PROPERTY INCREMENTAL_FE_COMPILE_FLAGS_DEBUG)
GET_PROPERTY(INCREMENTAL_FE_COMPILE_FLAGS_RELEASE GLOBAL PROPERTY INCREMENTAL_FE_COMPILE_FLAGS_RELEASE)
GET_PROPERTY(INCREMENTAL_FE_COMPILE_DEFINITIONS_DEBUG GLOBAL PROPERTY INCREMENTAL_FE_COMPILE_DEFINITIONS_DEBUG)
GET_PROPERTY(INCREMENTAL_FE_COMPILE_DEFINITIONS_RELEASE GLOBAL PROPERTY INCREMENTAL_FE_COMPILE_DEFINITIONS_RELEASE)
GET_PROPERTY(INCREMENTAL_FE_LINK_FLAGS_DEBUG GLOBAL PROPERTY INCREMENTAL_FE_LINK_FLAGS_DEBUG)
GET_PROPERTY(INCREMENTAL_FE_LINK_FLAGS_RELEASE GLOBAL PROPERTY INCREMENTAL_FE_LINK_FLAGS_RELEASE)

#configure the tests
#make sure that the tests know of the headers of the incremental_fe library and are linked against the incremental_fe library
INCLUDE_DIRECTORIES(${INCREMENTAL_FE_INCLUDE_DIRECTORIES})
SET(TEST_LIBRARIES_DEBUG ${INCREMENTAL_FE_LIBRARIES_DEBUG})
SET(TEST_LIBRARIES_RELEASE ${INCREMENTAL_FE_LIBRARIES_RELEASE})
#set compile and link flags
SET(DEAL_II_CXX_FLAGS "")
SET(DEAL_II_USER_DEFINITIONS "")
SET(DEAL_II_LINKER_FLAGS "")
FOREACH(_build "debug" "release")
  STRING(TOUPPER ${_build} _build_toupper)
  SET(DEAL_II_CXX_FLAGS_${_build_toupper} ${INCREMENTAL_FE_COMPILE_FLAGS_${_build_toupper}})
  SET(DEAL_II_USER_DEFINITIONS_${_build_toupper} ${INCREMENTAL_FE_COMPILE_DEFINITIONS_${_build_toupper}})
  SET(DEAL_II_LINKER_FLAGS_${_build_toupper} ${INCREMENTAL_FE_LINK_FLAGS_${_build_toupper}})
ENDFOREACH(_build)
#build the tests for both, debug and release mode
SET(CMAKE_BUILD_TYPE "DebugRelease")
#enable testing
ENABLE_TESTING()
#this does the actual work
ADD_SUBDIRECTORY(tests)

#set up and install the configuration information of the library incremental_fe
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/config/Config.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/incremental_feConfig.cmake" @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/config/ConfigVersion.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/incremental_feConfigVersion.cmake" @ONLY)
INSTALL(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/incremental_feConfig.cmake DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_LIBRARIES}/cmake/incremental_fe/)
INSTALL(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/incremental_feConfigVersion.cmake DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_LIBRARIES}/cmake/incremental_fe/)
