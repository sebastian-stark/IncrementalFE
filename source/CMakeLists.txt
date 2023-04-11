#source files from which library incremental_fe is compiled
SET(_target_src
    global_data_incremental_fe.cc
    fe_model.cc
    constraints.cc
    manufactured_solution.cc
    scalar_functionals/omega.cc
    scalar_functionals/psi.cc
)

#link against cmf and include headers
FIND_PACKAGE(cmf QUIET HINTS $ENV{CMF_DIR} ${CMF_DIR})
IF(NOT ${cmf_FOUND})
  MESSAGE("Checking for libCMF ...  not found")
ELSE()
  MESSAGE("Checking for libCMF ...  found")
  LINK_LIBRARIES(${CMF_LIBRARIES})
  INCLUDE_DIRECTORIES(${CMF_INSTALL_PATH_HEADERS})
  INCLUDE_DIRECTORIES(${CMF_INCLUDE_DIRECTORIES})
  OPTION(INCREMENTAL_FE_WITH_CMF "Enable CMF" ON)
ENDIF()

#find spline library of T. Kluge
IF(SPLINE_DIR)
  INCLUDE_DIRECTORIES (${SPLINE_DIR})
  OPTION(INCREMENTAL_FE_WITH_SPLINE "Enable SPLINE" ON)
ELSE()
  MESSAGE("You must pass the path to the header file of the spline library of T. Kluge (https://kluge.in-chemnitz.de/opensource/spline/) if you want to use that library. Note that the SPLINE library is distributed under GPLv2, while this library is distributed under LGPLv2.")
ENDIF()

#define the debug and release targets of the library incremental_fe
FOREACH(_build "debug" "release")

    #set up the target
    STRING(TOUPPER ${_build} _build_toupper)
    SET(_target ${INCREMENTAL_FE_TARGET_${_build_toupper}})
    ADD_LIBRARY(${_target} SHARED ${_target_src})

    #define compile flags and definitions and link flags (according to what has been used to compile GalerkinTools)
    SET_PROPERTY(TARGET ${_target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${GALERKIN_TOOLS_COMPILE_FLAGS_${_build_toupper}}")
    SET_PROPERTY(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS "${GALERKIN_TOOLS_COMPILE_DEFINITIONS_${_build_toupper}}")
    SET_PROPERTY(TARGET ${_target} APPEND PROPERTY LINK_FLAGS "${GALERKIN_TOOLS_LINK_FLAGS_${_build_toupper}}")

    #link against galerkin_tools
    TARGET_LINK_LIBRARIES(${_target} ${GALERKIN_TOOLS_LIBRARIES_${_build_toupper}})
 
    #install the library galerkin_tools (debug and release)
    INSTALL(TARGETS ${_target} LIBRARY DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_LIBRARIES})

    #the libraries to link to when using IncrementalFE (this will end up in the config file)
    GET_TARGET_PROPERTY(LINK_LIBRARIES_ ${_target} LINK_LIBRARIES)
    SET(INCREMENTAL_FE_LIBRARIES "${INCREMENTAL_FE_INSTALL_PATH_LIBRARIES}/libincremental_fe.${_build}.so")
    FOREACH(LINK_LIBRARY ${LINK_LIBRARIES_})
      STRING(APPEND INCREMENTAL_FE_LIBRARIES ";${LINK_LIBRARY}")
    ENDFOREACH(LINK_LIBRARY)
    SET_PROPERTY(GLOBAL PROPERTY INCREMENTAL_FE_LIBRARIES_${_build_toupper} ${INCREMENTAL_FE_LIBRARIES})

    #flags (this will end up in the config file)
    GET_TARGET_PROPERTY(INCREMENTAL_FE_COMPILE_FLAGS ${_target} COMPILE_FLAGS)
    SET_PROPERTY(GLOBAL PROPERTY INCREMENTAL_FE_COMPILE_FLAGS_${_build_toupper} ${INCREMENTAL_FE_COMPILE_FLAGS})
    GET_TARGET_PROPERTY(INCREMENTAL_FE_COMPILE_DEFINITIONS ${_target} COMPILE_DEFINITIONS)
    SET_PROPERTY(GLOBAL PROPERTY INCREMENTAL_FE_COMPILE_DEFINITIONS_${_build_toupper} ${INCREMENTAL_FE_COMPILE_DEFINITIONS})
    GET_TARGET_PROPERTY(INCREMENTAL_FE_LINK_FLAGS ${_target} LINK_FLAGS)
    SET_PROPERTY(GLOBAL PROPERTY INCREMENTAL_FE_LINK_FLAGS_${_build_toupper} ${INCREMENTAL_FE_LINK_FLAGS})

ENDFOREACH(_build)
INCLUDE_DIRECTORIES(${GALERKIN_TOOLS_INCLUDE_DIRECTORIES})

#the include directories needed when using IncrementalFE (this will end up in the config file)
GET_PROPERTY(INCLUDE_DIRECTORIES_ DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
SET(INCREMENTAL_FE_INCLUDE_DIRECTORIES "${INCREMENTAL_FE_INSTALL_PATH_HEADERS}")
FOREACH(INCLUDE_DIRECTORY ${INCLUDE_DIRECTORIES_})
  STRING(APPEND INCREMENTAL_FE_INCLUDE_DIRECTORIES ";${INCLUDE_DIRECTORY}")
ENDFOREACH(INCLUDE_DIRECTORY)
SET_PROPERTY(GLOBAL PROPERTY INCREMENTAL_FE_INCLUDE_DIRECTORIES ${INCREMENTAL_FE_INCLUDE_DIRECTORIES})

#headers of library incremental_fe itself
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/../include)

#make config.h
CONFIGURE_FILE(../include/incremental_fe/config.h.in ../include/incremental_fe/config.h)

#install the headers of the library incremental_fe
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/include/incremental_fe DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_HEADERS} FILES_MATCHING PATTERN "*.h")
INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../include/incremental_fe DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_HEADERS} FILES_MATCHING PATTERN "*.h")


