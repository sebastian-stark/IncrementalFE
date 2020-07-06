#source files from which library incremental_fe is compiled
SET(_target_src
    global_data_incremental_fe.cc
    fe_model.cc
    constraints.cc
    scalar_functionals/omega.cc
    scalar_functionals/psi.cc
)

#define the debug and release targets of the library incremental_fe
FOREACH(_build "debug" "release")

    #set up the target
    STRING(TOUPPER ${_build} _build_toupper)
    SET(_target ${INCREMENTAL_FE_TARGET_${_build_toupper}})
    ADD_LIBRARY(${_target} SHARED ${_target_src})

    #define compile flags and definitions (according to what has been used to compile deal.II)
    SET_PROPERTY(TARGET ${_target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_${_build_toupper}}")
    SET_PROPERTY(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS "${DEAL_II_USER_DEFINITIONS};${DEAL_II_USER_DEFINITIONS_${_build_toupper}}")

    #link against galerkin_tools and include headers of galerkin_tools
    #work around the cache of cmake
    UNSET(GALERKIN_TOOLS_LIBRARY CACHE)
    FIND_LIBRARY(GALERKIN_TOOLS_LIBRARY ${GALERKIN_TOOLS_TARGET_${_build_toupper}} HINTS ${GALERKIN_TOOLS_INSTALL_PATH_LIBRARIES})
    IF(${GALERKIN_TOOLS_LIBRARY} MATCHES "GALERKIN_TOOLS_LIBRARY-NOTFOUND")
      MESSAGE(FATAL_ERROR "\n"
        "*** Could not locate galerkin_tools ***\n")
    ELSE()
      TARGET_LINK_LIBRARIES(${_target} ${GALERKIN_TOOLS_LIBRARY})
      TARGET_INCLUDE_DIRECTORIES(${_target} PRIVATE ${GALERKIN_TOOLS_INSTALL_PATH_HEADERS})
    ENDIF()

    #link against deal.II and include headers of deal.II
    TARGET_LINK_LIBRARIES(${_target} ${DEAL_II_LIBRARIES_${_build_toupper}})
    TARGET_INCLUDE_DIRECTORIES(${_target} PRIVATE ${DEAL_II_INCLUDE_DIRS})

    #headers of library incremental_fe
    TARGET_INCLUDE_DIRECTORIES(${_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)

    #install the library galerkin_tools (debug and release)
    INSTALL(TARGETS ${_target} LIBRARY DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_LIBRARIES})

ENDFOREACH(_build)

#install the headers of the library incremental_fe
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/include/incremental_fe DESTINATION ${INCREMENTAL_FE_INSTALL_PATH_HEADERS})

