cmake_minimum_required(VERSION 3.12)

# Definitions
set(PROJECT_NAME diffeq_runtime)
project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES C)

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

# Set global compiler warnings
if(MSVC)
    add_compile_options(/W3 /WX)
else()
    add_compile_options(-Wall -pedantic)
endif()

set(LIBRARY_NAME ${PROJECT_NAME}_lib)
set(RUNTIME_NAME ${PROJECT_NAME})
set(TEST_NAME ${PROJECT_NAME}_test)
if (DEFINED EMSCRIPTEN)
  add_compile_definitions(EMSCRIPTEN)
endif()

set(TEST_FILES tests/test_logistic.c)
set(TEST_FILES_ODE tests/test_logistic_ode.c)
add_definitions(-DUNITY_INCLUDE_DOUBLE)
set(SOURCE_FILES src/diffeq.c)
set(HEADER_FILES src/diffeq.h)
set(MAIN_FILE src/main.c)
set(LOGISTIC_GROWTH_FILES tests/logistic_growth.c)
set(C_STANDARD 99)

## library
add_library(${LIBRARY_NAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET ${LIBRARY_NAME} PROPERTY C_STANDARD ${C_STANDARD}})
target_include_directories(${LIBRARY_NAME} PUBLIC src)

## c runtime
add_library(${RUNTIME_NAME} STATIC ${MAIN_FILE})
set_property(TARGET ${RUNTIME_NAME} PROPERTY C_STANDARD ${C_STANDARD})
target_link_libraries(${RUNTIME_NAME} PUBLIC ${LIBRARY_NAME})

if (DEFINED EMSCRIPTEN)
	set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-Os -s WASM=1 ")
	set_target_properties(${RUNTIME_NAME} PROPERTIES COMPILE_FLAGS "-Os -s WASM=1 ")
	set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS    "-Os -s WASM=1 ")
	set_target_properties(${RUNTIME_NAME} PROPERTIES LINK_FLAGS    "-Os -s WASM=1 ")
endif()

enable_testing()

## logistic growth executable test
add_library(logistic_growth OBJECT ${LOGISTIC_GROWTH_FILES})
target_link_libraries(logistic_growth PRIVATE ${LIBRARY_NAME})
add_executable(logistic_growth_test)
set_property(TARGET logistic_growth_test PROPERTY LINKER_LANGUAGE C)
target_link_libraries(logistic_growth_test PRIVATE ${RUNTIME_NAME} logistic_growth)

add_test(NAME logistic_growth_fixed_times_test COMMAND ${CMAKE_SOURCE_DIR}/tests/logistic_growth_fixed_times_test.sh )
add_test(NAME logistic_growth_solver_times_test COMMAND ${CMAKE_SOURCE_DIR}/tests/logistic_growth_solver_times_test.sh )

# Unity tests
add_executable(${TEST_NAME} ${TEST_FILES})
add_executable(${TEST_NAME}_ode ${TEST_FILES_ODE})
add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
add_test(NAME ${TEST_NAME}_ode COMMAND ${TEST_NAME}_ode)

# Add Sundials
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Directory to put third-party libs." FORCE)
FetchContent_Declare(
  sundials 
  GIT_REPOSITORY https://github.com/martinjrobins/sundials.git
  GIT_TAG        main
)

message("cmake module path is now ${CMAKE_MODULE_PATH}")
set(ENABLE_KLU ON CACHE BOOL "Enable KLU")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared library")

FetchContent_GetProperties(sundials)
if(NOT sundials_POPULATED)
  FetchContent_Populate(sundials)
  add_subdirectory(${sundials_SOURCE_DIR} ${sundials_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(Sundials_TARGETS 
    sundials_idas
    sundials_sunlinsolklu
    sundials_sunmatrixsparse
)
target_link_libraries(${LIBRARY_NAME} PUBLIC ${Sundials_TARGETS})

if (NOT DEFINED EMSCRIPTEN)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
      message(INFO "OpenMP found")
      target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_C)
  else()
      message(WARNING "OpenMP not found")
  endif()
endif()

# Add argparse
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/cofyc/argparse.git
  GIT_TAG        v1.1.0
)

set(ARGPARSE_SHARED OFF CACHE BOOL "Build shared library")
FetchContent_GetProperties(argparse)
if(NOT argparse_POPULATED)
  FetchContent_Populate(argparse)
  add_subdirectory(${argparse_SOURCE_DIR} ${argparse_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(Argparse_TARGETS argparse)
target_link_libraries(${RUNTIME_NAME} PRIVATE ${Argparse_TARGETS})
target_include_directories(${RUNTIME_NAME} PUBLIC "${FETCHCONTENT_BASE_DIR}/argparse-src")

# Add Unity 
FetchContent_Declare(
  unity
  GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
  GIT_TAG        v2.6.0_RC1
)

FetchContent_GetProperties(unity)
if(NOT unity_POPULATED)
  FetchContent_Populate(unity)
  add_subdirectory(${unity_SOURCE_DIR} ${unity_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(Unity_TARGETS unity)
target_link_libraries(${TEST_NAME} PRIVATE ${Unity_TARGETS} ${LIBRARY_NAME} ${Sundials_TARGETS})
target_include_directories(${TEST_NAME} PUBLIC "${FETCHCONTENT_BASE_DIR}/unity-src")
target_link_libraries(${TEST_NAME}_ode PRIVATE ${Unity_TARGETS} ${LIBRARY_NAME} ${Sundials_TARGETS})
target_include_directories(${TEST_NAME}_ode PUBLIC "${FETCHCONTENT_BASE_DIR}/unity-src")


# Install third-party libs if wasm
if (DEFINED EMSCRIPTEN)
  set(prefix ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(suffix ${CMAKE_STATIC_LIBRARY_SUFFIX})
  foreach (target ${Sundials_TARGETS})
    install(FILES $<TARGET_FILE:${target}> DESTINATION lib )
  endforeach ()
  install(FILES $<TARGET_FILE:argparse> DESTINATION lib )
endif()

# Install library
set_property(TARGET ${LIBRARY_NAME} PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

# Install runtime
set_property(TARGET ${RUNTIME_NAME} PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS ${RUNTIME_NAME} DESTINATION lib)

# Install library headers
install(FILES ${HEADER_FILES} DESTINATION include)
