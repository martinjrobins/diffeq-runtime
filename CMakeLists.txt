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

if (DEFINED EMSCRIPTEN)
  set(LIBRARY_NAME ${PROJECT_NAME}_lib_wasm)
  set(RUNTIME_NAME ${PROJECT_NAME}_wasm)
else()
  set(LIBRARY_NAME ${PROJECT_NAME}_lib)
  set(RUNTIME_NAME ${PROJECT_NAME})
endif()

set(TEST_FILES tests/tests.c)
set(SOURCE_FILES src/diffeq.c)
set(HEADER_FILES src/diffeq.h)
set(MAIN_FILE src/main.c)
set(C_STANDARD 99)

## library
add_library(${LIBRARY_NAME} ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET ${LIBRARY_NAME} PROPERTY C_STANDARD ${C_STANDARD}})
target_include_directories(${LIBRARY_NAME} PUBLIC src)

## c runtime
add_library(${RUNTIME_NAME} ${MAIN_FILE})
set_property(TARGET ${RUNTIME_NAME} PROPERTY C_STANDARD ${C_STANDARD})
target_link_libraries(${RUNTIME_NAME} PRIVATE ${LIBRARY_NAME})

if (DEFINED EMSCRIPTEN)
	set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-Os -s SIDE_MODULE=1 -s WASM=1 -s STANDALONE_WASM ")
	set_target_properties(${RUNTIME_NAME} PROPERTIES COMPILE_FLAGS "-Os -s SIDE_MODULE=1 -s WASM=1 ")
	set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS    "-Os -s SIDE_MODULE=1 -s WASM=1 -s STANDALONE_WASM ")
	set_target_properties(${RUNTIME_NAME} PROPERTIES LINK_FLAGS    "-Os -s SIDE_MODULE=1 -s WASM=1 ")
endif()

enable_testing()

## logistic growth test
if (DEFINED EMSCRIPTEN)
  set(LOGISTIC_GROWTH_OBJ_FILE ${CMAKE_SOURCE_DIR}/tests/logistic_growth_wasm.o)
else()
  set(LOGISTIC_GROWTH_OBJ_FILE ${CMAKE_SOURCE_DIR}/tests/logistic_growth.o)
endif()
add_library(logistic_growth OBJECT IMPORTED)
set_property(TARGET logistic_growth PROPERTY IMPORTED_OBJECTS ${LOGISTIC_GROWTH_OBJ_FILE})
add_executable(logistic_growth_test)
set_property(TARGET logistic_growth_test PROPERTY LINKER_LANGUAGE C)
target_link_libraries(logistic_growth_test PRIVATE ${RUNTIME_NAME} logistic_growth)

add_test(NAME logistic_growth_test COMMAND ${CMAKE_SOURCE_DIR}/tests/logistic_growth_test.sh )

# Add Sundials
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/third-party CACHE PATH "Directory to put third-party libs." FORCE)
FetchContent_Declare(
  sundials 
  GIT_REPOSITORY https://github.com/LLNL/sundials.git
  GIT_TAG        v6.6.0
)

message("cmake module path is now ${CMAKE_MODULE_PATH}")
FetchContent_GetProperties(sundials)
if(NOT sundials_POPULATED)
  FetchContent_Populate(sundials)
  add_subdirectory(${sundials_SOURCE_DIR} ${sundials_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(Sundials_TARGETS 
    sundials_idas
)
target_link_libraries(${LIBRARY_NAME} PRIVATE ${Sundials_TARGETS})
target_include_directories(${LIBRARY_NAME} PUBLIC "${FETCHCONTENT_BASE_DIR}/sundials-src/include" "${FETCHCONTENT_BASE_DIR}/sundials-build/include")

# Add argparse
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/cofyc/argparse.git
  GIT_TAG        v1.1.0
)

FetchContent_GetProperties(argparse)
if(NOT argparse_POPULATED)
  FetchContent_Populate(argparse)
  add_subdirectory(${argparse_SOURCE_DIR} ${argparse_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(Argparse_TARGETS argparse)
target_link_libraries(${RUNTIME_NAME} PRIVATE ${Argparse_TARGETS})
target_include_directories(${RUNTIME_NAME} PUBLIC "${FETCHCONTENT_BASE_DIR}/argparse-src")

# Install library
set_property(TARGET ${LIBRARY_NAME} PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

# Install runtime
set_property(TARGET ${RUNTIME_NAME} PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS ${RUNTIME_NAME} DESTINATION lib)

# Install library headers
install(FILES ${HEADER_FILES} DESTINATION include)
