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
set(EXE_NAME ${PROJECT_NAME})
set(TEST_FILES tests/tests.c)
set(SOURCE_FILES src/lib.c)
set(HEADER_FILES src/lib.h)
set(MAIN_FILE src/main.c)
set(C_STANDARD 99)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET ${LIBRARY_NAME} PROPERTY C_STANDARD ${C_STANDARD}})
target_include_directories(${LIBRARY_NAME} PUBLIC src)

add_executable(${EXE_NAME} ${MAIN_FILE})
set_property(TARGET ${EXE_NAME} PROPERTY C_STANDARD ${C_STANDARD})
target_link_libraries(${EXE_NAME} PRIVATE ${LIBRARY_NAME})

#enable_testing()

#set(TEST_NAME ${LIBRARY_NAME}_test)
#add_executable(${TEST_NAME} ${TEST_FILES})
#target_include_directories(${TEST_NAME} PRIVATE tests)
#target_link_libraries(${TEST_NAME} PRIVATE ${LIBRARY_NAME})

#add_test(test1 ${TEST_NAME})

# Add Sundials
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/third-party CACHE PATH "Directory to put third-party libs." FORCE)
FetchContent_Declare(
  sundials 
  GIT_REPOSITORY https://github.com/LLNL/sundials.git
  GIT_TAG        v6.6.0
)

FetchContent_MakeAvailable(sundials)
set(Sundials_TARGETS 
    sundials_cvode 
    sundials_cvode_obj 
    sundials_nvecserial 
    sundials_nvecserial_obj 
    sundials_sunlinsoldense 
    sundials_sunlinsoldense_obj 
    sundials_generic 
    sundials_generic_obj
    sundials_sunmatrixdense
    sundials_sunmatrixdense_obj
)
target_link_libraries(${LIBRARY_NAME} PRIVATE ${Sundials_TARGETS})
target_include_directories(${LIBRARY_NAME} PUBLIC "${FETCHCONTENT_BASE_DIR}/sundials-src/include" "${FETCHCONTENT_BASE_DIR}/sundials-build/include")

# Install library
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

# Install executable
install(TARGETS ${EXE_NAME} DESTINATION bin)

# Install library headers
install(FILES ${HEADER_FILES} DESTINATION include)
