CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(Spectro_Tools CXX)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
SET(CMAKE_INSTALL_PREFIX .)

MESSAGE(‘’Checks if GSL is here’’)
FIND_PACKAGE(GSL)
include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES})



MESSAGE("Now compiling the source in ${Spectro_Tools_SOURCE_DIR}")
MESSAGE("The executables will be created in ${Spectro_Tools_BINARY_DIR}")

INCLUDE_DIRECTORIES(headers)
ADD_SUBDIRECTORY(src)


ADD_EXECUTABLE(Pileup_Correction_ADONIS ./src/mains/main_pileup_correction_adonis.cpp)
TARGET_LINK_LIBRARIES(Pileup_Correction_ADONIS ${LIBS} sparse stats)
INSTALL(TARGETS Pileup_Correction_ADONIS DESTINATION bin)

ADD_EXECUTABLE(Preprocess_ADONIS ./src/mains/main_adonis_process.cpp)
TARGET_LINK_LIBRARIES(Preprocess_ADONIS ${LIBS} sparse)
INSTALL(TARGETS Preprocess_ADONIS DESTINATION bin)

ADD_EXECUTABLE(NHPP_Simulator ./src/mains/main_nhpp_simulation.cpp)
TARGET_LINK_LIBRARIES(NHPP_Simulator  ${LIBS} sparse stats)
INSTALL(TARGETS NHPP_Simulator DESTINATION bin)


ENABLE_TESTING()

INCLUDE(CMakeDoc.txt)
