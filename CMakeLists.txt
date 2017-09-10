cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set (CMAKE_CXX_STANDARD 11)

 
project(Haze)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
find_package(SUPERLU REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)
 
add_executable(haze MACOSX_BUNDLE read.cxx)
 
target_link_libraries(haze ${ITK_LIBRARIES} ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES})
