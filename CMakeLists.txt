cmake_minimum_required(VERSION 2.8)
 
project(Haze)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
 
add_executable(haze MACOSX_BUNDLE read.cxx)
 
target_link_libraries(haze ${ITK_LIBRARIES})
