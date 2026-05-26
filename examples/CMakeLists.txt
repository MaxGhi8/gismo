######################################################################
## CMakeLists.txt --- examples
## This file is part of the G+Smo library.
##
## Author: Angelos Mantzaflaris, Harald Weiner
######################################################################

project(examples)

set(CMAKE_DIRECTORY_LABELS "${PROJECT_NAME}") #CMake 3.10
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/)

# Add a grouping target that builds all tests this is equivalent to
# running "make" inside the tests build folder
add_custom_target(${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES LABELS "${PROJECT_NAME}" FOLDER "${PROJECT_NAME}")

#find_package(gismo REQUIRED)
#include_directories( ${GISMO_INCLUDE_DIRS})
#link_directories   ( ${GISMO_LIBRARY_DIR} )
#include( ${GISMO_USE_FILE})

# Collect source file names
aux_cpp_directory(${CMAKE_CURRENT_SOURCE_DIR} FILES)
#message("found files: ${FILES}")

if ("x${CMAKE_CXX_STANDARD}" STREQUAL "x98")
  set(IGNORE_FILES "kirchhoff-Love_example;")
  foreach(ign ${IGNORE_FILES})
    foreach(file ${FILES})
      string(REGEX MATCH "${ign}.cpp$" fmatch ${file})
      if(fmatch)
	list(REMOVE_ITEM FILES ${file})
      endif()
    endforeach()
  endforeach()
endif()

# --- ONNX Runtime support for neural-network examples (*_nn_example.cpp) ---
# Configure with -DONNXRUNTIME_ROOT=/path/to/onnxruntime to enable.
# Examples ending in _nn_example.cpp are skipped when ONNX Runtime is not found.
set(NN_EXAMPLE_FILES "")
foreach(file ${FILES})
  if(file MATCHES "_nn_example\\.cpp$")
    list(APPEND NN_EXAMPLE_FILES ${file})
  endif()
endforeach()

set(ORT_AVAILABLE FALSE)
if(ONNXRUNTIME_ROOT)
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${ONNXRUNTIME_ROOT}/include
                  ${ONNXRUNTIME_ROOT}/include/onnxruntime
                  ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
  find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
  if(ORT_INCLUDE_DIR AND ORT_LIB)
    set(ORT_AVAILABLE TRUE)
    message(STATUS "ONNX Runtime found: ${ORT_LIB}")
  else()
    message(WARNING "ONNXRUNTIME_ROOT set but onnxruntime headers or library not found under it.")
  endif()
endif()

if(NOT ORT_AVAILABLE)
  foreach(nn_file ${NN_EXAMPLE_FILES})
    list(REMOVE_ITEM FILES ${nn_file})
  endforeach()
endif()

foreach(file ${FILES})
  get_filename_component(tarname ${file} NAME_WE) # name without extension
  add_gismo_executable(${file})
  set_property(TEST ${tarname} PROPERTY LABELS "${PROJECT_NAME}")
  set_target_properties(${tarname} PROPERTIES FOLDER "${PROJECT_NAME}")
  add_dependencies(${PROJECT_NAME} ${tarname})

  if(ORT_AVAILABLE AND file MATCHES "_nn_example\\.cpp$")
    target_include_directories(${tarname} SYSTEM BEFORE PUBLIC ${ORT_INCLUDE_DIR})
    target_link_libraries(${tarname} ${ORT_LIB})
    # ORT 1.20+ requires C++17 (noexcept on function pointer types, P0012R1)
    set_target_properties(${tarname} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  endif()

  # install the example executables (optionally)
  install(TARGETS ${tarname} DESTINATION "${BIN_INSTALL_DIR}" COMPONENT exe OPTIONAL)
endforeach(file ${FILES})

list(LENGTH FILES len)
message(STATUS "Number of examples to compile: ${len}")

