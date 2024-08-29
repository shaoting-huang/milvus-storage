
macro(add_source_at_current_directory)
    file(GLOB SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc" "*.cpp" "*.c" "*.cxx")
    message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}  add new source files at current directory: ${SOURCE_FILES}")
endmacro()