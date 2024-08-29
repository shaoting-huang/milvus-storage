
add_source_at_current_directory()
add_definitions(-DAZURE_BUILD_DIR)
include_directories(azure-blob-storage)
include_directories("${AZURE_BUILD_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include")
add_library(azure_blob_chunk_manager SHARED IMPORTED)
set_target_properties(azure_blob_chunk_manager
    PROPERTIES
    IMPORTED_GLOBAL TRUE
    IMPORTED_LOCATION "${AZURE_BUILD_DIR}/libblob-chunk-manager${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
message("AZURE_IMPORTED_LOCATION: ${AZURE_IMPORTED_LOCATION}")
message("AZURE_INTERFACE_INCLUDE_DIRECTORIES: ${AZURE_INTERFACE_INCLUDE_DIRECTORIES}")

add_library(milvus_storage SHARED ${SOURCE_FILES})