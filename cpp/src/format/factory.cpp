// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/common/config.h"
#include <parquet/arrow/reader.h>
#include <parquet/metadata.h>

namespace internal::api {

// ==================== ChunkReaderFactory Implementation ====================

std::unique_ptr<milvus_storage::api::ChunkReader> ChunkReaderFactory::create_reader(
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::ReadProperties& properties) {
  if (!column_group) {
    throw std::runtime_error("Column group cannot be null");
  }

  const auto& format = column_group->format;
  const auto& file_path = column_group->path;

  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (column_group->contains_column(col_name)) {
      filtered_columns.push_back(col_name);
    }
  }

  if (column_group->format == milvus_storage::api::FileFormat::PARQUET) {
    // All PARQUET files now have packed metadata
    const int64_t default_buffer_size = 32 * 1024 * 1024;  // 32MB default
    auto reader = std::make_unique<milvus_storage::FileRowGroupReader>(
        fs, file_path, default_buffer_size, parquet::default_reader_properties(), filtered_columns);

    // Set up to read all row groups by default
    auto metadata = reader->file_metadata();
    if (metadata) {
      int total_row_groups = metadata->GetRowGroupMetadataVector().size();
      if (total_row_groups > 0) {
        auto status = reader->SetRowGroupOffsetAndCount(0, total_row_groups);
        if (!status.ok()) {
          throw std::runtime_error("Failed to set row group range: " + status.ToString());
        }
      }
    }
    return reader;
  } else {
    throw std::runtime_error("Unsupported file format: " + std::to_string(static_cast<int>(column_group->format)) +
                             ". Only PARQUET is supported.");
  }
}

// ==================== ChunkWriterFactory Implementation ====================

std::unique_ptr<milvus_storage::api::ColumnGroupWriter> ChunkWriterFactory::create_writer(
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const milvus_storage::StorageConfig& storage_config,
    const std::map<std::string, std::string>& custom_metadata) {
  if (!column_group) {
    throw std::runtime_error("Column group cannot be null");
  }

  if (!schema) {
    throw std::runtime_error("Schema cannot be null");
  }

  if (column_group->format != milvus_storage::api::FileFormat::PARQUET) {
    throw std::runtime_error("Only PARQUET format is supported for now");
  }

  // Create schema with only the columns for this column group
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& column_name : column_group->columns) {
    auto field = schema->GetFieldByName(column_name);
    if (!field) {
      throw std::runtime_error("Column '" + column_name + "' not found in schema");
    }
    fields.push_back(field);
  }
  auto column_group_schema = arrow::schema(fields);

  auto writer = std::make_unique<milvus_storage::ParquetFileWriter>(
      column_group_schema, fs, column_group->path, storage_config, parquet::default_writer_properties());

  // Initialize the writer
  auto init_status = writer->Init();
  if (!init_status.ok()) {
    throw std::runtime_error("Failed to initialize ParquetFileWriter: " + init_status.ToString());
  }

  // Add custom metadata to the writer
  for (const auto& [key, value] : custom_metadata) {
    auto status = writer->AppendKVMetadata(key, value);
    if (!status.ok()) {
      throw std::runtime_error("Failed to append metadata: " + status.ToString());
    }
  }

  return writer;
}

}  // namespace internal::api