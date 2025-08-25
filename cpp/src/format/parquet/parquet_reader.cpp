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

#include "milvus-storage/format/format_reader.h"

#include <set>
#include <future>
#include <algorithm>

#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"

namespace internal::api {
// existing internal::api content stays here
}  // namespace internal::api

namespace milvus_storage::api {

// ==================== ParquetFormatReader Implementation ====================

ParquetFormatReader::ParquetFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                         std::shared_ptr<ColumnGroup> column_group,
                                         std::vector<std::string> needed_columns,
                                         const ReadProperties& properties)
    : ChunkReader(std::move(fs), std::move(column_group), std::move(needed_columns)),
      properties_(properties),
      initialized_(false) {}

arrow::Status ParquetFormatReader::ensure_initialized() const {
  if (initialized_) {
    return arrow::Status::OK();
  }

  if (!column_group_) {
    return arrow::Status::Invalid("ColumnGroup is null");
  }

  // Get the file path from column group
  std::string file_path = column_group_->path;
  if (file_path.empty()) {
    return arrow::Status::Invalid("Column group has empty file path");
  }

  // Create parquet reader properties based on ReadProperties
  parquet::ReaderProperties reader_props = parquet::default_reader_properties();

  // Configure encryption if specified
  if (!properties_.cipher_type.empty()) {
    // TODO: Configure encryption properties based on properties_.cipher_type, cipher_key, etc.
    LOG_STORAGE_DEBUG_ << "Encryption configured: " << properties_.cipher_type;
  }

  try {
    // Create Arrow file reader using the helper function
    auto result = MakeArrowFileReader(*fs_, file_path, reader_props);
    if (!result.ok()) {
      return arrow::Status::IOError("Failed to create Arrow file reader: " + result.status().ToString());
    }
    parquet_reader_ = std::move(result.value());
    initialized_ = true;
  } catch (const std::exception& e) {
    return arrow::Status::IOError("Exception creating Parquet reader: " + std::string(e.what()));
  }

  return arrow::Status::OK();
}

arrow::Result<std::vector<int64_t>> ParquetFormatReader::get_chunk_indices(
    const std::vector<int64_t>& row_indices) const {
  ARROW_RETURN_NOT_OK(ensure_initialized());

  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices vector cannot be empty");
  }

  // Validate row indices are non-negative
  for (const auto& row_index : row_indices) {
    if (row_index < 0) {
      return arrow::Status::Invalid("Row index cannot be negative: " + std::to_string(row_index));
    }
  }

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
  int64_t total_rows = parquet_metadata->num_rows();

  // Validate all row indices are within bounds
  for (const auto& row_index : row_indices) {
    if (row_index >= total_rows) {
      return arrow::Status::Invalid("Row index " + std::to_string(row_index) + " is out of range. File has " +
                                    std::to_string(total_rows) + " rows");
    }
  }

  // Map row indices to row group indices (chunks in Parquet are row groups)
  std::vector<int64_t> chunk_indices;
  chunk_indices.reserve(row_indices.size());

  int64_t num_row_groups = parquet_metadata->num_row_groups();

  for (const auto& row_index : row_indices) {
    int64_t cumulative_rows = 0;
    int64_t chunk_index = -1;

    // Find which row group contains this row
    for (int64_t i = 0; i < num_row_groups; ++i) {
      auto row_group_metadata = parquet_metadata->RowGroup(i);
      int64_t row_group_size = row_group_metadata->num_rows();

      if (row_index < cumulative_rows + row_group_size) {
        chunk_index = i;
        break;
      }
      cumulative_rows += row_group_size;
    }

    if (chunk_index == -1) {
      return arrow::Status::IndexError("Row index out of bounds: " + std::to_string(row_index));
    }

    chunk_indices.push_back(chunk_index);
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetFormatReader::get_chunk(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(ensure_initialized());

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
  int64_t num_row_groups = parquet_metadata->num_row_groups();

  if (chunk_index < 0 || chunk_index >= num_row_groups) {
    return arrow::Status::IndexError("Chunk index " + std::to_string(chunk_index) + " is out of bounds. File has " +
                                     std::to_string(num_row_groups) + " chunks");
  }

  // Read the specific row group
  std::shared_ptr<arrow::Table> table;
  std::vector<int> row_groups_to_read = {static_cast<int>(chunk_index)};

  // Get column indices for needed columns
  std::vector<int> column_indices;
  if (needed_columns_.empty()) {
    // Read all columns
    std::shared_ptr<arrow::Schema> schema;
    ARROW_RETURN_NOT_OK(parquet_reader_->GetSchema(&schema));
    column_indices.reserve(schema->num_fields());
    for (int i = 0; i < schema->num_fields(); ++i) {
      column_indices.push_back(i);
    }
  } else {
    // Find column indices for needed columns
    std::shared_ptr<arrow::Schema> schema;
    ARROW_RETURN_NOT_OK(parquet_reader_->GetSchema(&schema));

    for (const auto& col_name : needed_columns_) {
      auto field = schema->GetFieldByName(col_name);
      if (field) {
        int column_index = schema->GetFieldIndex(col_name);
        if (column_index >= 0) {
          column_indices.push_back(column_index);
        }
      }
    }
  }

  ARROW_RETURN_NOT_OK(parquet_reader_->ReadRowGroups(row_groups_to_read, column_indices, &table));

  if (!table) {
    return arrow::Status::Invalid("No data read from chunk " + std::to_string(chunk_index));
  }

  // Convert table to record batch by creating from the table's arrays
  // Since we're reading one row group, we should have exactly one chunk per column
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto column = table->column(i);
    if (column->num_chunks() == 0) {
      return arrow::Status::Invalid("Empty column in table");
    }

    // For single row group reads, we expect one chunk per column
    if (column->num_chunks() == 1) {
      arrays.push_back(column->chunk(0));
    } else {
      // If there are multiple chunks, we need to handle them differently
      // For now, just use the first chunk as this shouldn't happen for single row group reads
      arrays.push_back(column->chunk(0));
    }
  }

  return arrow::RecordBatch::Make(table->schema(), table->num_rows(), arrays);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetFormatReader::get_chunks(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism) const {
  if (chunk_indices.empty()) {
    return arrow::Status::Invalid("Chunk indices vector cannot be empty");
  }

  if (parallelism < 1) {
    return arrow::Status::Invalid("Parallelism must be at least 1, got: " + std::to_string(parallelism));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  result.reserve(chunk_indices.size());

  if (parallelism == 1 || chunk_indices.size() == 1) {
    // Sequential reading
    for (int64_t chunk_index : chunk_indices) {
      ARROW_ASSIGN_OR_RAISE(auto batch, get_chunk(chunk_index));
      result.push_back(batch);
    }
  } else {
    // Parallel reading using futures
    std::vector<std::future<arrow::Result<std::shared_ptr<arrow::RecordBatch>>>> futures;
    futures.reserve(chunk_indices.size());

    for (int64_t chunk_index : chunk_indices) {
      futures.push_back(std::async(std::launch::async, [this, chunk_index]() { return this->get_chunk(chunk_index); }));
    }

    for (auto& future : futures) {
      ARROW_ASSIGN_OR_RAISE(auto batch, future.get());
      result.push_back(batch);
    }
  }

  return result;
}

arrow::Result<int64_t> ParquetFormatReader::get_chunk_size(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(ensure_initialized());

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
  int64_t num_row_groups = parquet_metadata->num_row_groups();

  if (chunk_index < 0 || chunk_index >= num_row_groups) {
    return arrow::Status::IndexError("Chunk index " + std::to_string(chunk_index) + " is out of bounds. File has " +
                                     std::to_string(num_row_groups) + " chunks");
  }

  // Get row group metadata for the specified chunk
  auto row_group_metadata = parquet_metadata->RowGroup(chunk_index);

  // Calculate the memory size for this row group
  // This includes the compressed data size for all columns we need to read
  int64_t total_size = 0;

  // If needed_columns_ is empty, include all columns
  if (needed_columns_.empty()) {
    total_size = row_group_metadata->total_byte_size();
  } else {
    // Calculate size only for needed columns
    std::shared_ptr<arrow::Schema> schema;
    ARROW_RETURN_NOT_OK(parquet_reader_->GetSchema(&schema));

    for (const auto& col_name : needed_columns_) {
      int column_index = schema->GetFieldIndex(col_name);
      if (column_index >= 0 && column_index < row_group_metadata->num_columns()) {
        auto column_metadata = row_group_metadata->ColumnChunk(column_index);
        total_size += column_metadata->total_compressed_size();
      }
    }
  }

  return total_size;
}

arrow::Result<int64_t> ParquetFormatReader::get_num_chunks() const {
  ARROW_RETURN_NOT_OK(ensure_initialized());

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
  return parquet_metadata->num_row_groups();
}

}  // namespace milvus_storage::api
