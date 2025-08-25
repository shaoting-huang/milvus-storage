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

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/reader.h"

namespace internal::api {

/**
 * @brief Factory for creating format-specific chunk readers
 *
 * This factory creates appropriate ChunkReader instances for different
 * file formats. Each reader is responsible for reading one column group only.
 */
class ChunkReaderFactory {
  public:
  /**
   * @brief Create a chunk reader for a single column group
   *
   * @param format The file format to create a reader for
   * @param fs Filesystem interface
   * @param column_group The column group this reader will handle
   * @param needed_columns Vector of column names to read (empty = all columns)
   * @param properties Read properties
   * @return Unique pointer to the created chunk reader
   */
  static std::unique_ptr<milvus_storage::api::ChunkReader> create_reader(
      milvus_storage::api::FileFormat format,
      std::shared_ptr<arrow::fs::FileSystem> fs,
      std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
      std::vector<std::string> needed_columns,
      const milvus_storage::api::ReadProperties& properties);

  private:
  ChunkReaderFactory() = default;
};

}  // namespace internal::api

namespace milvus_storage::api {

/**
 * @brief Parquet format reader implementation that extends ChunkReader
 *
 * This class extends the ChunkReader functionality with Parquet-specific
 * optimizations and direct access to Parquet row groups as chunks.
 */
class ParquetFormatReader : public ChunkReader {
  public:
  ParquetFormatReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::shared_ptr<ColumnGroup> column_group,
                      std::vector<std::string> needed_columns,
                      const ReadProperties& properties = default_read_properties);

  ~ParquetFormatReader() = default;

  // Override ChunkReader methods for Parquet-specific optimizations
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) const;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) const;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, int64_t parallelism = 1) const;
  [[nodiscard]] arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) const override;

  /**
   * @brief Get the number of row groups (chunks) in the Parquet file
   */
  [[nodiscard]] arrow::Result<int64_t> get_num_chunks() const;

  private:
  ReadProperties properties_;
  mutable std::unique_ptr<parquet::arrow::FileReader> parquet_reader_;
  mutable bool initialized_;

  /**
   * @brief Initialize the Parquet reader if not already initialized
   */
  [[nodiscard]] arrow::Status ensure_initialized() const;
};

}  // namespace milvus_storage::api