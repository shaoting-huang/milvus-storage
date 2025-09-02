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
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/reader.h"

namespace milvus_storage {

class FileRowGroupReader : public milvus_storage::api::ChunkReader {
  public:
  /**
   * @brief FileRowGroupReader reads specified row groups with memory constraints. The schema is the same as the file
   * schema.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Path to the Parquet file.
   * @param buffer_size Memory limit for reading row groups.
   * @param reader_props The reader properties.
   * @param needed_columns Subset of columns to read (empty = all columns)
   */
  FileRowGroupReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                     const std::string& path,
                     const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
                     parquet::ReaderProperties reader_props = parquet::default_reader_properties(),
                     const std::vector<std::string>& needed_columns = {});

  Status SetRowGroupOffsetAndCount(int row_group_offset, int row_group_num);

  std::shared_ptr<arrow::Schema> schema() const;
  /**
   * @brief Returns packed file metadata.
   */
  std::shared_ptr<PackedFileMetadata> file_metadata();

  /**
   * @brief Reads the next row group from the file.
   *
   * @param out A shared pointer to the output table.
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status ReadNextRowGroup(std::shared_ptr<arrow::Table>* out);

  // ChunkReader interface implementations
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(
      const std::vector<int64_t>& row_indices) const override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) const override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, int64_t parallelism = 1) const override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) const override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_row_num(int64_t chunk_index) const override;

  /**
   * @brief Closes the reader and releases resources.
   *
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status Close();

  protected:
  [[nodiscard]] arrow::Status validate_chunk_index(int64_t chunk_index) const override;

  private:
  Status init(std::shared_ptr<arrow::fs::FileSystem> fs,
              const std::string& path,
              const int64_t buffer_size,
              parquet::ReaderProperties reader_props = parquet::default_reader_properties());

  /**
   * @brief Slices a row group from the table and updates the buffer state.
   *
   * @param table The input table to slice from.
   * @param buffer_size The current buffer size.
   * @param current_rg The current row group index.
   * @param out The output sliced table.
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status SliceRowGroupFromTable(std::shared_ptr<arrow::Table>* out);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string path_;
  std::vector<int> needed_columns_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<parquet::arrow::FileReader> file_reader_;
  int rg_start_ = -1;
  int rg_end_ = -1;
  int current_rg_ = -1;

  int64_t buffer_size_limit_;
  int64_t buffer_size_ = 0;
  std::shared_ptr<PackedFileMetadata> file_metadata_;
  std::shared_ptr<arrow::Table> buffer_table_ = nullptr;
};

class Reader {
  public:
  virtual void Close() = 0;

  virtual Result<std::shared_ptr<arrow::Table>> ReadByOffsets(std::vector<int64_t>& offsets) = 0;

  virtual ~Reader() = default;
};

class ParquetFileReader : public Reader {
  public:
  ParquetFileReader(std::unique_ptr<parquet::arrow::FileReader> reader);

  void Close() override {}

  Result<std::shared_ptr<arrow::Table>> ReadByOffsets(std::vector<int64_t>& offsets) override;

  private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
};
}  // namespace milvus_storage
