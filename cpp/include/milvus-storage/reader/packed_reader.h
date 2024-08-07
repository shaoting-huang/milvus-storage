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

#include <parquet/arrow/reader.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace milvus_storage {

struct ColumnOffset {
  int file_index;
  int column_index;

  ColumnOffset(int file_index, int column_index) : file_index(file_index), column_index(column_index) {}
};

struct TableState {
  int64_t row_offset;
  int64_t row_group_offset;
  int64_t memory_size;

  TableState(int64_t row_offset, int64_t row_group_offset, int64_t memory_size)
      : row_offset(row_offset), row_group_offset(row_group_offset), memory_size(memory_size) {}

  void addRowOffset(int64_t row_offset) { this->row_offset += row_offset; }

  void setRowGroupOffset(int64_t row_group_offset) { this->row_group_offset = row_group_offset; }

  void addMemorySize(int64_t memory_size) { this->memory_size += memory_size; }

  void resetMemorySize() { this->memory_size = 0; }
};

struct ChunkState {
  int count;
  int64_t offset;

  ChunkState(int count, int64_t offset) : count(count), offset(offset) {}

  void reset() {
    resetOffset();
    resetCount();
  }

  void resetOffset() { this->offset = 0; }

  void addOffset(int64_t offset) { this->offset += offset; }

  void resetCount() { this->count = 0; }

  void addCount(int count) { this->count += count; }
};

// Default number of rows to read when using ::arrow::RecordBatchReader
static constexpr int64_t DefaultBatchSize = 1024;
static constexpr int64_t DefaultBufferSize = 16 * 1024 * 1024;

class PackedRecordBatchReader : public arrow::RecordBatchReader {
  public:
  PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                          std::vector<std::string>& paths,
                          std::shared_ptr<arrow::Schema> schema,
                          std::vector<ColumnOffset>& column_offsets,
                          std::vector<int>& needed_columns,
                          int64_t buffer_size = DefaultBufferSize);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  protected:
  private:
  // Advance buffer to fill the expected buffer size
  arrow::Status advanceBuffer();
  // Open file readers
  arrow::Status openInternal();

  size_t buffer_size_;
  size_t buffer_available_;

  // Files
  arrow::fs::FileSystem& fs_;
  std::vector<std::string>& paths_;
  std::set<int> needed_path_indices_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::unique_ptr<parquet::arrow::FileReader>> file_readers_;
  std::vector<ColumnOffset> needed_column_offsets_;
  std::vector<int> needed_columns_;

  std::vector<std::shared_ptr<arrow::Table>> tables_;
  std::vector<TableState> table_states_;
  int64_t limit_;

  std::vector<ChunkState> chunk_states_;
  int64_t absolute_row_position_;
};

}  // namespace milvus_storage
