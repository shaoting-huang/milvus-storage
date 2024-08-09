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

#include "packed/reader.h"
#include <arrow/array/array_base.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/properties.h>
#include "common/arrow_util.h"

namespace milvus_storage {

PackedRecordBatchReader::PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                                                 const std::vector<std::string>& paths,
                                                 const std::shared_ptr<arrow::Schema> schema,
                                                 const std::vector<std::pair<int, int>>& column_offsets,
                                                 const std::set<int>& needed_columns,
                                                 const int64_t buffer_size)
    : schema_(std::move(schema)), buffer_available_(buffer_size), row_offset_limit_(0), absolute_row_position_(0) {
  for (int i : needed_columns) {
    needed_column_offsets_.push_back(column_offsets[i]);
    auto result = MakeArrowFileReader(fs, paths[i]);
    if (!result.ok()) {
      throw std::runtime_error(result.status().ToString());
    }
    file_readers_.emplace_back(std::move(result.value()));
  }

  // Initialize table states and chunk states
  table_states_.resize(file_readers_.size(), TableState(0, -1, 0));
  chunk_states_.resize(needed_column_offsets_.size(), ChunkState(0, 0));

  // tables are referrenced by column_offsets, so it's size is of paths's size.
  tables_.resize(paths.size(), nullptr);
}

std::shared_ptr<arrow::Schema> PackedRecordBatchReader::schema() const { return schema_; }

arrow::Status PackedRecordBatchReader::advanceBuffer() {
  std::vector<std::vector<int>> rgs_to_read(file_readers_.size());
  size_t plan_buffer_size = 0;

  // Advances to the next row group in a specific file reader and calculates the required buffer size.
  auto advance_row_group = [&](int i) -> int64_t {
    auto& reader = file_readers_[i];
    int rg = table_states_[i].row_group_offset + 1;
    if (rg >= reader->parquet_reader()->metadata()->num_row_groups()) {
      // No more row groups. It means we're done or there is an error.
      return -1;
    }
    rgs_to_read[i].push_back(rg);
    const auto metadata = reader->parquet_reader()->metadata()->RowGroup(rg);
    int64_t rg_size = metadata->total_byte_size();
    plan_buffer_size += rg_size;
    table_states_[i].addMemorySize(rg_size);
    table_states_[i].setRowGroupOffset(rg);
    table_states_[i].addRowOffset(metadata->num_rows());
    return rg_size;
  };

  // Resets the chunk states for columns in a specific file.
  auto reset_chunk_states = [&](int i) {
    for (int j = 0; j < needed_column_offsets_.size(); ++j) {
      if (needed_column_offsets_[j].first == i) {
        chunk_states_[j].reset();
      }
    }
  };

  // Fill in tables that have no rows available
  int drained_index = -1;
  for (int i = 0; i < file_readers_.size(); ++i) {
    if (table_states_[i].row_offset > row_offset_limit_) {
      continue;
    }
    buffer_available_ += table_states_[i].memory_size;
    table_states_[i].resetMemorySize();
    if (advance_row_group(i) < 0) {
      drained_index = i;
      break;
    }
    reset_chunk_states(i);
  }

  if (drained_index >= 0 && plan_buffer_size == 0) {
    return arrow::Status::OK();
  }

  // Fill in tables if we have enough buffer size
  // find the lowest offset table and advance it
  ColumnOffsetMinHeap sorted_offsets;
  for (int i = 0; i < file_readers_.size(); ++i) {
    sorted_offsets.emplace(i, table_states_[i].row_offset);
  }
  while (true) {
    int i = sorted_offsets.top().path_index;
    int rg = table_states_[i].row_group_offset + 1;
    auto& reader = file_readers_[i];
    int64_t size_in_plan = advance_row_group(i);
    if (size_in_plan < 0 || plan_buffer_size + size_in_plan >= buffer_available_) {
      break;
    }
    sorted_offsets.pop();
    sorted_offsets.emplace(i, table_states_[i].row_offset);
  }

  // Conduct read and update buffer size
  for (int i = 0; i < file_readers_.size(); ++i) {
    RETURN_NOT_OK(file_readers_[i]->ReadRowGroups(rgs_to_read[i], &tables_[i]));
  }
  buffer_available_ -= plan_buffer_size;
  row_offset_limit_ = sorted_offsets.top().row_offset;

  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  if (absolute_row_position_ >= row_offset_limit_) {
    RETURN_NOT_OK(advanceBuffer());
    if (absolute_row_position_ >= row_offset_limit_) {
      *out = nullptr;
      return arrow::Status::OK();
    }
  }

  // Determine the maximum contiguous slice across all tables
  int64_t chunksize = std::min(row_offset_limit_ - absolute_row_position_, DefaultBatchSize);
  std::vector<const arrow::Array*> chunks = collectChunks(chunksize);

  // Slice chunks and advance chunk index as appropriate
  std::vector<std::shared_ptr<arrow::ArrayData>> batch_data(needed_column_offsets_.size());
  for (int i = 0; i < needed_column_offsets_.size(); ++i) {
    // Exhausted chunk
    auto chunk = chunks[i];
    auto& state = chunk_states_[i];
    int64_t offset = state.offset;
    std::shared_ptr<arrow::ArrayData> slice_data;
    if (chunk->length() - offset == chunksize) {
      state.addCount(1);
      state.resetOffset();
      slice_data = (offset > 0) ? chunk->Slice(offset, chunksize)->data() : chunk->data();
    } else {
      state.addOffset(chunksize);
      slice_data = chunk->Slice(offset, chunksize)->data();
    }
    batch_data[i] = std::move(slice_data);
  }

  absolute_row_position_ += chunksize;
  *out = arrow::RecordBatch::Make(schema_, chunksize, std::move(batch_data));

  return arrow::Status::OK();
}

std::vector<const arrow::Array*> PackedRecordBatchReader::collectChunks(int64_t chunksize) const {
  std::vector<const arrow::Array*> chunks(needed_column_offsets_.size());

  for (int i = 0; i < needed_column_offsets_.size(); ++i) {
    auto offset = needed_column_offsets_[i];
    auto table = tables_[offset.first];
    auto column = table->column(offset.second);

    auto chunk = column->chunk(chunk_states_[i].count).get();
    int64_t chunk_remaining = chunk->length() - chunk_states_[i].offset;

    if (chunk_remaining < chunksize) {
      chunksize = chunk_remaining;
    }

    chunks[i] = chunk;
  }
  return chunks;
}

}  // namespace milvus_storage