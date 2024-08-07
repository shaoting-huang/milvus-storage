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

#include "reader/packed_reader.h"
#include <arrow/array/array_base.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/properties.h>
#include <queue>
#include "common/arrow_util.h"

namespace milvus_storage {

PackedRecordBatchReader::PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                                                 std::vector<std::string>& paths,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 std::vector<ColumnOffset>& column_offsets,
                                                 std::vector<int>& needed_columns,
                                                 size_t buffer_size)
    : fs_(fs), paths_(paths), schema_(std::move(schema)), needed_columns_(needed_columns), buffer_size_(buffer_size) {
  buffer_available_ = buffer_size_;

  for (int i : needed_columns) {
    needed_column_offsets_.push_back(column_offsets[i]);
    needed_path_indices_.insert(column_offsets[i].file_index);
  }
}

std::shared_ptr<arrow::Schema> PackedRecordBatchReader::schema() const { return schema_; }

arrow::Status PackedRecordBatchReader::openInternal() {
  // auto read_properties = parquet::default_arrow_reader_properties();
  // read_properties.set_pre_buffer(true);
  for (int i = 0; i < paths_.size(); i++) {
    // auto res = MakeArrowFileReader(fs_, path, read_properties);
    if (needed_path_indices_.find(i) == needed_path_indices_.end()) {
      continue;
    }
    auto res = MakeArrowFileReader(fs_, paths_[i]);  // PreBuffer is turned on by default
    if (!res.ok()) {
      throw std::runtime_error(res.status().ToString());
    }
    file_readers_.emplace_back(std::move(res.value()));
  }
  row_group_offsets_.assign(file_readers_.size(), -1);
  row_offsets_.assign(file_readers_.size(), 0);
  table_memory_sizes_.assign(file_readers_.size(), 0);
  tables_.assign(paths_.size(),
                 nullptr);  // tables are referrenced by column_offsets, so it's size is of paths_'s size.
  limit_ = 0;
  absolute_row_position_ = 0;
  chunk_numbers_.assign(needed_columns_.size(), 0);
  chunk_offsets_.assign(needed_columns_.size(), 0);
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::advanceBuffer() {
  if (file_readers_.empty()) {
    RETURN_NOT_OK(openInternal());
  }

  std::vector<std::vector<int>> rgs_to_read(file_readers_.size(), std::vector<int>());
  size_t plan_buffer_size = 0;

  auto advance_row_group = [&](int file_index) {
    auto& reader = file_readers_[file_index];
    int rg = row_group_offsets_[file_index] + 1;
    if (rg < reader->parquet_reader()->metadata()->num_row_groups()) {
      rgs_to_read[file_index].push_back(rg);
      int64_t rg_size = reader->parquet_reader()->metadata()->RowGroup(rg)->total_byte_size();
      plan_buffer_size += rg_size;
      table_memory_sizes_[file_index] += rg_size;
      row_group_offsets_[file_index] = rg;
      row_offsets_[file_index] += reader->parquet_reader()->metadata()->RowGroup(rg)->num_rows();
      return rg;
    }
    // No more row groups. It means we're done or there is an error.
    return -1;
  };

  // Fill in tables that have no rows available
  int drained_index = -1;
  for (int i = 0; i < file_readers_.size(); ++i) {
    if (row_offsets_[i] > limit_) {
      continue;
    }
    buffer_available_ += table_memory_sizes_[i];  // Release memory
    table_memory_sizes_[i] = 0;
    int rg = advance_row_group(i);
    if (rg < 0) {
      drained_index = i;
      break;
    }
    // TODO: reset chunk_numbers_
    for (int j = 0; j < needed_column_offsets_.size(); ++j) {
      if (needed_column_offsets_[j].file_index == i) {
        chunk_numbers_[j] = 0;
        chunk_offsets_[j] = 0;
      }
    }
  }
  if (drained_index >= 0 && plan_buffer_size == 0) {
    return arrow::Status::OK();
  }

  // Fill in tables if we have enough buffer size
  // find the lowest offset table and advance it
  auto compareColumnOffset = [](const ColumnOffset& a, const ColumnOffset& b) {
    return a.column_index < b.column_index;
  };
  std::priority_queue<ColumnOffset, std::vector<ColumnOffset>, decltype(compareColumnOffset)> sorted_offsets(
      compareColumnOffset);
  for (int i = 0; i < row_offsets_.size(); ++i) {
    sorted_offsets.emplace(i, row_offsets_[i]);
  }
  while (true) {
    ColumnOffset lowest_offset = sorted_offsets.top();
    int file_index = lowest_offset.file_index;
    int rg = row_group_offsets_[file_index] + 1;
    auto& reader = file_readers_[file_index];
    if (rg < reader->parquet_reader()->metadata()->num_row_groups()) {
      int64_t size_in_plan = reader->parquet_reader()->metadata()->RowGroup(rg)->total_byte_size();
      if (plan_buffer_size + size_in_plan < buffer_available_) {
        int rg = advance_row_group(file_index);
        if (rg < 0) {
          break;
        }
        sorted_offsets.pop();
        sorted_offsets.emplace(file_index, row_offsets_[file_index]);
        continue;
      }
    }
    break;
  }

  // Conduct read and update buffer size
  for (int i = 0; i < file_readers_.size(); ++i) {
    RETURN_NOT_OK(file_readers_[i]->ReadRowGroups(rgs_to_read[i], &tables_[i]));
  }
  buffer_available_ -= plan_buffer_size;
  limit_ = sorted_offsets.top().column_index;

  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  if (absolute_row_position_ >= limit_) {
    RETURN_NOT_OK(advanceBuffer());
    if (absolute_row_position_ >= limit_) {
      *out = nullptr;
      return arrow::Status::OK();
    }
  }

  // Determine the maximum contiguous slice across all tables
  size_t chunksize = std::min(limit_ - absolute_row_position_, DefaultBatchSize);
  std::vector<const arrow::Array*> chunks(needed_column_offsets_.size());

  for (int i = 0; i < needed_column_offsets_.size(); ++i) {
    int column_index = needed_column_offsets_[i].column_index;
    auto column = tables_[needed_column_offsets_[i].file_index]->column(column_index);

    auto chunk = column->chunk(chunk_numbers_[i]).get();
    size_t chunk_remaining = chunk->length() - chunk_offsets_[i];

    if (chunk_remaining < chunksize) {
      chunksize = chunk_remaining;
    }

    chunks[i] = chunk;
  }

  // Slice chunks and advance chunk index as appropriate
  std::vector<std::shared_ptr<arrow::ArrayData>> batch_data(needed_column_offsets_.size());

  for (int i = 0; i < needed_column_offsets_.size(); ++i) {
    // Exhausted chunk
    auto chunk = chunks[i];
    auto offset = chunk_offsets_[i];
    std::shared_ptr<arrow::ArrayData> slice_data;
    if (chunk->length() - offset == chunksize) {
      ++chunk_numbers_[i];
      chunk_offsets_[i] = 0;
      slice_data = (offset > 0) ? chunk->Slice(offset, chunksize)->data() : chunk->data();
    } else {
      chunk_offsets_[i] += chunksize;
      slice_data = chunk->Slice(offset, chunksize)->data();
    }
    batch_data[i] = std::move(slice_data);
  }

  absolute_row_position_ += chunksize;
  *out = arrow::RecordBatch::Make(schema_, chunksize, std::move(batch_data));

  return arrow::Status::OK();
}

}  // namespace milvus_storage
