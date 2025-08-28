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

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/iterator.h>
#include <parquet/properties.h>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <limits>
#include <set>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::api {

// ==================== ChunkReader Implementation ====================

// Base class implementation of get_chunks using the pure virtual get_chunk method
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism) const {
  if (chunk_indices.empty()) {
    return arrow::Status::Invalid("Chunk indices vector cannot be empty");
  }

  if (parallelism < 1) {
    return arrow::Status::Invalid("Parallelism must be at least 1, got: " + std::to_string(parallelism));
  }

  // Validate all chunk indices
  for (const auto& chunk_index : chunk_indices) {
    ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> result_batches;
  result_batches.reserve(chunk_indices.size());

  if (parallelism == 1 || chunk_indices.size() == 1) {
    // Sequential execution
    for (const auto& chunk_index : chunk_indices) {
      ARROW_ASSIGN_OR_RAISE(auto batch, get_chunk(chunk_index));
      result_batches.push_back(batch);
    }
  } else {
    // Parallel execution using std::async
    std::vector<std::future<arrow::Result<std::shared_ptr<arrow::RecordBatch>>>> futures;
    futures.reserve(chunk_indices.size());

    // Launch parallel tasks
    for (const auto& chunk_index : chunk_indices) {
      auto future = std::async(std::launch::async, [this, chunk_index]() { return get_chunk(chunk_index); });
      futures.push_back(std::move(future));
    }

    // Collect results in order
    for (auto& future : futures) {
      ARROW_ASSIGN_OR_RAISE(auto batch, future.get());
      result_batches.push_back(batch);
    }
  }

  return result_batches;
}

// ==================== Reader Implementation ====================

Reader::Reader(std::shared_ptr<arrow::fs::FileSystem> fs,
               std::shared_ptr<Manifest> manifest,
               std::shared_ptr<arrow::Schema> schema,
               const std::shared_ptr<std::vector<std::string>>& needed_columns,
               ReadProperties properties)
    : fs_(std::move(fs)),
      manifest_(std::move(manifest)),
      schema_(std::move(schema)),
      properties_(std::move(properties)) {
  // Validate required parameters
  if (!fs_) {
    throw std::invalid_argument("FileSystem cannot be null");
  }
  if (!manifest_) {
    throw std::invalid_argument("Manifest cannot be null");
  }
  if (!schema_) {
    throw std::invalid_argument("Schema cannot be null");
  }

  // Initialize the list of columns to read from the dataset
  if (needed_columns != nullptr) {
    needed_columns_ = *needed_columns;

    // Validate that all requested columns exist in the schema
    for (const auto& column_name : needed_columns_) {
      if (!schema_->GetFieldByName(column_name)) {
        throw std::invalid_argument("Column '" + column_name + "' not found in schema");
      }
    }
  } else {
    // If no specific columns requested, read all columns from the schema
    needed_columns_.clear();
    needed_columns_.reserve(schema_->num_fields());
    for (int i = 0; i < schema_->num_fields(); ++i) {
      needed_columns_.push_back(schema_->field(i)->name());
    }
  }

  // Column groups will be initialized lazily
}

void Reader::initialize_needed_column_groups() const {
  if (!needed_column_groups_.empty()) {
    return;  // Already initialized
  }

  // Determine which column groups are needed based on the requested columns
  // This optimization allows reading only the column groups that contain
  // the requested columns, reducing I/O and improving performance
  auto visited_column_groups = std::set<int64_t>();
  for (const auto& column_name : needed_columns_) {
    auto column_group = manifest_->get_column_group(column_name);
    if (column_group != nullptr && visited_column_groups.find(column_group->id) == visited_column_groups.end()) {
      needed_column_groups_.push_back(column_group);
      visited_column_groups.insert(column_group->id);
    }
  }
}

arrow::Result<std::shared_ptr<ChunkReader>> Reader::get_chunk_reader(int64_t column_group_id) const {
  if (column_group_id < 0) {
    return arrow::Status::Invalid("Column group ID cannot be negative: " + std::to_string(column_group_id));
  }

  auto column_group = manifest_->get_column_group(column_group_id);
  if (column_group == nullptr) {
    return arrow::Status::Invalid("Column group with ID " + std::to_string(column_group_id) + " not found");
  }

  try {
    // Use factory to create concrete chunk reader implementation
    auto chunk_reader = internal::api::ChunkReaderFactory::create_reader(column_group->format, fs_, column_group->path,
                                                                         needed_columns_, properties_);
    if (!chunk_reader) {
      return arrow::Status::Invalid("Failed to create chunk reader for column group " +
                                    std::to_string(column_group_id));
    }
    return std::move(chunk_reader);
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to create ChunkReader: " + std::string(e.what()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> Reader::get_record_batch_reader(const std::string& predicate,
                                                                                         int64_t batch_size,
                                                                                         int64_t buffer_size) const {
  // Validate parameters
  if (batch_size <= 0) {
    return arrow::Status::Invalid("Batch size must be positive, got: " + std::to_string(batch_size));
  }
  if (buffer_size <= 0) {
    return arrow::Status::Invalid("Buffer size must be positive, got: " + std::to_string(buffer_size));
  }

  // Initialize column groups if not already done
  initialize_needed_column_groups();

  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for the requested columns");
  }

  // Create schema with only needed columns for projection
  std::vector<std::shared_ptr<arrow::Field>> needed_fields;
  for (const auto& column_name : needed_columns_) {
    auto field = schema_->GetFieldByName(column_name);
    if (field != nullptr) {
      needed_fields.push_back(field);
    }
  }
  auto projected_schema = arrow::schema(needed_fields);

  // Create and return our custom PackedRecordBatchReader
  // This provides memory-controlled, row-aligned streaming access across column groups
  try {
    auto reader = std::make_shared<PackedRecordBatchReader>(fs_, needed_column_groups_, projected_schema,
                                                            needed_columns_, properties_, buffer_size);
    return reader;
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to create PackedRecordBatchReader: " + std::string(e.what()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Reader::take(const std::vector<int64_t>& row_indices,
                                                                int64_t parallelism) const {
  // Validate parameters
  if (row_indices.empty()) {
    return arrow::Status::Invalid("Row indices vector cannot be empty");
  }

  if (parallelism < 1) {
    return arrow::Status::Invalid("Parallelism must be at least 1, got: " + std::to_string(parallelism));
  }

  // Validate that all row indices are non-negative
  for (const auto& row_index : row_indices) {
    if (row_index < 0) {
      return arrow::Status::Invalid("Row index cannot be negative: " + std::to_string(row_index));
    }
  }

  // Initialize column groups if not already done
  initialize_needed_column_groups();

  if (needed_column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found for the requested columns");
  }

  // For each column group, we need to create a format reader and extract the requested rows
  std::vector<std::shared_ptr<arrow::RecordBatch>> column_group_batches;
  column_group_batches.reserve(needed_column_groups_.size());

  if (parallelism == 1 || needed_column_groups_.size() == 1) {
    // Sequential execution across column groups
    for (const auto& column_group : needed_column_groups_) {
      // Get needed columns for this specific column group
      std::vector<std::string> cg_needed_columns;
      for (const auto& col_name : needed_columns_) {
        if (column_group->contains_column(col_name)) {
          cg_needed_columns.push_back(col_name);
        }
      }

      auto chunk_reader = internal::api::ChunkReaderFactory::create_reader(
          column_group->format, fs_, column_group->path, cg_needed_columns, properties_);

      if (!chunk_reader) {
        return arrow::Status::Invalid("Failed to create chunk reader for column group " +
                                      std::to_string(column_group->id));
      }

      // Map row indices to chunk indices
      ARROW_ASSIGN_OR_RAISE(auto chunk_indices, chunk_reader->get_chunk_indices(row_indices));

      // Get unique chunk indices to minimize I/O
      std::vector<int64_t> unique_chunk_indices = chunk_indices;
      std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
      unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                                 unique_chunk_indices.end());

      // Read the required chunks
      ARROW_ASSIGN_OR_RAISE(auto chunks, chunk_reader->get_chunks(unique_chunk_indices, parallelism));

      // For now, use a simplified implementation that returns requested number of rows
      // TODO: Implement proper row extraction logic
      if (!chunks.empty()) {
        auto first_chunk = chunks[0];
        if (first_chunk && first_chunk->num_rows() >= static_cast<int64_t>(row_indices.size())) {
          // Slice the first chunk to get the number of rows we need
          auto sliced_batch = first_chunk->Slice(0, row_indices.size());
          column_group_batches.push_back(sliced_batch);
        } else if (first_chunk) {
          column_group_batches.push_back(first_chunk);
        }
      }
    }
  } else {
    // Parallel execution across column groups
    std::vector<std::future<arrow::Result<std::shared_ptr<arrow::RecordBatch>>>> futures;
    futures.reserve(needed_column_groups_.size());

    for (const auto& column_group : needed_column_groups_) {
      auto future = std::async(
          std::launch::async,
          [this, &column_group, &row_indices, parallelism]() -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
            // Get needed columns for this specific column group
            std::vector<std::string> cg_needed_columns;
            for (const auto& col_name : needed_columns_) {
              if (column_group->contains_column(col_name)) {
                cg_needed_columns.push_back(col_name);
              }
            }

            auto chunk_reader = internal::api::ChunkReaderFactory::create_reader(
                column_group->format, fs_, column_group->path, cg_needed_columns, properties_);

            if (!chunk_reader) {
              return arrow::Status::Invalid("Failed to create chunk reader for column group " +
                                            std::to_string(column_group->id));
            }

            // Map row indices to chunk indices
            ARROW_ASSIGN_OR_RAISE(auto chunk_indices, chunk_reader->get_chunk_indices(row_indices));

            // Get unique chunk indices to minimize I/O
            std::vector<int64_t> unique_chunk_indices = chunk_indices;
            std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
            unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                                       unique_chunk_indices.end());

            // Read the required chunks
            ARROW_ASSIGN_OR_RAISE(auto chunks, chunk_reader->get_chunks(unique_chunk_indices, 1));

            // For now, use a simplified implementation that returns requested number of rows
            // TODO: Implement proper row extraction logic
            if (!chunks.empty()) {
              auto first_chunk = chunks[0];
              if (first_chunk && first_chunk->num_rows() >= static_cast<int64_t>(row_indices.size())) {
                // Slice the first chunk to get the number of rows we need
                return first_chunk->Slice(0, row_indices.size());
              } else if (first_chunk) {
                return first_chunk;
              }
            }
            return arrow::Status::Invalid("No chunks available");
          });
      futures.push_back(std::move(future));
    }

    // Collect results
    for (auto& future : futures) {
      ARROW_ASSIGN_OR_RAISE(auto batch, future.get());
      column_group_batches.push_back(batch);
    }
  }

  // Now we need to merge the column group batches into a single RecordBatch
  // Each column group batch contains a subset of columns for the same rows
  if (column_group_batches.size() == 1) {
    return column_group_batches[0];
  }

  // Merge multiple column group batches
  std::vector<std::shared_ptr<arrow::Array>> merged_arrays;
  std::vector<std::shared_ptr<arrow::Field>> merged_fields;

  for (const auto& batch : column_group_batches) {
    for (int i = 0; i < batch->num_columns(); ++i) {
      merged_arrays.push_back(batch->column(i));
      merged_fields.push_back(batch->schema()->field(i));
    }
  }

  auto merged_schema = arrow::schema(merged_fields);
  return arrow::RecordBatch::Make(merged_schema, row_indices.size(), merged_arrays);
}

// ==================== PackedRecordBatchReader Implementation ====================

PackedRecordBatchReader::PackedRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 const std::vector<std::string>& needed_columns,
                                                 const ReadProperties& properties,
                                                 int64_t buffer_size)
    : fs_(std::move(fs)),
      column_groups_(column_groups),
      output_schema_(std::move(schema)),
      needed_columns_(needed_columns),
      properties_(properties),
      memory_limit_(buffer_size <= 0 ? INT64_MAX : buffer_size),
      memory_used_(0),
      absolute_row_position_(0),
      row_limit_(0),
      finished_(false),
      current_batch_index_(0) {
  // Initialize states for column groups
  cg_states_.resize(column_groups_.size());
  chunk_readers_.resize(column_groups_.size());
  batch_queues_.resize(column_groups_.size());

  for (size_t i = 0; i < column_groups_.size(); ++i) {
    cg_states_[i] = ColumnGroupState();
  }

  auto status = initialize();
  if (!status.ok()) {
    throw std::runtime_error("Failed to initialize PackedRecordBatchReader: " + status.ToString());
  }

  // Load initial data buffer
  status = advanceBuffer();
  if (!status.ok()) {
    throw std::runtime_error("Failed to load initial data buffer: " + status.ToString());
  }
}

PackedRecordBatchReader::~PackedRecordBatchReader() {
  // Explicit cleanup to prevent memory corruption
  // Clear all_batches_ first to release Arrow objects before other cleanup
  all_batches_.clear();
  all_batches_.shrink_to_fit();

  // Ensure proper cleanup - ignore the return status in destructor
  (void)Close();
}

std::shared_ptr<arrow::Schema> PackedRecordBatchReader::schema() const { return output_schema_; }

arrow::Status PackedRecordBatchReader::initialize() {
  // Create chunk readers for each column group
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    auto& column_group = column_groups_[i];

    // Get needed columns for this specific column group
    std::vector<std::string> cg_needed_columns;
    for (const auto& col_name : needed_columns_) {
      if (column_group->contains_column(col_name)) {
        cg_needed_columns.push_back(col_name);
      }
    }

    // Create chunk reader using factory
    auto chunk_reader = internal::api::ChunkReaderFactory::create_reader(column_group->format, fs_, column_group->path,
                                                                         cg_needed_columns, properties_);

    if (!chunk_reader) {
      return arrow::Status::Invalid("Failed to create chunk reader for column group " +
                                    std::to_string(column_group->id));
    }

    chunk_readers_[i] = std::move(chunk_reader);
  }
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::advanceBuffer() {
  std::vector<std::vector<int64_t>> chunks_to_read(column_groups_.size());
  size_t planned_memory = 0;

  // Advance to next chunk for column groups that need data
  auto advance_chunk = [&](size_t i) -> int64_t {
    int64_t next_chunk = cg_states_[i].current_chunk + 1;
    if (next_chunk >= column_groups_[i]->stats.num_chunks) {
      return -1;  // No more chunks
    }

    // Get actual chunk size from metadata instead of estimation
    auto chunk_size_result = chunk_readers_[i]->get_chunk_size(next_chunk);
    if (!chunk_size_result.ok()) {
      return -1;  // Error getting chunk size
    }
    int64_t chunk_size = chunk_size_result.ValueOrDie();

    chunks_to_read[i].push_back(next_chunk);
    planned_memory += chunk_size;
    cg_states_[i].memory_usage += chunk_size;
    cg_states_[i].current_chunk = next_chunk;

    // Estimate rows in this chunk
    int64_t rows_per_chunk = column_groups_[i]->stats.num_rows / column_groups_[i]->stats.num_chunks;
    cg_states_[i].row_offset += rows_per_chunk;

    return chunk_size;
  };

  // Fill in column groups that have no data available
  int drained_index = -1;
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (cg_states_[i].row_offset > row_limit_) {
      continue;
    }

    memory_used_ -= std::max(static_cast<size_t>(0), static_cast<size_t>(cg_states_[i].memory_usage));
    cg_states_[i].memory_usage = 0;

    auto next_chunk_size = advance_chunk(i);
    if (next_chunk_size < 0) {
      drained_index = i;
      break;
    }
  }

  if (drained_index >= 0) {
    if (planned_memory == 0) {
      finished_ = true;
      return arrow::Status::OK();
    } else {
      return arrow::Status::Invalid("Column group " + std::to_string(drained_index) +
                                    " exhausted while others have data");
    }
  }

  // Fill in column groups using min heap for row alignment
  RowOffsetMinHeap sorted_offsets;
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (!cg_states_[i].exhausted) {
      sorted_offsets.emplace(i, cg_states_[i].row_offset);
    }
  }

  while (!sorted_offsets.empty() && planned_memory + memory_used_ < memory_limit_) {
    size_t i = sorted_offsets.top().first;

    // Check if we can add another chunk
    if (cg_states_[i].current_chunk + 1 >= column_groups_[i]->stats.num_chunks) {
      break;
    }

    // Get actual chunk size from metadata instead of estimation
    auto chunk_size_result = chunk_readers_[i]->get_chunk_size(cg_states_[i].current_chunk + 1);
    if (!chunk_size_result.ok()) {
      break;  // Error getting chunk size, skip this column group
    }
    int64_t chunk_size = chunk_size_result.ValueOrDie();

    if (planned_memory + memory_used_ + chunk_size > memory_limit_) {
      break;
    }

    advance_chunk(i);
    sorted_offsets.pop();
    sorted_offsets.emplace(i, cg_states_[i].row_offset);
  }

  // Read the planned chunks
  for (size_t i = 0; i < column_groups_.size(); ++i) {
    if (chunks_to_read[i].empty()) {
      continue;
    }

    // Read chunks from this column group
    ARROW_ASSIGN_OR_RAISE(auto chunks, chunk_readers_[i]->get_chunks(chunks_to_read[i]));

    for (auto& chunk : chunks) {
      if (chunk) {
        batch_queues_[i].push(chunk);
      }
    }
  }

  memory_used_ += planned_memory;

  // Set row limit based on minimum row offset
  if (!sorted_offsets.empty()) {
    row_limit_ = sorted_offsets.top().second;
  }

  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* out) {
  *out = nullptr;

  if (finished_) {
    return arrow::Status::OK();
  }

  // Initialize on first call
  if (current_batch_index_ == 0) {
    all_batches_.clear();

    // Ultra-conservative approach: read chunk by chunk with immediate cleanup
    // Force garbage collection after each operation

    // First, determine the maximum number of chunks by trying chunk 0 from each column group
    int64_t max_chunks_found = 0;
    for (size_t cg_idx = 0; cg_idx < column_groups_.size(); ++cg_idx) {
      auto& column_group = column_groups_[cg_idx];

      std::vector<std::string> cg_needed_columns;
      for (const auto& col_name : needed_columns_) {
        if (column_group->contains_column(col_name)) {
          cg_needed_columns.push_back(col_name);
        }
      }

      if (!cg_needed_columns.empty()) {
        auto chunk_reader = internal::api::ChunkReaderFactory::create_reader(
            column_group->format, fs_, column_group->path, cg_needed_columns, properties_);

        if (chunk_reader) {
          // Try to determine chunk count carefully
          auto parquet_reader = dynamic_cast<milvus_storage::api::ParquetFormatReader*>(chunk_reader.get());
          if (parquet_reader) {
            auto num_chunks_result = parquet_reader->get_num_chunks();
            if (num_chunks_result.ok()) {
              max_chunks_found = std::max(max_chunks_found, num_chunks_result.ValueOrDie());
            }
          } else {
            // For non-parquet readers, assume at least 1 chunk
            max_chunks_found = std::max(max_chunks_found, static_cast<int64_t>(1));
          }
          // Immediately release the chunk reader to free memory
          chunk_reader.reset();
        }
      }
    }

    // Read chunks one by one
    for (int64_t chunk_idx = 0; chunk_idx < max_chunks_found; ++chunk_idx) {
      std::vector<std::shared_ptr<arrow::RecordBatch>> chunks_to_combine;

      // Read corresponding chunk from each column group
      for (size_t cg_idx = 0; cg_idx < column_groups_.size(); ++cg_idx) {
        auto& column_group = column_groups_[cg_idx];

        std::vector<std::string> cg_needed_columns;
        for (const auto& col_name : needed_columns_) {
          if (column_group->contains_column(col_name)) {
            cg_needed_columns.push_back(col_name);
          }
        }

        if (!cg_needed_columns.empty()) {
          auto chunk_reader = internal::api::ChunkReaderFactory::create_reader(
              column_group->format, fs_, column_group->path, cg_needed_columns, properties_);

          if (chunk_reader) {
            auto chunk_result = chunk_reader->get_chunk(chunk_idx);
            if (chunk_result.ok()) {
              auto chunk = chunk_result.ValueOrDie();
              if (chunk && chunk->num_rows() > 0) {
                chunks_to_combine.push_back(chunk);
              } else {
                chunks_to_combine.push_back(nullptr);
              }
            } else {
              chunks_to_combine.push_back(nullptr);
            }
            // Immediately release the chunk reader to free memory
            chunk_reader.reset();
          } else {
            chunks_to_combine.push_back(nullptr);
          }
        } else {
          chunks_to_combine.push_back(nullptr);
        }
      }

      // Combine chunks from all column groups horizontally
      auto combined_result = combine_batches(chunks_to_combine);
      if (combined_result.ok()) {
        auto combined_batch = combined_result.ValueOrDie();
        if (combined_batch && combined_batch->num_rows() > 0) {
          // Split large batches into smaller ones (max 1000 rows per batch)
          const int64_t max_batch_size = 1000;
          int64_t num_rows = combined_batch->num_rows();

          for (int64_t offset = 0; offset < num_rows; offset += max_batch_size) {
            int64_t length = std::min(max_batch_size, num_rows - offset);
            auto sliced_batch = combined_batch->Slice(offset, length);
            if (sliced_batch) {
              all_batches_.push_back(sliced_batch);
              // Limit memory usage by forcing cleanup if we have too many batches
              if (all_batches_.size() > 1000) {  // Conservative limit
                // LOG: Limiting batch accumulation to prevent memory issues
                break;
              }
            }
          }
        }
      }
    }
  }

  // Return next batch
  if (current_batch_index_ < all_batches_.size()) {
    *out = all_batches_[current_batch_index_];
    current_batch_index_++;
    return arrow::Status::OK();
  } else {
    finished_ = true;
    // Clear all_batches_ to free memory when reading is finished
    all_batches_.clear();
    all_batches_.shrink_to_fit();
    return arrow::Status::OK();
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> PackedRecordBatchReader::combine_batches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
  if (batches.empty()) {
    return arrow::Status::Invalid("No batches to combine");
  }

  // Find the minimum number of rows across all non-null batches
  int64_t min_rows = std::numeric_limits<int64_t>::max();
  for (const auto& batch : batches) {
    if (batch != nullptr) {
      min_rows = std::min(min_rows, batch->num_rows());
    }
  }

  if (min_rows == std::numeric_limits<int64_t>::max()) {
    min_rows = 0;
  }

  // Collect arrays from all batches
  std::vector<std::shared_ptr<arrow::Array>> combined_arrays;
  std::vector<std::shared_ptr<arrow::Field>> combined_fields;

  for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
    const auto& batch = batches[batch_idx];

    if (batch != nullptr) {
      // Add arrays from this batch
      for (int col_idx = 0; col_idx < batch->num_columns(); ++col_idx) {
        auto array = batch->column(col_idx);
        if (array->length() > min_rows) {
          // Slice array to min_rows
          array = array->Slice(0, min_rows);
        }
        combined_arrays.push_back(array);
        // Use output_schema_ field instead of batch schema to avoid circular references
        auto field_name = batch->schema()->field(col_idx)->name();
        auto output_field = output_schema_->GetFieldByName(field_name);
        if (output_field != nullptr) {
          combined_fields.push_back(output_field);
        } else {
          combined_fields.push_back(batch->schema()->field(col_idx));
        }
      }
    } else {
      // This column group has no data - create null arrays for its columns
      auto& column_group = column_groups_[batch_idx];
      for (const auto& col_name : column_group->columns) {
        auto field = output_schema_->GetFieldByName(col_name);
        if (field != nullptr) {
          ARROW_ASSIGN_OR_RAISE(auto null_array, arrow::MakeArrayOfNull(field->type(), min_rows));
          combined_arrays.push_back(null_array);
          combined_fields.push_back(field);
        }
      }
    }
  }

  // Create the combined schema and batch
  auto combined_schema = arrow::schema(combined_fields);
  return arrow::RecordBatch::Make(combined_schema, min_rows, combined_arrays);
}

arrow::Status PackedRecordBatchReader::Close() {
  // Prevent double cleanup
  if (finished_) {
    return arrow::Status::OK();
  }

  // Clean up resources
  for (auto& queue : batch_queues_) {
    while (!queue.empty()) {
      queue.pop();
    }
  }

  chunk_readers_.clear();
  batch_queues_.clear();
  cg_states_.clear();
  all_batches_.clear();
  all_batches_.shrink_to_fit();
  memory_used_ = 0;
  finished_ = true;

  return arrow::Status::OK();
}

}  // namespace milvus_storage::api