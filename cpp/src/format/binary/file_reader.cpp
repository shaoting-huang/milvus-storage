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

#include "milvus-storage/format/binary/file_reader.h"
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/logging.h>
#include <set>

namespace milvus_storage {

BinaryFileReader::BinaryFileReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                   const std::string& file_path,
                                   const std::vector<std::string>& column_names)
    : ChunkReader(fs, file_path, column_names) {}

arrow::Status BinaryFileReader::Init() {
  if (!fs_) {
    return arrow::Status::Invalid("File system is null");
  }

  auto result = fs_->OpenInputFile(file_path_);
  if (!result.ok()) {
    return result.status();
  }
  input_stream_ = result.ValueOrDie();

  ARROW_RETURN_NOT_OK(LoadMetadata());

  return arrow::Status::OK();
}

arrow::Status BinaryFileReader::LoadMetadata() {
  // Read metadata offset from the end of file
  auto size_result = input_stream_->GetSize();
  if (!size_result.ok()) {
    return size_result.status();
  }

  int64_t file_size = size_result.ValueOrDie();
  int64_t metadata_offset_pos = file_size - sizeof(int64_t);

  auto read_result = input_stream_->ReadAt(metadata_offset_pos, sizeof(int64_t));
  if (!read_result.ok()) {
    return read_result.status();
  }

  auto metadata_offset_buffer = read_result.ValueOrDie();
  int64_t metadata_offset = *reinterpret_cast<const int64_t*>(metadata_offset_buffer->data());

  // Read chunk information
  auto chunk_data_result = input_stream_->ReadAt(metadata_offset, sizeof(int64_t));
  if (!chunk_data_result.ok()) {
    return chunk_data_result.status();
  }

  auto chunk_count_buffer = chunk_data_result.ValueOrDie();
  int64_t num_chunks = *reinterpret_cast<const int64_t*>(chunk_count_buffer->data());

  chunks_.reserve(num_chunks);
  int64_t offset = metadata_offset + sizeof(int64_t);

  for (int64_t i = 0; i < num_chunks; ++i) {
    ChunkInfo chunk;

    auto chunk_info_result = input_stream_->ReadAt(offset, 3 * sizeof(int64_t));
    if (!chunk_info_result.ok()) {
      return chunk_info_result.status();
    }

    auto chunk_info_buffer = chunk_info_result.ValueOrDie();
    const int64_t* chunk_data = reinterpret_cast<const int64_t*>(chunk_info_buffer->data());

    chunk.offset = chunk_data[0];
    chunk.length = chunk_data[1];
    chunk.row_count = chunk_data[2];

    chunks_.push_back(chunk);
    offset += 3 * sizeof(int64_t);
  }

  // Read schema
  auto schema_size_result = input_stream_->ReadAt(offset, sizeof(int64_t));
  if (!schema_size_result.ok()) {
    return schema_size_result.status();
  }

  auto schema_size_buffer = schema_size_result.ValueOrDie();
  int64_t schema_size = *reinterpret_cast<const int64_t*>(schema_size_buffer->data());
  offset += sizeof(int64_t);

  auto schema_data_result = input_stream_->ReadAt(offset, schema_size);
  if (!schema_data_result.ok()) {
    return schema_data_result.status();
  }

  auto schema_buffer = schema_data_result.ValueOrDie();
  arrow::io::BufferReader buffer_reader(schema_buffer);
  auto schema_result = arrow::ipc::ReadSchema(&buffer_reader, nullptr);
  if (!schema_result.ok()) {
    return schema_result.status();
  }

  schema_ = schema_result.ValueOrDie();
  offset += schema_size;

  // Read key-value metadata
  auto kv_count_result = input_stream_->ReadAt(offset, sizeof(int64_t));
  if (!kv_count_result.ok()) {
    return kv_count_result.status();
  }

  auto kv_count_buffer = kv_count_result.ValueOrDie();
  int64_t kv_count = *reinterpret_cast<const int64_t*>(kv_count_buffer->data());
  offset += sizeof(int64_t);

  // Skip reading key-value pairs for now (can be implemented if needed)
  for (int64_t i = 0; i < kv_count; ++i) {
    // Read key size and skip key
    auto key_size_result = input_stream_->ReadAt(offset, sizeof(int64_t));
    if (!key_size_result.ok()) {
      return key_size_result.status();
    }
    auto key_size_buffer = key_size_result.ValueOrDie();
    int64_t key_size = *reinterpret_cast<const int64_t*>(key_size_buffer->data());
    offset += sizeof(int64_t) + key_size;

    // Read value size and skip value
    auto value_size_result = input_stream_->ReadAt(offset, sizeof(int64_t));
    if (!value_size_result.ok()) {
      return value_size_result.status();
    }
    auto value_size_buffer = value_size_result.ValueOrDie();
    int64_t value_size = *reinterpret_cast<const int64_t*>(value_size_buffer->data());
    offset += sizeof(int64_t) + value_size;
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> BinaryFileReader::Next() {
  if (current_chunk_index_ >= static_cast<int64_t>(chunks_.size())) {
    return nullptr;  // End of file
  }

  const auto& chunk = chunks_[current_chunk_index_];

  // Read chunk length
  auto length_result = input_stream_->ReadAt(chunk.offset, sizeof(int64_t));
  if (!length_result.ok()) {
    return length_result.status();
  }

  auto length_buffer = length_result.ValueOrDie();
  int64_t chunk_length = *reinterpret_cast<const int64_t*>(length_buffer->data());

  // Read chunk data
  auto data_result = input_stream_->ReadAt(chunk.offset + sizeof(int64_t), chunk_length);
  if (!data_result.ok()) {
    return data_result.status();
  }

  auto data_buffer = data_result.ValueOrDie();
  std::string serialized_data(reinterpret_cast<const char*>(data_buffer->data()), data_buffer->size());

  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(DeserializeBatch(serialized_data, batch));

  current_chunk_index_++;
  return batch;
}

arrow::Status BinaryFileReader::DeserializeBatch(const std::string& data, std::shared_ptr<arrow::RecordBatch>& batch) {
  auto buffer = arrow::Buffer::FromString(data);
  arrow::io::BufferReader buffer_reader(buffer);
  std::shared_ptr<arrow::ipc::RecordBatchReader> reader;

  auto result = arrow::ipc::RecordBatchStreamReader::Open(&buffer_reader);
  if (!result.ok()) {
    return result.status();
  }

  reader = result.ValueOrDie();
  auto batch_result = reader->Next();
  if (!batch_result.ok()) {
    return batch_result.status();
  }

  batch = batch_result.ValueOrDie();
  if (!batch) {
    return arrow::Status::Invalid("Failed to deserialize batch");
  }

  // Filter columns if needed
  if (!needed_columns_.empty()) {
    std::vector<std::shared_ptr<arrow::Array>> filtered_columns;
    std::vector<std::shared_ptr<arrow::Field>> filtered_fields;

    for (const auto& col_name : needed_columns_) {
      int col_index = batch->schema()->GetFieldIndex(col_name);
      if (col_index >= 0) {
        filtered_columns.push_back(batch->column(col_index));
        filtered_fields.push_back(batch->schema()->field(col_index));
      }
    }

    if (!filtered_columns.empty()) {
      auto filtered_schema = arrow::schema(filtered_fields);
      batch = arrow::RecordBatch::Make(filtered_schema, batch->num_rows(), filtered_columns);
    }
  }

  return arrow::Status::OK();
}

arrow::Status BinaryFileReader::SeekToChunk(int64_t chunk_index) {
  if (chunk_index < 0 || chunk_index >= static_cast<int64_t>(chunks_.size())) {
    return arrow::Status::Invalid("Invalid chunk index");
  }

  current_chunk_index_ = chunk_index;
  return arrow::Status::OK();
}

arrow::Status BinaryFileReader::Close() {
  if (closed_) {
    return arrow::Status::OK();
  }

  if (input_stream_) {
    ARROW_RETURN_NOT_OK(input_stream_->Close());
  }

  closed_ = true;
  return arrow::Status::OK();
}

arrow::Result<std::vector<int64_t>> BinaryFileReader::get_chunk_indices(const std::vector<int64_t>& row_indices) const {
  std::vector<int64_t> chunk_indices;
  std::set<int64_t> unique_chunks;

  for (int64_t row_index : row_indices) {
    int64_t current_row = 0;
    for (size_t i = 0; i < chunks_.size(); ++i) {
      current_row += chunks_[i].row_count;
      if (row_index < current_row) {
        unique_chunks.insert(i);
        break;
      }
    }
  }

  chunk_indices.assign(unique_chunks.begin(), unique_chunks.end());
  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> BinaryFileReader::get_chunk(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));

  const auto& chunk = chunks_[chunk_index];

  // Read chunk length
  auto length_result = input_stream_->ReadAt(chunk.offset, sizeof(int64_t));
  if (!length_result.ok()) {
    return length_result.status();
  }

  auto length_buffer = length_result.ValueOrDie();
  int64_t chunk_length = *reinterpret_cast<const int64_t*>(length_buffer->data());

  // Read chunk data
  auto data_result = input_stream_->ReadAt(chunk.offset + sizeof(int64_t), chunk_length);
  if (!data_result.ok()) {
    return data_result.status();
  }

  auto data_buffer = data_result.ValueOrDie();
  std::string serialized_data(reinterpret_cast<const char*>(data_buffer->data()), data_buffer->size());

  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(const_cast<BinaryFileReader*>(this)->DeserializeBatch(serialized_data, batch));

  // Filter columns if needed
  if (!needed_columns_.empty()) {
    std::vector<std::shared_ptr<arrow::Array>> filtered_columns;
    std::vector<std::shared_ptr<arrow::Field>> filtered_fields;

    for (const auto& col_name : needed_columns_) {
      int col_index = batch->schema()->GetFieldIndex(col_name);
      if (col_index >= 0) {
        filtered_columns.push_back(batch->column(col_index));
        filtered_fields.push_back(batch->schema()->field(col_index));
      }
    }

    if (!filtered_columns.empty()) {
      auto filtered_schema = arrow::schema(filtered_fields);
      batch = arrow::RecordBatch::Make(filtered_schema, batch->num_rows(), filtered_columns);
    }
  }

  return batch;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> BinaryFileReader::get_chunks(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism) const {
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  batches.reserve(chunk_indices.size());

  for (int64_t chunk_index : chunk_indices) {
    auto batch_result = get_chunk(chunk_index);
    if (!batch_result.ok()) {
      return batch_result.status();
    }
    batches.push_back(batch_result.ValueOrDie());
  }

  return batches;
}

arrow::Result<int64_t> BinaryFileReader::get_chunk_size(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));
  return chunks_[chunk_index].length;
}

arrow::Result<int64_t> BinaryFileReader::get_chunk_row_num(int64_t chunk_index) const {
  ARROW_RETURN_NOT_OK(validate_chunk_index(chunk_index));
  return chunks_[chunk_index].row_count;
}

arrow::Status BinaryFileReader::validate_chunk_index(int64_t chunk_index) const {
  if (chunk_index < 0 || chunk_index >= static_cast<int64_t>(chunks_.size())) {
    return arrow::Status::Invalid("Invalid chunk index: ", chunk_index);
  }
  return arrow::Status::OK();
}

}  // namespace milvus_storage