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

#include "milvus-storage/format/binary/file_writer.h"
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/logging.h>

namespace milvus_storage {

BinaryFileWriter::BinaryFileWriter(std::shared_ptr<arrow::Schema> schema,
                                   std::shared_ptr<arrow::fs::FileSystem> fs,
                                   const std::string& file_path,
                                   const StorageConfig& storage_config)
    : fs_(std::move(fs)), schema_(std::move(schema)), file_path_(file_path), storage_config_(storage_config) {
  kv_metadata_ = std::make_shared<arrow::KeyValueMetadata>();
}

arrow::Status BinaryFileWriter::Init() {
  if (!fs_) {
    return arrow::Status::Invalid("File system is null");
  }

  if (!schema_) {
    return arrow::Status::Invalid("Schema is null");
  }

  auto result = fs_->OpenOutputStream(file_path_);
  if (!result.ok()) {
    return result.status();
  }
  output_stream_ = result.ValueOrDie();

  return arrow::Status::OK();
}

arrow::Status BinaryFileWriter::Write(const std::shared_ptr<arrow::RecordBatch> record) {
  if (!record) {
    return arrow::Status::Invalid("Record batch is null");
  }

  if (closed_) {
    return arrow::Status::Invalid("Writer is already closed");
  }

  return WriteChunk(record);
}

arrow::Status BinaryFileWriter::WriteChunk(const std::shared_ptr<arrow::RecordBatch>& batch) {
  int64_t start_offset = bytes_written_;

  std::string serialized_data;
  ARROW_RETURN_NOT_OK(SerializeBatch(batch, serialized_data));

  // Write chunk length first (8 bytes)
  int64_t chunk_length = serialized_data.size();
  ARROW_RETURN_NOT_OK(output_stream_->Write(&chunk_length, sizeof(chunk_length)));

  // Write the serialized batch data
  ARROW_RETURN_NOT_OK(output_stream_->Write(serialized_data.data(), chunk_length));

  // Update statistics
  ChunkInfo chunk_info;
  chunk_info.offset = start_offset;
  chunk_info.length = chunk_length + sizeof(chunk_length);  // Include length header
  chunk_info.row_count = batch->num_rows();

  chunks_.push_back(chunk_info);
  total_rows_ += batch->num_rows();
  bytes_written_ += chunk_info.length;

  return arrow::Status::OK();
}

arrow::Status BinaryFileWriter::SerializeBatch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                               std::string& serialized_data) {
  // Create a buffer output stream using proper factory method
  ARROW_ASSIGN_OR_RAISE(auto stream, arrow::io::BufferOutputStream::Create());

  // Create IPC stream writer with schema first, then write batch
  arrow::ipc::IpcWriteOptions options = arrow::ipc::IpcWriteOptions::Defaults();
  auto writer_result = arrow::ipc::MakeStreamWriter(stream.get(), batch->schema(), options);
  if (!writer_result.ok()) {
    return writer_result.status();
  }
  auto writer = writer_result.ValueOrDie();

  // Write the batch
  ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
  ARROW_RETURN_NOT_OK(writer->Close());

  // Get the buffer
  auto buffer_result = stream->Finish();
  if (!buffer_result.ok()) {
    return buffer_result.status();
  }
  auto buffer = buffer_result.ValueOrDie();

  serialized_data = std::string(reinterpret_cast<const char*>(buffer->data()), buffer->size());

  return arrow::Status::OK();
}

arrow::Status BinaryFileWriter::Flush() {
  if (!output_stream_) {
    return arrow::Status::Invalid("Output stream is not initialized");
  }
  return output_stream_->Flush();
}

arrow::Status BinaryFileWriter::Close() {
  if (closed_) {
    return arrow::Status::OK();
  }

  // Write metadata at the end of file
  ARROW_RETURN_NOT_OK(WriteMetadata());

  if (output_stream_) {
    ARROW_RETURN_NOT_OK(output_stream_->Close());
  }

  closed_ = true;
  return arrow::Status::OK();
}

arrow::Status BinaryFileWriter::WriteMetadata() {
  // Serialize chunk information
  // Format: [num_chunks][chunk1_offset][chunk1_length][chunk1_row_count]...[metadata_offset]

  int64_t num_chunks = chunks_.size();
  ARROW_RETURN_NOT_OK(output_stream_->Write(&num_chunks, sizeof(num_chunks)));

  for (const auto& chunk : chunks_) {
    ARROW_RETURN_NOT_OK(output_stream_->Write(&chunk.offset, sizeof(chunk.offset)));
    ARROW_RETURN_NOT_OK(output_stream_->Write(&chunk.length, sizeof(chunk.length)));
    ARROW_RETURN_NOT_OK(output_stream_->Write(&chunk.row_count, sizeof(chunk.row_count)));
  }

  // Write schema
  auto schema_result = arrow::ipc::SerializeSchema(*schema_, arrow::default_memory_pool());
  if (!schema_result.ok()) {
    return schema_result.status();
  }

  auto schema_buffer = schema_result.ValueOrDie();
  int64_t schema_size = schema_buffer->size();
  ARROW_RETURN_NOT_OK(output_stream_->Write(&schema_size, sizeof(schema_size)));
  ARROW_RETURN_NOT_OK(output_stream_->Write(schema_buffer->data(), schema_size));

  // Write key-value metadata
  if (kv_metadata_ && kv_metadata_->size() > 0) {
    int64_t kv_count = kv_metadata_->size();
    ARROW_RETURN_NOT_OK(output_stream_->Write(&kv_count, sizeof(kv_count)));

    for (int64_t i = 0; i < kv_count; ++i) {
      const std::string& key = kv_metadata_->key(i);
      const std::string& value = kv_metadata_->value(i);

      int64_t key_size = key.size();
      int64_t value_size = value.size();

      ARROW_RETURN_NOT_OK(output_stream_->Write(&key_size, sizeof(key_size)));
      ARROW_RETURN_NOT_OK(output_stream_->Write(key.data(), key_size));
      ARROW_RETURN_NOT_OK(output_stream_->Write(&value_size, sizeof(value_size)));
      ARROW_RETURN_NOT_OK(output_stream_->Write(value.data(), value_size));
    }
  } else {
    int64_t kv_count = 0;
    ARROW_RETURN_NOT_OK(output_stream_->Write(&kv_count, sizeof(kv_count)));
  }

  // Write metadata offset at the very end
  int64_t metadata_start_offset = bytes_written_;
  ARROW_RETURN_NOT_OK(output_stream_->Write(&metadata_start_offset, sizeof(metadata_start_offset)));

  return arrow::Status::OK();
}

arrow::Status BinaryFileWriter::AppendKVMetadata(const std::string& key, const std::string& value) {
  if (!kv_metadata_) {
    kv_metadata_ = std::make_shared<arrow::KeyValueMetadata>();
  }

  auto updated_metadata = kv_metadata_->Copy();
  updated_metadata->Append(key, value);
  kv_metadata_ = updated_metadata;

  return arrow::Status::OK();
}

arrow::Status BinaryFileWriter::AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata) {
  for (const auto& [key, value] : metadata) {
    ARROW_RETURN_NOT_OK(AppendKVMetadata(key, value));
  }
  return arrow::Status::OK();
}

}  // namespace milvus_storage