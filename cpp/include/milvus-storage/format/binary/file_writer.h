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
#include <fstream>
#include <vector>
#include "arrow/filesystem/filesystem.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/util/key_value_metadata.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/common/status.h"

namespace milvus_storage {

struct ChunkInfo {
  int64_t offset;
  int64_t length;
  int64_t row_count;
};

class BinaryFileWriter : public milvus_storage::api::ColumnGroupWriter {
  public:
  BinaryFileWriter(std::shared_ptr<arrow::Schema> schema,
                   std::shared_ptr<arrow::fs::FileSystem> fs,
                   const std::string& file_path,
                   const StorageConfig& storage_config);

  arrow::Status Init() override;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Status Close() override;

  arrow::Status AppendKVMetadata(const std::string& key, const std::string& value) override;

  arrow::Status AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata) override;

  int64_t count() const override { return total_rows_; }
  int64_t bytes_written() const override { return bytes_written_; }
  int64_t num_chunks() const override { return chunks_.size(); }

  private:
  arrow::Status WriteChunk(const std::shared_ptr<arrow::RecordBatch>& batch);
  arrow::Status WriteMetadata();
  arrow::Status SerializeBatch(const std::shared_ptr<arrow::RecordBatch>& batch, std::string& serialized_data);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  const std::string file_path_;
  const StorageConfig& storage_config_;

  std::shared_ptr<arrow::io::OutputStream> output_stream_;
  std::vector<ChunkInfo> chunks_;
  std::shared_ptr<arrow::KeyValueMetadata> kv_metadata_;
  int64_t total_rows_ = 0;
  int64_t bytes_written_ = 0;
  bool closed_ = false;
};

}  // namespace milvus_storage