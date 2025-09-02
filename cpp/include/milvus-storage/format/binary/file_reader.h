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
#include <fstream>
#include "arrow/filesystem/filesystem.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/format/binary/file_writer.h"

namespace milvus_storage {

class BinaryFileReader : public milvus_storage::api::ChunkReader {
  public:
  BinaryFileReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                   const std::string& file_path,
                   const std::vector<std::string>& column_names);

  arrow::Status Init();

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next();

  arrow::Status Close();

  arrow::Status SeekToChunk(int64_t chunk_index);

  int64_t total_chunks() const { return chunks_.size(); }

  // ChunkReader interface implementations
  arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) const override;
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) const override;
  arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(const std::vector<int64_t>& chunk_indices,
                                                                             int64_t parallelism = 1) const override;
  arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) const override;
  arrow::Result<int64_t> get_chunk_row_num(int64_t chunk_index) const override;

  protected:
  arrow::Status validate_chunk_index(int64_t chunk_index) const override;

  private:
  arrow::Status LoadMetadata();
  arrow::Status DeserializeBatch(const std::string& data, std::shared_ptr<arrow::RecordBatch>& batch);

  std::shared_ptr<arrow::io::RandomAccessFile> input_stream_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<ChunkInfo> chunks_;
  int64_t current_chunk_index_ = 0;
  bool closed_ = false;
};

}  // namespace milvus_storage