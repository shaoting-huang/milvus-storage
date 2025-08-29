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

#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/format/format_reader.h"

namespace internal::api {

// ==================== FormatWriterFactory Implementation ====================

std::unique_ptr<FormatWriter> FormatWriterFactory::create_writer(
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    std::shared_ptr<arrow::Schema> schema,
    const milvus_storage::api::WriteProperties& properties) {
  if (!column_group) {
    throw std::runtime_error("Column group cannot be null");
  }

  // Extract values from column group
  const auto& format = column_group->format;
  const auto& file_path = column_group->path;

  switch (format) {
    case milvus_storage::api::FileFormat::PARQUET:
      return std::make_unique<ParquetFormatWriter>(std::move(fs), file_path, std::move(schema), properties);

    default:
      throw std::runtime_error("Unsupported file format: " + std::to_string(static_cast<int>(format)) +
                               ". Only PARQUET is supported.");
  }
}

// ==================== ChunkReaderFactory Implementation ====================

std::unique_ptr<milvus_storage::api::ChunkReader> ChunkReaderFactory::create_reader(
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::ReadProperties& properties) {
  if (!column_group) {
    throw std::runtime_error("Column group cannot be null");
  }

  const auto& format = column_group->format;
  const auto& file_path = column_group->path;

  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (column_group->contains_column(col_name)) {
      filtered_columns.push_back(col_name);
    }
  }

  switch (format) {
    case milvus_storage::api::FileFormat::PARQUET:
      return std::make_unique<milvus_storage::api::ParquetFormatReader>(fs, file_path, std::move(filtered_columns),
                                                                        properties);

    default:
      throw std::runtime_error("Unsupported file format: " + std::to_string(static_cast<int>(format)) +
                               ". Only PARQUET is supported.");
  }
}

}  // namespace internal::api