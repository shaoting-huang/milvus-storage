// Copyright 2024 Zilliz
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

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/localfs.h>
#include <gtest/gtest.h>
#include <arrow/api.h>
#include <packed/writer.h>
#include <parquet/properties.h>
#include <packed/reader.h>
#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include "test_util.h"
#include "common/fs_util.h"
#include "packed_test_base.h"

#include "common/macro.h"

using namespace std;
namespace milvus_storage {

class PackedIntegrationTest : public PackedTestBase {
  protected:
  PackedIntegrationTest() : props_(*parquet::default_writer_properties()) {}

  void SetUp() override {
    ASSERT_AND_ASSIGN(fs_, BuildFileSystem("file:///tmp/", &file_path_));
    SetUpCommonData();
    props_ = *parquet::default_writer_properties();
    writer_memory_ = 1024 * 1024;  // 1 MB memory for writing
    reader_memory_ = 1024 * 1024;  // 1 MB memory for reading
  }

  size_t writer_memory_;
  size_t reader_memory_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string file_path_;
  parquet::WriterProperties props_;
  const int bath_size = 1000;
};

TEST_F(PackedIntegrationTest, WriteAndRead) {
  PackedWriter writer(writer_memory_, schema_, *fs_, file_path_, props_);
  EXPECT_TRUE(writer.Init(record_batch_).ok());
  for (int i = 1; i < bath_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::set<int> needed_columns = {0, 1, 2};
  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(1, 0),
      ColumnOffset(1, 1),
  };

  auto paths = std::vector<std::string>{file_path_ + "/0", file_path_ + "/1"};

  // after writing, the column of large_str is in 0th file, and the last int64 columns are in 1st file
  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("str", arrow::utf8()),
      arrow::field("int32", arrow::int32()),
      arrow::field("int64", arrow::int64()),
  };
  auto new_schema = arrow::schema(fields);

  PackedRecordBatchReader pr(*fs_, paths, new_schema, column_offsets, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_AND_ARROW_ASSIGN(auto combined_table, table->CombineChunks());
  ASSERT_STATUS_OK(pr.Close());

  auto str_res = std::dynamic_pointer_cast<arrow::StringArray>(combined_table->GetColumnByName("str")->chunk(0));
  std::vector<std::string> strs;
  strs.reserve(str_res->length());
  for (int i = 0; i < str_res->length(); ++i) {
    strs.push_back(str_res->GetString(i));
  }

  auto int32_res = std::dynamic_pointer_cast<arrow::Int32Array>(combined_table->GetColumnByName("int32")->chunk(0));
  std::vector<int32_t> int32s;
  int32s.reserve(int32_res->length());
  for (int i = 0; i < int32_res->length(); ++i) {
    int32s.push_back(int32_res->Value(i));
  }

  auto int64_res = std::dynamic_pointer_cast<arrow::Int64Array>(combined_table->GetColumnByName("int64")->chunk(0));
  std::vector<int64_t> int64s;
  int64s.reserve(int64_res->length());
  for (int i = 0; i < int64_res->length(); ++i) {
    int64s.push_back(int64_res->Value(i));
  }

  std::vector<std::string> expected_strs;
  std::vector<int32_t> expected_int32s;
  std::vector<int64_t> expected_int64s;
  expected_strs.reserve(str_res->length());
  expected_int32s.reserve(str_res->length());
  expected_int64s.reserve(str_res->length());
  for (int i = 0; i < bath_size; ++i) {
    expected_strs.insert(std::end(expected_strs), std::begin(str_values), std::end(str_values));
    expected_int32s.insert(std::end(expected_int32s), std::begin(int32_values), std::end(int32_values));
    expected_int64s.insert(std::end(expected_int64s), std::begin(int64_values), std::end(int64_values));
  }

  ASSERT_EQ(strs, expected_strs);
  ASSERT_EQ(int32s, expected_int32s);
  ASSERT_EQ(int64s, expected_int64s);
}

}  // namespace milvus_storage