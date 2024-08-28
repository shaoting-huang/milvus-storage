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
#pragma once

#include "arrow/filesystem/s3fs.h"
#include <memory>
#include <string>
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include <aws/core/utils/StringUtils.h>


using namespace arrow;

namespace milvus_storage {

class MilvusS3FileSystem : public fs::S3FileSystem {
  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata) override;
};


}  // namespace milvus_storage