
#pragma once


#include <string>
#include <aws/core/Aws.h>
#include <aws/core/utils/StringUtils.h>

namespace milvus_storage {

inline Aws::String ToAwsString(const std::string& s) {
  // Direct construction of Aws::String from std::string doesn't work because
  // it uses a specific Allocator class.
  return Aws::String(s.begin(), s.end());
}

inline Aws::String ToURLEncodedAwsString(const std::string& s) {
  return Aws::Utils::StringUtils::URLEncode(s.data());
}

} // namespace milvus_storage