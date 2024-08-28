
#pragma once

#include <string>
#include <aws/core/Aws.h>
#include <aws/core/utils/StringUtils.h>
#include "arrow/status.h"

namespace milvus_storage {

inline Aws::String ToAwsString(const std::string& s) {
  return Aws::String(s.begin(), s.end());
}

inline std::string_view FromAwsString(const Aws::String& s) {
  return {s.data(), s.length()};
}

inline Aws::String ToURLEncodedAwsString(const std::string& s) {
  return Aws::Utils::StringUtils::URLEncode(s.data());
}

template <typename ErrorType>
arrow::Status ErrorToStatus(const std::string& prefix,
                     const Aws::Client::AWSError<ErrorType>& error) {
  // XXX Handle fine-grained error types
  // See
  // https://sdk.amazonaws.com/cpp/api/LATEST/namespace_aws_1_1_s3.html#ae3f82f8132b619b6e91c88a9f1bde371
  return arrow::Status::IOError(prefix, "AWS Error [code ",
                         static_cast<int>(error.GetErrorType()),
                         "]: ", error.GetMessage());
}

template <typename ErrorType, typename... Args>
arrow::Status ErrorToStatus(const std::tuple<Args&...>& prefix,
                     const Aws::Client::AWSError<ErrorType>& error) {
  std::stringstream ss;
  return ErrorToStatus(ss.str(), error);
}

template <typename ErrorType>
arrow::Status ErrorToStatus(const Aws::Client::AWSError<ErrorType>& error) {
  return ErrorToStatus(std::string(), error);
}

} // namespace milvus_storage