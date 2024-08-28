
#pragma once

#include <memory>
#include <string>
#include <aws/s3/S3Client.h>
#include <vector>

#include "arrow/filesystem/filesystem.h"
#include "arrow/util/macros.h"
#include "arrow/util/uri.h"

namespace S3Model = Aws::S3::Model;

namespace milvus_storage {


/// Pure virtual class for describing custom S3 retry strategies
class ARROW_EXPORT S3RetryStrategy {
 public:
  virtual ~S3RetryStrategy() = default;

  /// Simple struct where each field corresponds to a field in Aws::Client::AWSError
  struct AWSErrorDetail {
    /// Corresponds to AWSError::GetErrorType()
    int error_type;
    /// Corresponds to AWSError::GetMessage()
    std::string message;
    /// Corresponds to AWSError::GetExceptionName()
    std::string exception_name;
    /// Corresponds to AWSError::ShouldRetry()
    bool should_retry;
  };
  /// Returns true if the S3 request resulting in the provided error should be retried.
  virtual bool ShouldRetry(const AWSErrorDetail& error, int64_t attempted_retries) = 0;
  /// Returns the time in milliseconds the S3 client should sleep for until retrying.
  virtual int64_t CalculateDelayBeforeNextRetry(const AWSErrorDetail& error,
                                                int64_t attempted_retries) = 0;
  /// Returns a stock AWS Default retry strategy.
  static std::shared_ptr<S3RetryStrategy> GetAwsDefaultRetryStrategy(
      int64_t max_attempts);
  /// Returns a stock AWS Standard retry strategy.
  static std::shared_ptr<S3RetryStrategy> GetAwsStandardRetryStrategy(
      int64_t max_attempts);
};

class S3Client : public Aws::S3::S3Client {
  public:
  using Aws::S3::S3Client::S3Client;

  arrow::Result<std::string> GetBucketRegion(const S3Model::HeadBucketRequest& request);

  arrow::Result<std::string> GetBucketRegion(const std::string& bucket);

  S3Model::CompleteMultipartUploadOutcome CompleteMultipartUploadWithErrorFixup(
      S3Model::CompleteMultipartUploadRequest&& request) const;
  
  private:
  std::shared_ptr<S3RetryStrategy> s3_retry_strategy_;
};

} 