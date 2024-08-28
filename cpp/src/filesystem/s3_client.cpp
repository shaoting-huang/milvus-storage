

#include <arrow/status.h>
#include <aws/core/Aws.h>
#include <aws/core/Region.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/DeleteObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListBucketsResult.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/ObjectCannedACL.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/UploadPartRequest.h>
#include "filesystem/s3_client.h"

#include <arrow/result.h>

#include "common/log.h"
#include "filesystem/s3_internal.h"

using ::Aws::Client::AWSError;
using ::Aws::S3::S3Errors;

namespace milvus_storage {


// An AWS RetryStrategy that wraps a provided arrow::fs::S3RetryStrategy
class WrappedRetryStrategy : public Aws::Client::RetryStrategy {
 public:
  explicit WrappedRetryStrategy(const std::shared_ptr<S3RetryStrategy>& s3_retry_strategy)
      : s3_retry_strategy_(s3_retry_strategy) {}

  bool ShouldRetry(const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
                   long attempted_retries) const override {  // NOLINT runtime/int
    S3RetryStrategy::AWSErrorDetail detail = ErrorToDetail(error);
    return s3_retry_strategy_->ShouldRetry(detail,
                                           static_cast<int64_t>(attempted_retries));
  }

  long CalculateDelayBeforeNextRetry(  // NOLINT runtime/int
      const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
      long attempted_retries) const override {  // NOLINT runtime/int
    S3RetryStrategy::AWSErrorDetail detail = ErrorToDetail(error);
    return static_cast<long>(  // NOLINT runtime/int
        s3_retry_strategy_->CalculateDelayBeforeNextRetry(
            detail, static_cast<int64_t>(attempted_retries)));
  }

 private:
  template <typename ErrorType>
  static S3RetryStrategy::AWSErrorDetail ErrorToDetail(
      const Aws::Client::AWSError<ErrorType>& error) {
    S3RetryStrategy::AWSErrorDetail detail;
    detail.error_type = static_cast<int>(error.GetErrorType());
    detail.message = std::string(FromAwsString(error.GetMessage()));
    detail.exception_name = std::string(FromAwsString(error.GetExceptionName()));
    detail.should_retry = error.ShouldRetry();
    return detail;
  }

  std::shared_ptr<S3RetryStrategy> s3_retry_strategy_;
};


// To get a bucket's region, we must extract the "x-amz-bucket-region" header
// from the response to a HEAD bucket request.
// Unfortunately, the S3Client APIs don't let us access the headers of successful
// responses.  So we have to cook a AWS request and issue it ourselves.

arrow::Result<std::string> S3Client::GetBucketRegion(const S3Model::HeadBucketRequest& request) {
  auto uri = GeneratePresignedUrl(request.GetBucket(),
                                  /*key=*/"", Aws::Http::HttpMethod::HTTP_HEAD);
  // NOTE: The signer region argument isn't passed here, as there's no easy
  // way of computing it (the relevant method is private).
  auto outcome = MakeRequest(uri, request, Aws::Http::HttpMethod::HTTP_HEAD,
                              Aws::Auth::SIGV4_SIGNER);
  const auto code = outcome.IsSuccess() ? outcome.GetResult().GetResponseCode()
                                        : outcome.GetError().GetResponseCode();
  const auto& headers = outcome.IsSuccess()
                            ? outcome.GetResult().GetHeaderValueCollection()
                            : outcome.GetError().GetResponseHeaders();

  const auto it = headers.find(ToAwsString("x-amz-bucket-region"));
  if (it == headers.end()) {
    if (code == Aws::Http::HttpResponseCode::NOT_FOUND) {
      return arrow::Status::IOError("Bucket '", request.GetBucket(), "' not found");
    } else if (!outcome.IsSuccess()) {
      return ErrorToStatus(std::forward_as_tuple("When resolving region for bucket '",
                                                  request.GetBucket(), "': "),
                            outcome.GetError());
    } else {
      return arrow::Status::IOError("When resolving region for bucket '", request.GetBucket(),
                              "': missing 'x-amz-bucket-region' header in response");
    }
  }
  return std::string(FromAwsString(it->second));
}

arrow::Result<std::string> S3Client::GetBucketRegion(const std::string& bucket) {
  S3Model::HeadBucketRequest req;
  req.SetBucket(ToAwsString(bucket));
  return GetBucketRegion(req);
}

S3Model::CompleteMultipartUploadOutcome S3Client::CompleteMultipartUploadWithErrorFixup(
    S3Model::CompleteMultipartUploadRequest&& request) const {
  // CompletedMultipartUpload can return a 200 OK response with an error
  // encoded in the response body, in which case we should either retry
  // or propagate the error to the user (see
  // https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html).
  //
  // Unfortunately the AWS SDK doesn't detect such situations but lets them
  // return successfully (see https://github.com/aws/aws-sdk-cpp/issues/658).
  //
  // We work around the issue by registering a DataReceivedEventHandler
  // which parses the XML response for embedded errors.

  std::optional<AWSError<Aws::Client::CoreErrors>> aws_error;

  auto handler = [&](const Aws::Http::HttpRequest* http_req,
                      Aws::Http::HttpResponse* http_resp,
                      long long) {  // NOLINT runtime/int
    auto& stream = http_resp->GetResponseBody();
    const auto pos = stream.tellg();
    const auto doc = Aws::Utils::Xml::XmlDocument::CreateFromXmlStream(stream);
    // Rewind stream for later
    stream.clear();
    stream.seekg(pos);

    if (doc.WasParseSuccessful()) {
      auto root = doc.GetRootElement();
      if (!root.IsNull()) {
        // Detect something that looks like an abnormal CompletedMultipartUpload
        // response.
        if (root.GetName() != "CompleteMultipartUploadResult" ||
            !root.FirstChild("Error").IsNull() || !root.FirstChild("Errors").IsNull()) {
          // Make sure the error marshaller doesn't see a 200 OK
          http_resp->SetResponseCode(
              Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR);
          aws_error = GetErrorMarshaller()->Marshall(*http_resp);
          // Rewind stream for later
          stream.clear();
          stream.seekg(pos);
        }
      }
    }
  };

  request.SetDataReceivedEventHandler(std::move(handler));

  // We don't have access to the configured AWS retry strategy
  // (m_retryStrategy is a private member of AwsClient), so don't use that.
  std::unique_ptr<Aws::Client::RetryStrategy> retry_strategy;
  if (s3_retry_strategy_) {
    retry_strategy.reset(new WrappedRetryStrategy(s3_retry_strategy_));
  } else {
    // Note that DefaultRetryStrategy, unlike StandardRetryStrategy,
    // has empty definitions for RequestBookkeeping() and GetSendToken(),
    // which simplifies the code below.
    retry_strategy.reset(new Aws::Client::DefaultRetryStrategy());
  }

  for (int32_t retries = 0;; retries++) {
    aws_error.reset();
    auto outcome = Aws::S3::S3Client::S3Client::CompleteMultipartUpload(request);
    if (!outcome.IsSuccess()) {
      // Error returned in HTTP headers (or client failure)
      return outcome;
    }
    if (!aws_error.has_value()) {
      // Genuinely successful outcome
      return outcome;
    }

    const bool should_retry = retry_strategy->ShouldRetry(*aws_error, retries);

    LOG_STORAGE_WARNING_
        << "CompletedMultipartUpload got error embedded in a 200 OK response: "
        << aws_error->GetExceptionName() << " (\"" << aws_error->GetMessage()
        << "\"), retry = " << should_retry;

    if (!should_retry) {
      break;
    }
    const auto delay = std::chrono::milliseconds(
        retry_strategy->CalculateDelayBeforeNextRetry(*aws_error, retries));
    std::this_thread::sleep_for(delay);
  }

  DCHECK(aws_error.has_value());
  auto s3_error = AWSError<S3Errors>(std::move(aws_error).value());
  return S3Model::CompleteMultipartUploadOutcome(std::move(s3_error));
}

}  // namespace milvus_storage