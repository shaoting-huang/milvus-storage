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

#include <filesystem/s3fs.h>
#include "arrow/filesystem/path_util.h"
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <filesystem/s3_internal.h>
#include "arrow/util/future.h"

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

static const char kSep = '/';

static constexpr int64_t kMinimumPartUpload = 5 * 1024 * 1024;

using namespace arrow::fs;



namespace milvus_storage {

struct S3Path {
  std::string full_path;
  std::string bucket;
  std::string key;
  std::vector<std::string> key_parts;

  static Result<S3Path> FromString(const std::string& s) {
    if (fs::internal::IsLikelyUri(s)) {
      return Status::Invalid(
          "Expected an S3 object path of the form 'bucket/key...', got a URI: '", s, "'");
    }
    const auto src = fs::internal::RemoveTrailingSlash(s);
    auto first_sep = src.find_first_of(kSep);
    if (first_sep == 0) {
      return Status::Invalid("Path cannot start with a separator ('", s, "')");
    }
    if (first_sep == std::string::npos) {
      return S3Path{std::string(src), std::string(src), "", {}};
    }
    S3Path path;
    path.full_path = std::string(src);
    path.bucket = std::string(src.substr(0, first_sep));
    path.key = std::string(src.substr(first_sep + 1));
    path.key_parts = fs::internal::SplitAbstractPath(path.key);
    RETURN_NOT_OK(Validate(path));
    return path;
  }

  static Status Validate(const S3Path& path) {
    auto result = fs::internal::ValidateAbstractPathParts(path.key_parts);
    if (!result.ok()) {
      return Status::Invalid(result.message(), " in path ", path.full_path);
    } else {
      return result;
    }
  }

  Aws::String ToAwsString() const {
    Aws::String res(bucket.begin(), bucket.end());
    res.reserve(bucket.size() + key.size() + 1);
    res += kSep;
    res.append(key.begin(), key.end());
    return res;
  }

  Aws::String ToURLEncodedAwsString() const {
    // URL-encode individual parts, not the '/' separator
    Aws::String res;
    res += milvus_storage::ToURLEncodedAwsString(bucket);
    for (const auto& part : key_parts) {
      res += kSep;
      res += milvus_storage::ToURLEncodedAwsString(part);
    }
    return res;
  }

  S3Path parent() const {
    DCHECK(!key_parts.empty());
    auto parent = S3Path{"", bucket, "", key_parts};
    parent.key_parts.pop_back();
    parent.key = fs::internal::JoinAbstractPath(parent.key_parts);
    parent.full_path = parent.bucket + kSep + parent.key;
    return parent;
  }

  bool has_parent() const { return !key.empty(); }

  bool empty() const { return bucket.empty() && key.empty(); }

  bool operator==(const S3Path& other) const {
    return bucket == other.bucket && key == other.key;
  }
};


template <typename ObjectRequest>
struct ObjectMetadataSetter {
  using Setter = std::function<Status(const std::string& value, ObjectRequest* req)>;

  static std::unordered_map<std::string, Setter> GetSetters() {
    return {{"ACL", CannedACLSetter()},
            {"Cache-Control", StringSetter(&ObjectRequest::SetCacheControl)},
            {"Content-Type", StringSetter(&ObjectRequest::SetContentType)},
            {"Content-Language", StringSetter(&ObjectRequest::SetContentLanguage)},
            {"Expires", DateTimeSetter(&ObjectRequest::SetExpires)}};
  }

 private:
  static Setter StringSetter(void (ObjectRequest::*req_method)(Aws::String&&)) {
    return [req_method](const std::string& v, ObjectRequest* req) {
      (req->*req_method)(ToAwsString(v));
      return Status::OK();
    };
  }

  static Setter DateTimeSetter(
      void (ObjectRequest::*req_method)(Aws::Utils::DateTime&&)) {
    return [req_method](const std::string& v, ObjectRequest* req) {
      (req->*req_method)(
          Aws::Utils::DateTime(v.data(), Aws::Utils::DateFormat::ISO_8601));
      return Status::OK();
    };
  }

  static Setter CannedACLSetter() {
    return [](const std::string& v, ObjectRequest* req) {
      ARROW_ASSIGN_OR_RAISE(auto acl, ParseACL(v));
      req->SetACL(acl);
      return Status::OK();
    };
  }

  static Result<Aws::S3::Model::ObjectCannedACL> ParseACL(const std::string& v) {
    if (v.empty()) {
      return Aws::S3::Model::ObjectCannedACL::NOT_SET;
    }
    auto acl = Aws::S3::Model::ObjectCannedACLMapper::GetObjectCannedACLForName(ToAwsString(v));
    if (acl == Aws::S3::Model::ObjectCannedACL::NOT_SET) {
      // XXX This actually never happens, as the AWS SDK dynamically
      // expands the enum range using Aws::GetEnumOverflowContainer()
      return Status::Invalid("Invalid S3 canned ACL: '", v, "'");
    }
    return acl;
  }
};


template <typename ObjectRequest>
Status SetObjectMetadata(const std::shared_ptr<const KeyValueMetadata>& metadata,
                         ObjectRequest* req) {
  static auto setters = ObjectMetadataSetter<ObjectRequest>::GetSetters();

  DCHECK_NE(metadata, nullptr);
  const auto& keys = metadata->keys();
  const auto& values = metadata->values();

  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = setters.find(keys[i]);
    if (it != setters.end()) {
      RETURN_NOT_OK(it->second(values[i], req));
    }
  }
  return Status::OK();
}



// An OutputStream that writes to a S3 object
class ObjectOutputStream final : public io::OutputStream {
 protected:
  struct UploadState;

 public:
  ObjectOutputStream(std::shared_ptr<Aws::S3::S3Client> client, const io::IOContext& io_context,
                     const S3Path& path, const S3Options& options,
                     const std::shared_ptr<const KeyValueMetadata>& metadata)
      : client_(std::move(client)),
        io_context_(io_context),
        path_(path),
        metadata_(metadata),
        default_metadata_(options.default_metadata),
        background_writes_(options.background_writes) {}

  ~ObjectOutputStream() override {}

  Status Init() {
    // Initiate the multi-part upload
    Aws::S3::Model::CreateMultipartUploadRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    if (metadata_ && metadata_->size() != 0) {
      RETURN_NOT_OK(SetObjectMetadata(metadata_, &req));
    } else if (default_metadata_ && default_metadata_->size() != 0) {
      RETURN_NOT_OK(SetObjectMetadata(default_metadata_, &req));
    }

    // If we do not set anything then the SDK will default to application/xml
    // which confuses some tools (https://github.com/apache/arrow/issues/11934)
    // So we instead default to application/octet-stream which is less misleading
    if (!req.ContentTypeHasBeenSet()) {
      req.SetContentType("application/octet-stream");
    }

    auto outcome = client_->CreateMultipartUpload(req);
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(
          std::forward_as_tuple("When initiating multiple part upload for key '",
                                path_.key, "' in bucket '", path_.bucket, "': "),
          outcome.GetError());
    }
    upload_id_ = outcome.GetResult().GetUploadId();
    upload_state_ = std::make_shared<UploadState>();
    closed_ = false;
    return Status::OK();
  }

  Status Abort() override {
    if (closed_) {
      return Status::OK();
    }

    Aws::S3::Model::AbortMultipartUploadRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    req.SetUploadId(upload_id_);

    auto outcome = client_->AbortMultipartUpload(req);
    if (!outcome.IsSuccess()) {
      return ErrorToStatus(
          std::forward_as_tuple("When aborting multiple part upload for key '", path_.key,
                                "' in bucket '", path_.bucket, "': "),
          outcome.GetError());
    }
    current_part_.reset();
    client_ = nullptr;
    closed_ = true;
    return Status::OK();
  }

  // OutputStream interface

  Status Close() override {
    auto fut = CloseAsync();
    return fut.status();
  }

  Future<> CloseAsync() override {
    if (closed_) return Status::OK();

    if (current_part_) {
      // Upload last part
      RETURN_NOT_OK(CommitCurrentPart());
    }

    // S3 mandates at least one part, upload an empty one if necessary
    if (part_number_ == 1) {
      RETURN_NOT_OK(UploadPart("", 0));
    }

    // Wait for in-progress uploads to finish (if async writes are enabled)
    return FlushAsync().Then([this]() {
      // At this point, all part uploads have finished successfully
      DCHECK_GT(part_number_, 1);
      DCHECK_EQ(upload_state_->completed_parts.size(),
                static_cast<size_t>(part_number_ - 1));

      Aws::S3::Model::CompletedMultipartUpload completed_upload;
      completed_upload.SetParts(upload_state_->completed_parts);
      Aws::S3::Model::CompleteMultipartUploadRequest req;
      req.SetBucket(ToAwsString(path_.bucket));
      req.SetKey(ToAwsString(path_.key));
      req.SetUploadId(upload_id_);
      req.SetMultipartUpload(std::move(completed_upload));

      auto outcome = client_->CompleteMultipartUploadWithErrorFixup(std::move(req));
      if (!outcome.IsSuccess()) {
        return ErrorToStatus(
            std::forward_as_tuple("When completing multiple part upload for key '",
                                  path_.key, "' in bucket '", path_.bucket, "': "),
            outcome.GetError());
      }

      client_ = nullptr;
      closed_ = true;
      return Status::OK();
    });
  }

  bool closed() const override { return closed_; }

  Result<int64_t> Tell() const override {
    if (closed_) {
      return Status::Invalid("Operation on closed stream");
    }
    return pos_;
  }

  Status Write(const std::shared_ptr<Buffer>& buffer) override {
    return DoWrite(buffer->data(), buffer->size(), buffer);
  }

  Status Write(const void* data, int64_t nbytes) override {
    return DoWrite(data, nbytes);
  }

  Status DoWrite(const void* data, int64_t nbytes,
                 std::shared_ptr<Buffer> owned_buffer = nullptr) {
    if (closed_) {
      return Status::Invalid("Operation on closed stream");
    }

    if (!current_part_ && nbytes >= part_upload_threshold_) {
      // No current part and data large enough, upload it directly
      // (without copying if the buffer is owned)
      RETURN_NOT_OK(UploadPart(data, nbytes, owned_buffer));
      pos_ += nbytes;
      return Status::OK();
    }
    // Can't upload data on its own, need to buffer it
    if (!current_part_) {
      ARROW_ASSIGN_OR_RAISE(
          current_part_,
          io::BufferOutputStream::Create(part_upload_threshold_, io_context_.pool()));
      current_part_size_ = 0;
    }
    RETURN_NOT_OK(current_part_->Write(data, nbytes));
    pos_ += nbytes;
    current_part_size_ += nbytes;

    if (current_part_size_ >= part_upload_threshold_) {
      // Current part large enough, upload it
      RETURN_NOT_OK(CommitCurrentPart());
    }

    return Status::OK();
  }

  Status Flush() override {
    auto fut = FlushAsync();
    return fut.status();
  }

  Future<> FlushAsync() {
    if (closed_) {
      return Status::Invalid("Operation on closed stream");
    }
    // Wait for background writes to finish
    std::unique_lock<std::mutex> lock(upload_state_->mutex);
    return upload_state_->pending_parts_completed;
  }

  // Upload-related helpers

  Status CommitCurrentPart() {
    ARROW_ASSIGN_OR_RAISE(auto buf, current_part_->Finish());
    current_part_.reset();
    current_part_size_ = 0;
    return UploadPart(buf);
  }

  Status UploadPart(std::shared_ptr<Buffer> buffer) {
    return UploadPart(buffer->data(), buffer->size(), buffer);
  }

  Status UploadPart(const void* data, int64_t nbytes,
                    std::shared_ptr<Buffer> owned_buffer = nullptr) {
    Aws::S3::Model::UploadPartRequest req;
    req.SetBucket(ToAwsString(path_.bucket));
    req.SetKey(ToAwsString(path_.key));
    req.SetUploadId(upload_id_);
    req.SetPartNumber(part_number_);
    req.SetContentLength(nbytes);

    if (!background_writes_) {
      req.SetBody(std::make_shared<StringViewStream>(data, nbytes));
      auto outcome = client_->UploadPart(req);
      if (!outcome.IsSuccess()) {
        return UploadPartError(req, outcome);
      } else {
        AddCompletedPart(upload_state_, part_number_, outcome.GetResult());
      }
    } else {
      // If the data isn't owned, make an immutable copy for the lifetime of the closure
      if (owned_buffer == nullptr) {
        ARROW_ASSIGN_OR_RAISE(owned_buffer, AllocateBuffer(nbytes, io_context_.pool()));
        memcpy(owned_buffer->mutable_data(), data, nbytes);
      } else {
        DCHECK_EQ(data, owned_buffer->data());
        DCHECK_EQ(nbytes, owned_buffer->size());
      }
      req.SetBody(
          std::make_shared<StringViewStream>(owned_buffer->data(), owned_buffer->size()));

      {
        std::unique_lock<std::mutex> lock(upload_state_->mutex);
        if (upload_state_->parts_in_progress++ == 0) {
          upload_state_->pending_parts_completed = Future<>::Make();
        }
      }
      auto client = client_;
      ARROW_ASSIGN_OR_RAISE(auto fut, SubmitIO(io_context_, [client, req]() {
                              return client->UploadPart(req);
                            }));
      // The closure keeps the buffer and the upload state alive
      auto state = upload_state_;
      auto part_number = part_number_;
      auto handler = [owned_buffer, state, part_number,
                      req](const Result<Aws::S3::Model::UploadPartOutcome>& result) -> void {
        HandleUploadOutcome(state, part_number, req, result);
      };
      fut.AddCallback(std::move(handler));
    }

    ++part_number_;
    // With up to 10000 parts in an upload (S3 limit), a stream writing chunks
    // of exactly 5MB would be limited to 50GB total.  To avoid that, we bump
    // the upload threshold every 100 parts.  So the pattern is:
    // - part 1 to 99: 5MB threshold
    // - part 100 to 199: 10MB threshold
    // - part 200 to 299: 15MB threshold
    // ...
    // - part 9900 to 9999: 500MB threshold
    // So the total size limit is 2475000MB or ~2.4TB, while keeping manageable
    // chunk sizes and avoiding too much buffering in the common case of a small-ish
    // stream.  If the limit's not enough, we can revisit.
    if (part_number_ % 100 == 0) {
      part_upload_threshold_ += kMinimumPartUpload;
    }

    return Status::OK();
  }

  static void HandleUploadOutcome(const std::shared_ptr<UploadState>& state,
                                  int part_number, const Aws::S3::Model::UploadPartRequest& req,
                                  const Result<Aws::S3::Model::UploadPartOutcome>& result) {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (!result.ok()) {
      state->status &= result.status();
    } else {
      const auto& outcome = *result;
      if (!outcome.IsSuccess()) {
        state->status &= UploadPartError(req, outcome);
      } else {
        AddCompletedPart(state, part_number, outcome.GetResult());
      }
    }
    // Notify completion
    if (--state->parts_in_progress == 0) {
      state->pending_parts_completed.MarkFinished(state->status);
    }
  }

  static void AddCompletedPart(const std::shared_ptr<UploadState>& state, int part_number,
                               const Aws::S3::Model::UploadPartResult& result) {
    Aws::S3::Model::CompletedPart part;
    // Append ETag and part number for this uploaded part
    // (will be needed for upload completion in Close())
    part.SetPartNumber(part_number);
    part.SetETag(result.GetETag());
    int slot = part_number - 1;
    if (state->completed_parts.size() <= static_cast<size_t>(slot)) {
      state->completed_parts.resize(slot + 1);
    }
    DCHECK(!state->completed_parts[slot].PartNumberHasBeenSet());
    state->completed_parts[slot] = std::move(part);
  }

  static Status UploadPartError(const Aws::S3::Model::UploadPartRequest& req,
                                const Aws::S3::Model::UploadPartOutcome& outcome) {
    return ErrorToStatus(
        std::forward_as_tuple("When uploading part for key '", req.GetKey(),
                              "' in bucket '", req.GetBucket(), "': "),
        outcome.GetError());
  }

 protected:
  std::shared_ptr<Aws::S3::S3Client> client_;
  const io::IOContext io_context_;
  const S3Path path_;
  const std::shared_ptr<const KeyValueMetadata> metadata_;
  const std::shared_ptr<const KeyValueMetadata> default_metadata_;
  const bool background_writes_;

  Aws::String upload_id_;
  bool closed_ = true;
  int64_t pos_ = 0;
  int32_t part_number_ = 1;
  std::shared_ptr<io::BufferOutputStream> current_part_;
  int64_t current_part_size_ = 0;
  int64_t part_upload_threshold_ = kMinimumPartUpload;

  // This struct is kept alive through background writes to avoid problems
  // in the completion handler.
  struct UploadState {
    std::mutex mutex;
    Aws::Vector<Aws::S3::Model::CompletedPart> completed_parts;
    int64_t parts_in_progress = 0;
    Status status;
    Future<> pending_parts_completed = Future<>::MakeFinished(Status::OK());
  };
  std::shared_ptr<UploadState> upload_state_;
};

Result<std::shared_ptr<io::OutputStream>> MilvusS3FileSystem::OpenOutputStream(const std::string& s, const std::shared_ptr<const KeyValueMetadata>& metadata) {
  ARROW_RETURN_NOT_OK(fs::internal::AssertNoTrailingSlash(s));
  ARROW_ASSIGN_OR_RAISE(auto path, S3Path::FromString(s));
  RETURN_NOT_OK(ValidateFilePath(path));

  auto ptr = std::make_shared<ObjectOutputStream>(impl_->client_, io_context(), path,
                                                  impl_->options(), metadata);
  RETURN_NOT_OK(ptr->Init());
  return ptr;
}

} // namespace milvus_storage