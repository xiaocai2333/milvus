// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package externalspec

import (
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/pkg/v2/util/merr"
)

// File formats supported by external collections. Mirror of LOON_FORMAT_*
// in the C++ FFI layer — keep the two in sync.
const (
	FormatParquet      = "parquet"
	FormatLanceTable   = "lance-table"
	FormatVortex       = "vortex"
	FormatIcebergTable = "iceberg-table"
)

// ExternalSpec represents the parsed external collection specification
type ExternalSpec struct {
	Format     string            `json:"format"`                // one of Format* constants
	Columns    []string          `json:"columns"`               // optional: specific columns to load
	Extfs      map[string]string `json:"extfs,omitempty"`       // optional: extfs config overrides (non-sensitive only)
	SnapshotID *int64            `json:"snapshot_id,omitempty"` // Iceberg snapshot ID (required for iceberg-table)
}

// supportedFormats lists the file formats supported for external collections
var supportedFormats = map[string]bool{
	FormatParquet:      true,
	FormatLanceTable:   true,
	FormatVortex:       true,
	FormatIcebergTable: true,
}

// allowedExtfsKeys lists extfs configuration keys that can be passed via ExternalSpec.
// Credential keys are allowed because cross-bucket scenarios require different
// authentication from the main Milvus storage (e.g., local MinIO vs external AWS S3).
// `address` and `bucket_name` are allowed so users can write standard AWS-style
// URIs (`s3://<bucket>/<key>`) and let the endpoint come from extfs instead of
// the URI host. When `address` is present in extfs, BuildExtfsOverrides flips
// to AWS-style URI interpretation (URI host = bucket).
//
// AWS-style activation: URI interpretation switches to AWS-style whenever
// there is an effective endpoint (see EffectiveAddressFromExtfs). That means
// `cloud_provider` + `region` alone — without an explicit `address` — also
// activate the flip, via Tier-2 derivation. Users who write a Milvus-form
// URI (`s3://<endpoint>/<bucket>/...`) alongside provider/region MUST ensure
// the URI host matches the derived endpoint exactly (the equality guard in
// NormalizeExternalSource and BuildExtfsOverrides relies on a string match),
// otherwise the URI will be silently rewritten as AWS-style and the original
// endpoint will be misinterpreted as a bucket. Self-hosted S3-compatible
// targets (MinIO, R2, Backblaze) must NOT set cloud_provider/region that
// conflicts with the true storage — use `address` explicitly instead.
// Note: these values are stored in etcd as part of CollectionSchema.
var allowedExtfsKeys = map[string]bool{
	"use_iam":          true,
	"use_ssl":          true,
	"use_virtual_host": true,
	"region":           true,
	"cloud_provider":   true,
	"iam_endpoint":     true,
	"storage_type":     true,
	"ssl_ca_cert":      true,
	"access_key_id":    true,
	"access_key_value": true,
	"role_arn":         true,
	"session_name":     true,
	"external_id":      true,
	"load_frequency":   true,
	"address":          true,
	"bucket_name":      true,
}

// booleanExtfsKeys lists extfs keys that only accept "true" or "false".
var booleanExtfsKeys = map[string]bool{
	"use_iam":          true,
	"use_ssl":          true,
	"use_virtual_host": true,
}

// allowedExternalSourceSchemes lists URL schemes accepted in ExternalSource.
// This is a defense-in-depth allowlist to prevent unvalidated SSRF / arbitrary
// endpoint injection at CreateCollection / RefreshExternalCollection time.
// Add new schemes here only after confirming the storage backend is supported.
//
// This list MUST stay in sync with the C++ segcore same-bucket prefix list in
// ChunkedSegmentSealedImpl.cpp: {aws://, s3://, minio://, gcs://, gs://} plus
// the loon FFI's supported scheme set. Any scheme that the storage layer
// understands must also be allowlisted here so that cross-bucket URIs written
// via those schemes are not rejected by Proxy/RootCoord with a misleading
// "scheme not allowed" error.
var allowedExternalSourceSchemes = map[string]bool{
	"s3":    true, // S3-compatible (AWS S3, MinIO, Tencent COS via s3, etc.)
	"s3a":   true, // Hadoop-style S3
	"aws":   true, // milvus-storage loon FFI native AWS prefix
	"minio": true, // milvus-storage loon FFI native MinIO prefix
	"oss":   true, // Aliyun OSS
	"gs":    true, // Google Cloud Storage
	"gcs":   true, // Google Cloud Storage (alias)
}

// secretExtfsKeys lists extfs keys whose values are sensitive credentials
// and MUST be redacted before logging or being persisted to user-visible
// surfaces (logs, error messages, audit trails). The values still flow
// through the FFI layer for actual storage authentication; this set only
// gates the redaction path used by RedactExternalSpec.
var secretExtfsKeys = map[string]bool{
	"access_key_id":    true,
	"access_key_value": true,
	"ssl_ca_cert":      true,
}

// ParseExternalSpec parses the JSON external spec string
func ParseExternalSpec(specStr string) (*ExternalSpec, error) {
	if specStr == "" {
		return &ExternalSpec{Format: FormatParquet}, nil // default
	}

	var spec ExternalSpec
	if err := json.Unmarshal([]byte(specStr), &spec); err != nil {
		return nil, merr.WrapErrParameterInvalidMsg("invalid external spec JSON: %s", err.Error())
	}

	if spec.Format == "" {
		spec.Format = FormatParquet // default format
	}

	if !supportedFormats[spec.Format] {
		return nil, merr.WrapErrParameterInvalidMsg("unsupported format %q, supported formats: %s",
			spec.Format, strings.Join(sortedKeys(supportedFormats), ", "))
	}

	for key, val := range spec.Extfs {
		if !allowedExtfsKeys[key] {
			return nil, merr.WrapErrParameterInvalidMsg("extfs key %q is not allowed; allowed keys: %s",
				key, strings.Join(sortedKeys(allowedExtfsKeys), ", "))
		}
		if booleanExtfsKeys[key] && val != "true" && val != "false" {
			return nil, merr.WrapErrParameterInvalidMsg("extfs key %q must be \"true\" or \"false\", got %q", key, val)
		}
	}

	return &spec, nil
}

// PropertyIcebergSnapshotID is the property key for the Iceberg snapshot ID.
const PropertyIcebergSnapshotID = "iceberg.snapshot_id"

// BuildFormatProperties returns format-specific properties for the milvus-storage FFI layer.
func (s *ExternalSpec) BuildFormatProperties() map[string]string {
	props := make(map[string]string)
	if s.Format == FormatIcebergTable && s.SnapshotID != nil {
		props[PropertyIcebergSnapshotID] = strconv.FormatInt(*s.SnapshotID, 10)
	}
	return props
}

// BuildExtfsOverrides returns the extfs map with the given prefix prepended to each key.
func (s *ExternalSpec) BuildExtfsOverrides(prefix string) map[string]string {
	if len(s.Extfs) == 0 {
		return nil
	}
	m := make(map[string]string, len(s.Extfs))
	for k, v := range s.Extfs {
		m[prefix+k] = v
	}
	return m
}

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ValidateExternalSource enforces a scheme allowlist on the user-supplied
// external_source URL. This is the first line of defense against SSRF /
// arbitrary endpoint injection: without it a malicious caller could point
// the storage layer at internal services (http://169.254.169.254/,
// gopher://, etc.). The C++ extfs layer has its own defense-in-depth check,
// but the Go-side allowlist must reject the request before any FFI call is
// made.
//
// Beyond the scheme check, this function also rejects shapes that are
// unambiguously unsafe or ambiguous:
//   - URLs with embedded userinfo (`s3://user:pass@bucket/...`): credentials
//     in URLs tend to leak into access logs, error messages, and audit
//     trails. Cross-bucket credentials MUST flow through the extfs path so
//     that they can be redacted by RedactExternalSpec.
//   - URLs that fail to parse or have empty scheme.
//
// Empty input is rejected — an external collection must have a source.
func ValidateExternalSource(source string) error {
	if source == "" {
		return fmt.Errorf("external_source is empty")
	}
	u, err := url.Parse(source)
	if err != nil {
		return fmt.Errorf("invalid external_source URL: %w", err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme == "" {
		// No scheme = same-bucket relative path (e.g. "my-data/parquet/").
		// This is the original format supported since Part1. The path is
		// relative to Milvus's own storage bucket and root path.
		return nil
	}
	if !allowedExternalSourceSchemes[scheme] {
		return fmt.Errorf("external_source scheme %q is not allowed; allowed schemes: %s",
			scheme, strings.Join(sortedKeys(allowedExternalSourceSchemes), ", "))
	}
	// Reject URL-embedded userinfo. The URL-parsing library strips the
	// credentials into u.User; they are then silently ignored by the
	// storage layer, but they travel through logs first. Force callers to
	// put credentials in the extfs map so RedactExternalSpec can scrub them.
	if u.User != nil {
		return fmt.Errorf("external_source must not embed credentials in the URL (use extfs.access_key_id / extfs.access_key_value instead)")
	}
	// Empty host allowed: `s3:///bucket/path` means same-endpoint cross-bucket
	// (uses Milvus's own storage endpoint; BuildExtfsOverrides skips address
	// override when host is empty).
	// Non-empty host carries two meanings depending on whether ExternalSpec
	// provides extfs.address:
	// - Milvus form (default, spec.extfs.address absent): host is the
	//   storage endpoint and the first path segment is the bucket, e.g.
	//   `s3://minio:9000/bucket/key` or `aws://s3.us-west-2.amazonaws.com/bkt/key`.
	// - AWS standard form (spec.extfs.address present): host is the bucket
	//   and the remainder of the path is the key, matching aws-cli shape
	//   `s3://bucket/key`. Endpoint taken from spec.extfs.address.
	return nil
}

// ValidateSourceAndSpec validates both the external_source URL and the
// external_spec JSON of a schema in one call. It is used by both Proxy
// and RootCoord (defense in depth) on the create-collection path.
// The returned error is already wrapped via merr.WrapErrParameterInvalid
// so callers can return it directly.
func ValidateSourceAndSpec(externalSource, externalSpec string) error {
	if err := ValidateExternalSource(externalSource); err != nil {
		return merr.WrapErrParameterInvalid("valid external_source", externalSource, err.Error())
	}
	if _, err := ParseExternalSpec(externalSpec); err != nil {
		return merr.WrapErrParameterInvalid("valid external_spec", "<redacted>", err.Error())
	}
	return nil
}

// DeriveEndpoint returns the storage endpoint URL for a (cloud_provider,
// region) pair when it can be mechanically derived from documented public
// endpoint patterns. It is the second-tier source of the effective endpoint
// used by NormalizeExternalSource / BuildExtfsOverrides when the caller
// omits `spec.extfs.address`.
//
// Coverage matches milvus-storage cpp/include/milvus-storage/filesystem/fs.h
// provider enum minus Azure: AWS, GCP, Aliyun OSS, Tencent COS, Huawei OBS.
// Azure Blob is intentionally excluded because its endpoint
// (`<account>.blob.core.windows.net`) carries an account-specific subdomain
// that cannot be reconstructed from (provider, region) alone — Azure users
// must provide extfs.address explicitly. Self-hosted S3-compatible targets
// (MinIO, R2, Backblaze B2, Wasabi) fall outside the provider enum entirely
// and likewise require an explicit address.
func DeriveEndpoint(cloudProvider, region string) (string, bool) {
	switch strings.ToLower(cloudProvider) {
	case "aws":
		if region == "" {
			return "", false
		}
		// AWS China regions live under a separate top-level domain.
		if strings.HasPrefix(region, "cn-") {
			return "https://s3." + region + ".amazonaws.com.cn", true
		}
		return "https://s3." + region + ".amazonaws.com", true
	case "gcp":
		// GCS uses a single global endpoint for all regions; region is a
		// bucket property, not part of the endpoint URL.
		return "https://storage.googleapis.com", true
	case "aliyun":
		if region == "" {
			return "", false
		}
		return "https://oss-" + region + ".aliyuncs.com", true
	case "tencent":
		if region == "" {
			return "", false
		}
		return "https://cos." + region + ".myqcloud.com", true
	case "huawei":
		if region == "" {
			return "", false
		}
		return "https://obs." + region + ".myhuaweicloud.com", true
	}
	return "", false
}

// EffectiveAddressFromExtfs returns the effective endpoint for a spec extfs
// map using the three-tier priority:
//  1. explicit spec.extfs.address
//  2. derived from (cloud_provider, region) via DeriveEndpoint
//  3. "" (signalling "no extfs-supplied endpoint — defer to URI host",
//     i.e. the pre-existing Milvus-form semantics)
//
// Extfs keys are unprefixed in this call (callers at the packed FFI layer
// that work with prefixed maps are expected to strip the prefix first).
func EffectiveAddressFromExtfs(extfs map[string]string) string {
	if len(extfs) == 0 {
		return ""
	}
	if a, ok := extfs["address"]; ok && a != "" {
		return a
	}
	if d, ok := DeriveEndpoint(extfs["cloud_provider"], extfs["region"]); ok {
		return d
	}
	return ""
}

// NormalizeExternalSource rewrites an AWS-style URI (`scheme://bucket/key`) into
// the canonical Milvus form (`scheme://endpoint/bucket/key`) whenever the
// caller supplies an endpoint through ExternalSpec.extfs — either explicitly
// via `address` or derivable from `cloud_provider + region`. The canonical
// form is what gets persisted into CollectionSchema, so every downstream
// consumer (DataCoord explore, DataNode fetch, QueryNode LoadColumnGroups,
// index build) sees a URI whose host is the storage endpoint — matching the
// URI shape produced by milvus-storage explore.
//
// Three-tier priority for the effective endpoint:
//  1. `spec.extfs.address` explicit (wins outright)
//  2. Derived from `spec.extfs.cloud_provider + region` for the 5 public
//     clouds milvus-storage supports with a region-based endpoint pattern
//  3. None of the above — URI is returned unchanged (Milvus-form semantics)
//
// Equality guard: if the URI host already equals the effective endpoint, the
// user is writing a Milvus-form URI that happens to redundantly carry
// cloud_provider + region in the spec. The guard avoids misinterpreting the
// endpoint as a bucket in that case and returns the source untouched, leaving
// the existing Milvus-form path to handle it.
//
// Relative paths, empty-host URIs, and URIs whose host is not a safe bucket
// name are returned as-is so downstream validators can surface a precise
// error rather than seeing a silently-mangled URI.
func NormalizeExternalSource(externalSource, externalSpec string) string {
	if externalSource == "" || externalSpec == "" {
		return externalSource
	}
	spec, err := ParseExternalSpec(externalSpec)
	if err != nil {
		return externalSource
	}
	return RewriteAWSStyleToMilvusForm(externalSource, EffectiveAddressFromExtfs(spec.Extfs))
}

// RewriteAWSStyleToMilvusForm is the shared core rewrite used by both
// NormalizeExternalSource (ExternalSpec JSON entry point) and the prefix-aware
// variant in internal/storagev2/packed. Given a pre-computed effective
// endpoint, it rewrites an AWS-style URI (`scheme://bucket/key`) into the
// canonical Milvus form (`scheme://endpoint/bucket/key`). All safety gates
// (empty endpoint, invalid URL, empty/unsafe host, equality guard) are
// applied inside this single function so the two entry points cannot drift.
func RewriteAWSStyleToMilvusForm(externalSource, effectiveAddr string) string {
	if effectiveAddr == "" {
		return externalSource
	}
	u, err := url.Parse(externalSource)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return externalSource
	}
	if !IsSafeURIHost(u.Host) {
		return externalSource
	}
	endpoint := StripURLScheme(effectiveAddr)
	if u.Host == endpoint {
		return externalSource
	}
	normalized := &url.URL{
		Scheme:   u.Scheme,
		Host:     endpoint,
		Path:     "/" + u.Host + u.Path,
		RawQuery: u.RawQuery,
		Fragment: u.Fragment,
	}
	return normalized.String()
}

// IsSafeURIHost accepts hosts composed only of RFC-3986 unreserved + `:` (port)
// + `-` + `.` characters. The whitelist is deliberately stricter than what
// url.Parse tolerates so the reassembled URI cannot carry characters that
// would alter parsing on the C++ side (bracketed IPv6 literals, userinfo,
// whitespace, percent-encodings we did not introduce ourselves).
// Accepting a whitelist also future-proofs the check — new RFC-tolerated
// characters do not silently become valid bucket names.
func IsSafeURIHost(host string) bool {
	if host == "" {
		return false
	}
	for i := 0; i < len(host); i++ {
		c := host[i]
		switch {
		case c >= 'a' && c <= 'z':
		case c >= 'A' && c <= 'Z':
		case c >= '0' && c <= '9':
		case c == '.' || c == '-' || c == ':':
		default:
			return false
		}
	}
	return true
}

// StripURLScheme removes a leading "http://" or "https://" from address,
// leaving bare host[:port]. Matches the behaviour of the FFI-side helper so
// the persisted URI does not carry a nested scheme when reassembled.
func StripURLScheme(address string) string {
	if strings.HasPrefix(address, "http://") {
		return address[len("http://"):]
	}
	if strings.HasPrefix(address, "https://") {
		return address[len("https://"):]
	}
	return address
}

// isSafeURIHost / stripURLScheme are retained as lowercase wrappers because
// existing test files reference them; delegating to the exported versions
// keeps a single implementation.
func isSafeURIHost(host string) bool       { return IsSafeURIHost(host) }
func stripURLScheme(address string) string { return StripURLScheme(address) }


// RedactExternalSpec returns a log-safe representation of an external spec
// JSON string. Secret extfs values (see secretExtfsKeys) are replaced with
// "***" so that AK/SK/PEM material never reaches log sinks. On parse failure
// it returns "<invalid spec>" rather than the raw input — the input itself
// may already contain a partially-recognized credential blob, so we never
// echo it back. Empty input returns empty string for log readability.
func RedactExternalSpec(specStr string) string {
	if specStr == "" {
		return ""
	}
	var spec ExternalSpec
	if err := json.Unmarshal([]byte(specStr), &spec); err != nil {
		return "<invalid spec>"
	}
	if len(spec.Extfs) > 0 {
		redacted := make(map[string]string, len(spec.Extfs))
		for k, v := range spec.Extfs {
			if secretExtfsKeys[k] && v != "" {
				redacted[k] = "***"
			} else {
				redacted[k] = v
			}
		}
		spec.Extfs = redacted
	}
	out, err := json.Marshal(spec)
	if err != nil {
		return "<marshal error>"
	}
	return string(out)
}
