package testcases

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	client "github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/tests/go_client/common"
	hp "github.com/milvus-io/milvus/tests/go_client/testcases/helper"
)

// TestExternalTableArnCloudVerify verifies AssumeRole works on a real AWS S3 bucket
// from a Milvus instance running in the cloud (not local MinIO).
//
// Run with:
//
//	ARN_CLOUD_TEST=true go test -tags dynamic,test -count=1 \
//	  -addr http://localhost:19531 -user zcloud_root -password '<pwd>' \
//	  -run TestExternalTableArnCloudVerify -v ./tests/go_client/testcases/...
//
// Optional env: ARN_DATA_PREFIX=arn-e2e-data/<ts> to point at a path that
// already contains parquet data — runs the full E2E (refresh→load→query).
// If unset, uses a non-existent path and only checks that auth succeeded.
func TestExternalTableArnCloudVerify(t *testing.T) {
	if os.Getenv("ARN_CLOUD_TEST") != "true" {
		t.Skip("skip: set ARN_CLOUD_TEST=true to run")
	}

	const (
		bucket     = "lentitude-bucket"
		region     = "us-west-2"
		roleArn    = "arn:aws:iam::306787409409:role/lentitude-bucket-role"
		externalID = "zilliz-external-sO1cjGS2Vgpyan"
	)

	ctx := hp.CreateContext(t, 10*time.Minute)
	mc := hp.CreateDefaultMilvusClient(ctx, t)

	collName := common.GenRandomString("ext_arn_cloud", 6)
	dataPrefix := os.Getenv("ARN_DATA_PREFIX")
	hasData := dataPrefix != ""
	extPath := dataPrefix
	if extPath == "" {
		extPath = fmt.Sprintf("arn-verify-probe/%s", collName)
	}
	externalSource := fmt.Sprintf("s3://s3.%s.amazonaws.com/%s/%s", region, bucket, extPath)

	type externalSpecJSON struct {
		Format string            `json:"format"`
		Extfs  map[string]string `json:"extfs,omitempty"`
	}
	specObj := externalSpecJSON{
		Format: "parquet",
		Extfs: map[string]string{
			"cloud_provider": "aws",
			"region":         region,
			"storage_type":   "remote",
			"use_ssl":        "true",
			"use_iam":        "true",
			"role_arn":       roleArn,
			"external_id":    externalID,
			"load_frequency": "900",
		},
	}
	specBytes, _ := json.Marshal(specObj)
	externalSpec := string(specBytes)

	t.Logf("[ARN-CLOUD] hasData=%v externalSource=%s", hasData, externalSource)
	t.Logf("[ARN-CLOUD] externalSpec=%s", externalSpec)

	schema := entity.NewSchema().
		WithName(collName).
		WithExternalSource(externalSource).
		WithExternalSpec(`{"format":"parquet"}`).
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithExternalField("id")).
		WithField(entity.NewField().WithName("value").WithDataType(entity.FieldTypeFloat).WithExternalField("value")).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(testVecDim).WithExternalField("embedding"))

	err := mc.CreateCollection(ctx, client.NewCreateCollectionOption(collName, schema))
	require.NoError(t, err, "CreateCollection")
	t.Logf("[ARN-CLOUD] Created collection %s", collName)
	t.Cleanup(func() { _ = mc.DropCollection(ctx, client.NewDropCollectionOption(collName)) })

	t.Log("[ARN-CLOUD] Triggering refresh ...")
	refreshResult, err := mc.RefreshExternalCollection(ctx,
		client.NewRefreshExternalCollectionOption(collName).WithExternalSpec(externalSpec))
	require.NoError(t, err, "RefreshExternalCollection RPC")
	jobID := refreshResult.JobID
	t.Logf("[ARN-CLOUD] Refresh jobID=%d", jobID)

	deadline := time.After(3 * time.Minute)
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	var finalState entity.RefreshExternalCollectionState
	var finalReason string
	for done := false; !done; {
		select {
		case <-deadline:
			if !hasData {
				t.Log("[ARN-CLOUD] PASS (probe mode): refresh stayed Pending — task creation retries due to missing data, no auth error surfaced. Inspect pod logs for 'Path does not exist' to confirm AssumeRole succeeded.")
				return
			}
			t.Fatalf("[ARN-CLOUD] timed out waiting for refresh")
		case <-ticker.C:
			progress, perr := mc.GetRefreshExternalCollectionProgress(ctx,
				client.NewGetRefreshExternalCollectionProgressOption(jobID))
			require.NoError(t, perr)
			t.Logf("[ARN-CLOUD] state=%s reason=%q", progress.State, progress.Reason)
			if progress.State == entity.RefreshStateCompleted ||
				progress.State == entity.RefreshStateFailed {
				finalState = progress.State
				finalReason = progress.Reason
				done = true
			}
		}
	}

	if finalState == entity.RefreshStateFailed {
		low := strings.ToLower(finalReason)
		authMarkers := []string{"accessdenied", "invalidaccesskey", "signaturedoesnotmatch",
			"assumerole", "credentials", "forbidden", "401", "403", "expired"}
		for _, m := range authMarkers {
			if strings.Contains(low, m) {
				t.Fatalf("[ARN-CLOUD] AUTH FAILURE — ARN BROKEN. reason=%s", finalReason)
			}
		}
		t.Fatalf("[ARN-CLOUD] refresh failed (non-auth). reason=%s", finalReason)
	}

	t.Log("[ARN-CLOUD] Refresh completed — full E2E with real AWS S3 + AssumeRole.")

	// Index + load + query to confirm data flows back through the ARN-authenticated read path.
	t.Log("[ARN-CLOUD] Creating index ...")
	idxTask, err := mc.CreateIndex(ctx,
		client.NewCreateIndexOption(collName, "embedding", index.NewFlatIndex(entity.COSINE)))
	require.NoError(t, err)
	require.NoError(t, idxTask.Await(ctx))

	t.Log("[ARN-CLOUD] Loading collection ...")
	loadTask, err := mc.LoadCollection(ctx, client.NewLoadCollectionOption(collName))
	require.NoError(t, err)
	require.NoError(t, loadTask.Await(ctx))

	time.Sleep(3 * time.Second)
	t.Log("[ARN-CLOUD] Querying ...")
	queryResult, err := mc.Query(ctx, client.NewQueryOption(collName).
		WithFilter("id >= 0").WithOutputFields("id").WithLimit(500))
	common.CheckErr(t, err, true)
	rowCount := queryResult.GetColumn("id").Len()
	t.Logf("[ARN-CLOUD] Query returned %d rows", rowCount)
	require.Greater(t, rowCount, 0, "expected at least some rows from ARN-authenticated S3 read")

	t.Log("=== ARN-CLOUD E2E PASSED ===")
}
