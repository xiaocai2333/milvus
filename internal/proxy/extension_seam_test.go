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

package proxy

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/util/hookutil"
	"github.com/milvus-io/milvus/pkg/v3/extension"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type testProvider struct{ caps extension.Capabilities }

func (testProvider) Name() string                           { return "test" }
func (testProvider) Requires() []extension.CapabilityID     { return nil }
func (p testProvider) Capabilities() extension.Capabilities { return p.caps }

type recordingAdmissionChecker struct {
	collectionCalls int
	databaseCalls   int
	coordSeen       extension.CoordClient
	err             error
}

func (r *recordingAdmissionChecker) CheckCreateCollection(ctx context.Context, _ *milvuspb.CreateCollectionRequest, coord extension.CoordClient) error {
	r.collectionCalls++
	r.coordSeen = coord
	return r.err
}

func (r *recordingAdmissionChecker) CheckCreateDatabase(ctx context.Context, _ *milvuspb.CreateDatabaseRequest, coord extension.CoordClient) error {
	r.databaseCalls++
	r.coordSeen = coord
	return r.err
}

func TestCheckCreateCollectionAdmissionNoOpWithNoProvider(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)

	// mocks.NewMockMixCoordClient registers no expectations here, so calling
	// any of its methods fails the test immediately (mockery routes an
	// unmatched call through m.Test(t)). Reaching the assertion below without
	// failure, plus the explicit Calls check, is the zero-call proof.
	mockCoord := mocks.NewMockMixCoordClient(t)

	err := checkCreateCollectionAdmission(context.Background(), &milvuspb.CreateCollectionRequest{}, mockCoord)
	assert.NoError(t, err)
	assert.Empty(t, mockCoord.Calls, "with no provider installed checkCreateCollectionAdmission must not touch coord at all")
}

func TestCheckCreateCollectionAdmissionPassesCoordThroughToChecker(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)

	mockCoord := mocks.NewMockMixCoordClient(t)
	wantResp := &milvuspb.ListDatabasesResponse{DbNames: []string{"probe-db"}}
	mockCoord.EXPECT().ListDatabases(mock.Anything, mock.Anything).Return(wantResp, nil).Once()

	checker := &recordingAdmissionChecker{}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	err := checkCreateCollectionAdmission(context.Background(), &milvuspb.CreateCollectionRequest{}, mockCoord)
	assert.NoError(t, err)
	assert.Equal(t, 1, checker.collectionCalls, "the installed checker must be consulted")

	if assert.NotNil(t, checker.coordSeen, "the checker must receive a non-nil CoordClient") {
		gotResp, err := checker.coordSeen.ListDatabases(context.Background(), &milvuspb.ListDatabasesRequest{})
		assert.NoError(t, err)
		assert.Same(t, wantResp, gotResp,
			"the CoordClient handed to the checker must forward calls to the underlying mixCoord, proving the adapter is not a stub")
	}
}

func TestCheckCreateCollectionAdmissionErrorPropagatesUnchanged(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)

	sentinel := errors.New("quota exhausted for this instance")
	checker := &recordingAdmissionChecker{err: sentinel}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	mockCoord := mocks.NewMockMixCoordClient(t)
	err := checkCreateCollectionAdmission(context.Background(), &milvuspb.CreateCollectionRequest{}, mockCoord)
	assert.Same(t, sentinel, err, "the checker's error must reach the caller unchanged, not rewrapped into a different error")
}

// newAdmissionTestCreateCollectionTask builds a createCollectionTask whose
// schema passes every validation step ahead of the admission short-circuit,
// so PreExecute reaches the code under test.
func newAdmissionTestCreateCollectionTask(t *testing.T, cache Cache, collectionName string) *createCollectionTask {
	t.Helper()
	fieldName2Type := map[string]schemapb.DataType{
		"int64": schemapb.DataType_Int64,
		"fvec":  schemapb.DataType_FloatVector,
	}
	schema := constructCollectionSchemaByDataType(collectionName, fieldName2Type, "int64", false)
	marshaledSchema, err := proto.Marshal(schema)
	assert.NoError(t, err)

	ctx := context.Background()
	return &createCollectionTask{
		baseTask:  baseTask{metaCache: cache},
		Condition: NewTaskCondition(ctx),
		CreateCollectionRequest: &milvuspb.CreateCollectionRequest{
			Base:           &commonpb.MsgBase{},
			CollectionName: collectionName,
			Schema:         marshaledSchema,
			ShardsNum:      1,
		},
		ctx:      ctx,
		mixCoord: mocks.NewMockMixCoordClient(t),
	}
}

// TestCheckCreateCollectionAdmissionSkipsExistenceLookupWhenCheckerAdmits pins
// the reordered contract's main payoff: when admission admits -- the common
// case under capacity, and the only case with no provider installed -- the
// existence lookup (and the coordinator round trip it can cost on a cache
// miss) never runs at all. The recorder on globalMetaCache makes "never
// consulted" an assertion about a call count, not an inference from the
// return value.
func TestCheckCreateCollectionAdmissionSkipsExistenceLookupWhenCheckerAdmits(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	// No GetCollectionID expectation is registered: an unexpected call here
	// fails the test on its own, before the explicit assertion below even
	// runs.
	cache := NewMockCache(t)

	checker := &recordingAdmissionChecker{err: nil}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	task := newAdmissionTestCreateCollectionTask(t, cache, "brand_new_coll")
	err := task.PreExecute(task.ctx)

	assert.NoError(t, err)
	assert.Equal(t, 1, checker.collectionCalls, "admission is always consulted first, regardless of existence")
	cache.AssertNumberOfCalls(t, "GetCollectionID", 0)
}

// TestCheckCreateCollectionAdmissionAdmitsRetryWhenCollectionAlreadyExists
// pins the other half of the reordered contract: when admission would
// reject, the existence lookup runs, and finding the collection already
// counted turns the rejection into a nil so a retry still reaches
// rootcoord's own idempotent answer instead of seeing ResourceExhausted.
func TestCheckCreateCollectionAdmissionAdmitsRetryWhenCollectionAlreadyExists(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	collectionName := "already_exists_coll"
	cache := NewMockCache(t)
	cache.On("GetCollectionID", mock.Anything, "", collectionName).Return(UniqueID(1001), nil)

	checker := &recordingAdmissionChecker{err: errors.New("quota exhausted")}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	task := newAdmissionTestCreateCollectionTask(t, cache, collectionName)
	err := task.PreExecute(task.ctx)

	assert.NoError(t, err, "an idempotent re-create of an existing collection must not be blocked by admission")
	assert.Equal(t, 1, checker.collectionCalls, "admission is consulted first, even though it ends up overridden")
	cache.AssertNumberOfCalls(t, "GetCollectionID", 1)
}

// TestCheckCreateCollectionAdmissionRejectsWhenCollectionIsGenuinelyNew
// confirms the reorder does not weaken the rejection itself: when admission
// would reject and the collection genuinely does not exist, the rejection
// still surfaces unchanged.
func TestCheckCreateCollectionAdmissionRejectsWhenCollectionIsGenuinelyNew(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	collectionName := "brand_new_coll"
	cache := NewMockCache(t)
	cache.On("GetCollectionID", mock.Anything, "", collectionName).Return(UniqueID(0), merr.WrapErrCollectionNotFound(collectionName))

	sentinel := errors.New("quota exhausted for this instance")
	checker := &recordingAdmissionChecker{err: sentinel}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	task := newAdmissionTestCreateCollectionTask(t, cache, collectionName)
	err := task.PreExecute(task.ctx)

	assert.Equal(t, 1, checker.collectionCalls, "admission must be consulted for a genuinely new collection")
	cache.AssertNumberOfCalls(t, "GetCollectionID", 1)
	assert.Same(t, sentinel, err, "the checker's error must surface from PreExecute unchanged")
}

// newAdmissionTestCreateDatabaseTask builds a createDatabaseTask whose name
// passes validation ahead of the admission short-circuit, so PreExecute
// reaches the code under test.
func newAdmissionTestCreateDatabaseTask(t *testing.T, cache Cache, dbName string) *createDatabaseTask {
	t.Helper()
	ctx := context.Background()
	return &createDatabaseTask{
		baseTask:  baseTask{metaCache: cache},
		Condition: NewTaskCondition(ctx),
		CreateDatabaseRequest: &milvuspb.CreateDatabaseRequest{
			DbName: dbName,
		},
		ctx:      ctx,
		mixCoord: mocks.NewMockMixCoordClient(t),
	}
}

func TestCheckCreateDatabaseAdmissionSkippedWhenDatabaseAlreadyExists(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	dbName := "already_exists_db"
	cache := NewMockCache(t)
	// GetDatabaseInfo, not HasDatabase: the local peek only knows databases
	// whose collections this proxy has cached, so an empty or un-cached
	// database would be refused on an idempotent retry. GetDatabaseInfo
	// carries the RPC fallback.
	cache.On("GetDatabaseInfo", mock.Anything, dbName).Return(&databaseInfo{DBID: 7}, nil)

	checker := &recordingAdmissionChecker{err: errors.New("quota exhausted")}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	task := newAdmissionTestCreateDatabaseTask(t, cache, dbName)
	err := task.PreExecute(task.ctx)

	assert.NoError(t, err, "an idempotent re-create of an existing database must not be blocked by admission")
	assert.Equal(t, 1, checker.databaseCalls,
		"admission runs first - the existence lookup is paid only on rejection, same as the collection path")
}

func TestCheckCreateDatabaseAdmissionConsultedWhenDatabaseIsNew(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	dbName := "brand_new_db"
	cache := NewMockCache(t)
	cache.On("GetDatabaseInfo", mock.Anything, dbName).Return(nil, errors.New("database not found"))

	sentinel := errors.New("quota exhausted for this instance")
	checker := &recordingAdmissionChecker{err: sentinel}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	task := newAdmissionTestCreateDatabaseTask(t, cache, dbName)
	err := task.PreExecute(task.ctx)

	assert.Equal(t, 1, checker.databaseCalls, "admission must be consulted for a genuinely new database")
	assert.Same(t, sentinel, err, "the checker's error must surface from PreExecute unchanged")
}

// TestCheckCreateDatabaseAdmissionSkipsExistenceLookupWithNoProvider pins the
// property the round-5 fix restores, now through admissionChecker() and the
// direct caps.Admission.CheckCreateDatabase call in task_database.go: with no
// provider installed, createDatabaseTask.PreExecute must reach exactly the
// statements it reached before the admission seam existed, touching neither
// the metadata cache nor the coordinator. Both recorders (globalMetaCache and
// the mixCoord mock) make "never consulted" an assertion about a call count,
// the same shape as
// TestCheckCreateCollectionAdmissionSkipsExistenceLookupWhenCheckerAdmits.
func TestCheckCreateDatabaseAdmissionSkipsExistenceLookupWithNoProvider(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	// No HasDatabase expectation is registered: an unexpected call fails the
	// test on its own, before the explicit assertion below even runs.
	cache := NewMockCache(t)

	task := newAdmissionTestCreateDatabaseTask(t, cache, "brand_new_db")
	err := task.PreExecute(task.ctx)

	assert.NoError(t, err)
	cache.AssertNumberOfCalls(t, "GetDatabaseInfo", 0)
	mockCoord, ok := task.mixCoord.(*mocks.MockMixCoordClient)
	if assert.True(t, ok, "test helper must build mixCoord as a MockMixCoordClient") {
		assert.Empty(t, mockCoord.Calls, "with no provider installed PreExecute must not touch coord at all")
	}
}

// TestCheckCreateDatabaseAdmissionPassesCoordThroughAtPreExecute proves the
// CoordClient the installed checker receives, when reached through
// createDatabaseTask.PreExecute, forwards to the real mixCoord. This is the
// production-path replacement for the coverage a standalone
// checkCreateDatabaseAdmission wrapper used to provide before task_database.go
// started calling admissionChecker() and CheckCreateDatabase directly.
func TestCheckCreateDatabaseAdmissionPassesCoordThroughAtPreExecute(t *testing.T) {
	extension.ResetForTest()
	t.Cleanup(extension.ResetForTest)
	dbName := "brand_new_db_passthrough"
	// No expectations: the checker admits, so the existence lookup never runs.
	cache := NewMockCache(t)

	checker := &recordingAdmissionChecker{}
	assert.NoError(t, extension.SetProvider(testProvider{caps: extension.Capabilities{Admission: checker}}))

	task := newAdmissionTestCreateDatabaseTask(t, cache, dbName)
	mockCoord := task.mixCoord.(*mocks.MockMixCoordClient)
	wantResp := &milvuspb.ShowCollectionsResponse{CollectionNames: []string{"probe-coll"}}
	mockCoord.EXPECT().ShowCollections(mock.Anything, mock.Anything).Return(wantResp, nil).Once()

	err := task.PreExecute(task.ctx)
	assert.NoError(t, err)

	if assert.NotNil(t, checker.coordSeen, "the checker must receive a non-nil CoordClient") {
		gotResp, err := checker.coordSeen.ShowCollections(context.Background(), &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.Same(t, wantResp, gotResp,
			"the CoordClient handed to the checker via task_database.go must forward calls to the underlying mixCoord")
	}
}

type fakeReplicateStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (f fakeReplicateStream) Context() context.Context                  { return f.ctx }
func (f fakeReplicateStream) Send(*milvuspb.ReplicateResponse) error    { return nil }
func (f fakeReplicateStream) Recv() (*milvuspb.ReplicateRequest, error) { return nil, nil }

// CreateReplicateStream is the one RPC that consults the hook by hand, because
// the interceptor that consults it for every other RPC is a unary one and an
// interceptor chain binds to one of gRPC's two call kinds. That hand-written
// call is exactly the kind a refactor can drop without anything failing, so it
// is pinned here.
func TestCreateReplicateStreamConsultsTheHook(t *testing.T) {
	hookutil.InitOnceHook()
	hookutil.SetTestHook(refusingStreamHook{})
	defer hookutil.SetTestHook(hookutil.DefaultHook{})

	node := &Proxy{}
	node.UpdateStateCode(commonpb.StateCode_Healthy)

	err := node.CreateReplicateStream(fakeReplicateStream{ctx: context.Background()})
	assert.ErrorIs(t, err, merr.ErrServiceUnimplemented)
}

// refusingStreamHook withholds only the replicate stream, so the test cannot
// pass by the RPC failing for some unrelated reason.
type refusingStreamHook struct {
	hookutil.DefaultHook
}

func (refusingStreamHook) Before(ctx context.Context, req interface{}, fullMethod string) (context.Context, error) {
	if fullMethod == milvuspb.MilvusService_CreateReplicateStream_FullMethodName {
		return ctx, merr.ErrServiceUnimplemented
	}
	return ctx, nil
}
