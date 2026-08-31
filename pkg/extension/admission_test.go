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

package extension

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
)

// fakeCoordClient is a no-op CoordClient used only to prove that the client
// handed to an AdmissionChecker method is the same instance the caller passed
// in.
type fakeCoordClient struct{}

func (fakeCoordClient) ListDatabases(context.Context, *milvuspb.ListDatabasesRequest) (*milvuspb.ListDatabasesResponse, error) {
	return nil, nil
}

func (fakeCoordClient) ShowCollections(context.Context, *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error) {
	return nil, nil
}

// fakeAdmissionChecker records the CoordClient it was called with, per
// method, and returns a preconfigured error per method.
type fakeAdmissionChecker struct {
	createCollectionErr  error
	createDatabaseErr    error
	seenCreateCollection CoordClient
	seenCreateDatabase   CoordClient
	seenCollectionReq    *milvuspb.CreateCollectionRequest
	seenDatabaseReq      *milvuspb.CreateDatabaseRequest
}

func (f *fakeAdmissionChecker) CheckCreateCollection(ctx context.Context, req *milvuspb.CreateCollectionRequest, coord CoordClient) error {
	f.seenCollectionReq = req
	f.seenCreateCollection = coord
	return f.createCollectionErr
}

func (f *fakeAdmissionChecker) CheckCreateDatabase(ctx context.Context, req *milvuspb.CreateDatabaseRequest, coord CoordClient) error {
	f.seenDatabaseReq = req
	f.seenCreateDatabase = coord
	return f.createDatabaseErr
}

func TestCapabilitiesReportsAdmissionPresence(t *testing.T) {
	assert.False(t, Capabilities{}.has(CapAdmission),
		"an empty table must not claim to supply the admission capability")
	assert.True(t, Capabilities{Admission: &fakeAdmissionChecker{}}.has(CapAdmission))
}

func TestSetProviderRejectsMissingAdmissionCapability(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	err := SetProvider(fakeProvider{
		name:     "testprovider",
		requires: []CapabilityID{CapAdmission},
		caps:     Capabilities{},
	})
	assert.ErrorContains(t, err, string(CapAdmission))
}

func TestInstalledAdmissionCheckerIsReachableThroughCaps(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	checker := &fakeAdmissionChecker{}
	assert.NoError(t, SetProvider(fakeProvider{name: "testprovider", caps: Capabilities{Admission: checker}}))

	got := Caps().Admission
	assert.NotNil(t, got)

	coord := fakeCoordClient{}
	collReq := &milvuspb.CreateCollectionRequest{DbName: "tenant-db", CollectionName: "coll"}
	dbReq := &milvuspb.CreateDatabaseRequest{DbName: "tenant-db"}

	assert.NoError(t, got.CheckCreateCollection(context.Background(), collReq, coord))
	assert.Equal(t, CoordClient(coord), checker.seenCreateCollection,
		"the CoordClient passed to CheckCreateCollection must reach the implementation unchanged")
	assert.Same(t, collReq, checker.seenCollectionReq,
		"the request must reach the implementation, or a per-database quota has no database to count in")

	assert.NoError(t, got.CheckCreateDatabase(context.Background(), dbReq, coord))
	assert.Equal(t, CoordClient(coord), checker.seenCreateDatabase,
		"the CoordClient passed to CheckCreateDatabase must reach the implementation unchanged")
	assert.Same(t, dbReq, checker.seenDatabaseReq)
}

func TestAdmissionCheckerErrorIsPropagated(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	wantCollErr := errors.New("collection admission rejected")
	wantDBErr := errors.New("database admission rejected")
	checker := &fakeAdmissionChecker{createCollectionErr: wantCollErr, createDatabaseErr: wantDBErr}
	assert.NoError(t, SetProvider(fakeProvider{name: "testprovider", caps: Capabilities{Admission: checker}}))

	coord := fakeCoordClient{}

	collErr := Caps().Admission.CheckCreateCollection(context.Background(), &milvuspb.CreateCollectionRequest{}, coord)
	assert.ErrorIs(t, collErr, wantCollErr,
		"an error from CheckCreateCollection must survive install, Caps, and the call unwrapped and unreplaced")

	dbErr := Caps().Admission.CheckCreateDatabase(context.Background(), &milvuspb.CreateDatabaseRequest{}, coord)
	assert.ErrorIs(t, dbErr, wantDBErr,
		"an error from CheckCreateDatabase must survive install, Caps, and the call unwrapped and unreplaced")
}

// NoopAdmissionChecker admits everything: an inert default that refused would
// stop a stock binary from creating a single collection.
func TestNoopAdmissionCheckerAdmits(t *testing.T) {
	type embedder struct{ NoopAdmissionChecker }
	var c AdmissionChecker = embedder{}
	assert.NoError(t, c.CheckCreateCollection(context.Background(), &milvuspb.CreateCollectionRequest{}, fakeCoordClient{}))
	assert.NoError(t, c.CheckCreateDatabase(context.Background(), &milvuspb.CreateDatabaseRequest{}, fakeCoordClient{}))
}
