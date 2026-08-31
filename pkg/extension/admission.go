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

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
)

// CoordClient is the slice of the coordinator API an admission check may call.
// It is deliberately two methods wide: counting what exists needs to list, not
// to mutate. milvus implements it; it may gain methods and never loses one.
type CoordClient interface {
	ListDatabases(ctx context.Context, req *milvuspb.ListDatabasesRequest) (*milvuspb.ListDatabasesResponse, error)
	ShowCollections(ctx context.Context, req *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error)
}

// AdmissionChecker enforces limits milvus itself has no concept of, such as a
// per-instance or per-database cap on how many databases or collections may
// be held.
//
// milvus decides WHEN to ask; the policy is entirely the implementation's. An
// error rejects the request, and milvus surfaces it to the caller through
// merr.Status - so it must be or wrap a merr sentinel, and a limit the
// request ran into is an input-class one (merr.ErrCollectionNumLimitExceeded
// or a wrap of it), not a service error the client would retry.
//
// An implementation is expected to fail open on its own infrastructure errors:
// refusing a user's DDL because a counting call hiccupped is worse than briefly
// admitting one request too many. It should record that it did so, so the
// bypass is visible rather than silent.
//
// # Mutation
//
// req is READ-ONLY; milvus runs the create on it after admission. There is no
// way to adjust a request here: admission admits or refuses.
//
// NoopAdmissionChecker is the Noop base under the package evolution policy.
type AdmissionChecker interface {
	// CheckCreateCollection runs before a collection is created. req is the
	// request as received, so the target database and the collection's shape
	// are in view - a per-database quota needs the former, a per-shard one the
	// latter.
	CheckCreateCollection(ctx context.Context, req *milvuspb.CreateCollectionRequest, coord CoordClient) error
	// CheckCreateDatabase runs before a database is created; req is the
	// request as received.
	CheckCreateDatabase(ctx context.Context, req *milvuspb.CreateDatabaseRequest, coord CoordClient) error
}

// NoopAdmissionChecker admits everything, which is what a stock binary does.
type NoopAdmissionChecker struct{}

var _ AdmissionChecker = NoopAdmissionChecker{}

func (NoopAdmissionChecker) CheckCreateCollection(context.Context, *milvuspb.CreateCollectionRequest, CoordClient) error {
	return nil
}

func (NoopAdmissionChecker) CheckCreateDatabase(context.Context, *milvuspb.CreateDatabaseRequest, CoordClient) error {
	return nil
}
