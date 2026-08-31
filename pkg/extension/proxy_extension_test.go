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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestNoopProxyExtensionIsInert(t *testing.T) {
	n := NoopProxyExtension{}
	ctx := context.Background()

	assert.NoError(t, n.InterceptDML(ctx, DMLInsert, &milvuspb.InsertRequest{}),
		"the native default must not reject a write, or a stock binary would refuse its own DML")
	assert.NoError(t, n.InterceptAdminRPC(ctx, AdminCreateCredential),
		"the native default must not withhold an administrative RPC")

	assert.NoError(t, n.OnConnect(ctx, 1, &commonpb.ClientInfo{}),
		"the native default must not refuse a connection, or a stock binary could not be connected to")
	assert.NotPanics(t, func() { n.OnDisconnect(1) })

	// Start must return rather than block, and must not panic on a nil
	// registry: a caller is entitled to hand it one, and the inert default
	// never touches it.
	n.Start(ctx, nil)
}

// TestShortCircuitCannotBeProducedByAPassingCheck pins the reason the reject
// hooks return error rather than *commonpb.Status: merr.Status(nil) is a
// non-nil success status, so a Status-returning hook written as
// `return merr.Status(check())` would have short-circuited on the pass path.
// With error as the type the same idiom is `return check()`, and a passing
// check falls through by construction.
func TestShortCircuitCannotBeProducedByAPassingCheck(t *testing.T) {
	require.NotNil(t, merr.Status(nil), "the trap this test guards against: a nil error becomes a non-nil status")

	check := func(pass bool) error {
		if pass {
			return nil
		}
		return merr.WrapErrServiceUnavailable("not yet")
	}

	var e ProxyExtension = checkingProxyExtension{check: check}
	assert.NoError(t, e.InterceptDML(context.Background(), DMLInsert, &milvuspb.InsertRequest{}),
		"a passing check must fall through")
	assert.ErrorIs(t, e.InterceptAdminRPC(context.Background(), AdminGetReplicas), merr.ErrServiceUnavailable,
		"a failing check must short-circuit with the sentinel it chose, unwrapped and unreplaced")
}

// checkingProxyExtension is the shape a real implementation takes: embed the
// Noop base, override the hooks in use, forward a check's error.
type checkingProxyExtension struct {
	NoopProxyExtension
	check func(pass bool) error
}

func (c checkingProxyExtension) InterceptDML(context.Context, DMLOp, proto.Message) error {
	return c.check(true)
}

func (c checkingProxyExtension) InterceptAdminRPC(context.Context, AdminOp) error {
	return c.check(false)
}

// TestNoopRewriteRequestParamsReturnsItsArgumentsUntouched pins the one inert
// answer that is not the zero value. Every caller installs both returns, so a
// native default that returned nil or a fresh empty slice would silently erase
// the search parameters of any implementation that embedded it without
// overriding this method, and one that derived a context would allocate on
// every DQL request a stock binary serves.
//
// Equality is not enough for either: the assertions are on identity, the very
// same context and the very same backing array.
func TestNoopRewriteRequestParamsReturnsItsArgumentsUntouched(t *testing.T) {
	params := []*commonpb.KeyValuePair{
		{Key: "metric_type", Value: "L2"},
		{Key: "x-form-reserved", Value: "in07-a"},
	}
	ctx := context.Background()

	gotCtx, cleaned := NoopProxyExtension{}.RewriteRequestParams(ctx, params)

	assert.True(t, ctx == gotCtx, "the native default must hand back the caller's own context, not a derived one")
	require.Len(t, cleaned, 2)
	assert.True(t, &params[0] == &cleaned[0],
		"the native default must hand back the caller's own slice, reserved-looking entry included: it has no protocol of its own to strip")
}

// TestNoopProxyExtensionFallsThroughOnLoadSemantics pins the inert answer for
// the load-semantics group. (false, nil) and (nil, nil) are "fall through",
// and they are the only answers a stock binary can give: handled == true is
// what an implementation returns, and a native default that answered it would
// turn every load, release and progress query in a community build into a
// no-op that reported success - the collection never loaded, the client never
// told.
func TestNoopProxyExtensionFallsThroughOnLoadSemantics(t *testing.T) {
	n := NoopProxyExtension{}
	ctx := context.Background()

	fallsThrough := func(t *testing.T, handled bool, err error, msg string) {
		t.Helper()
		assert.False(t, handled, msg)
		assert.NoError(t, err, msg)
	}

	handled, err := n.InterceptLoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: "coll"})
	fallsThrough(t, handled, err, "a stock binary must load the collection its client asked for")
	handled, err = n.InterceptLoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: "coll", Refresh: true})
	fallsThrough(t, handled, err, "a stock binary must let a refresh reach querycoord")
	handled, err = n.InterceptReleaseCollection(ctx, &milvuspb.ReleaseCollectionRequest{CollectionName: "coll"})
	fallsThrough(t, handled, err, "a stock binary must release the collection its client asked for")
	handled, err = n.InterceptLoadPartitions(ctx, &milvuspb.LoadPartitionsRequest{CollectionName: "coll"})
	fallsThrough(t, handled, err, "a stock binary must load the partitions its client asked for")
	handled, err = n.InterceptLoadPartitions(ctx, &milvuspb.LoadPartitionsRequest{CollectionName: "coll", Refresh: true})
	fallsThrough(t, handled, err, "a stock binary must let a refresh reach querycoord")
	handled, err = n.InterceptReleasePartitions(ctx, &milvuspb.ReleasePartitionsRequest{CollectionName: "coll"})
	fallsThrough(t, handled, err, "a stock binary must release the partitions its client asked for")

	state, err := n.InterceptGetLoadState(ctx, &milvuspb.GetLoadStateRequest{CollectionName: "coll"})
	assert.Nil(t, state, "a stock binary must report the load state it actually has")
	assert.NoError(t, err)
	progress, err := n.InterceptGetLoadingProgress(ctx, &milvuspb.GetLoadingProgressRequest{CollectionName: "coll"})
	assert.Nil(t, progress, "a stock binary must report the loading progress it actually has")
	assert.NoError(t, err)
}

// TestNoopProxyExtensionEmbedsWithoutOverride checks the promise the doc
// comment makes to implementations: embedding the noop is enough to satisfy
// the interface, and an embedder that overrides nothing still gets the inert
// answer rather than a nil method value.
func TestNoopProxyExtensionEmbedsWithoutOverride(t *testing.T) {
	type embedder struct{ NoopProxyExtension }

	var e ProxyExtension = embedder{}
	ctx := context.Background()
	assert.NoError(t, e.InterceptDML(ctx, DMLInsert, &milvuspb.InsertRequest{}),
		"an implementation that overrides nothing must inherit the inert answer")
	assert.NoError(t, e.InterceptAdminRPC(ctx, AdminSelectUser))

	params := []*commonpb.KeyValuePair{{Key: "metric_type", Value: "L2"}}
	gotCtx, cleaned := e.RewriteRequestParams(ctx, params)
	assert.True(t, ctx == gotCtx,
		"an embedder that overrides nothing must inherit the pass-through, not a nil method value")
	assert.Equal(t, params, cleaned)
	assert.NoError(t, e.OnConnect(ctx, 1, nil))
	e.OnDisconnect(1)
	e.Start(ctx, nil)

	handled, err := e.InterceptLoadCollection(ctx, &milvuspb.LoadCollectionRequest{})
	assert.False(t, handled, "an embedder that overrides nothing must inherit the fall-through, not take over the load")
	assert.NoError(t, err)
	handled, err = e.InterceptReleaseCollection(ctx, &milvuspb.ReleaseCollectionRequest{})
	assert.False(t, handled)
	assert.NoError(t, err)
	handled, err = e.InterceptLoadPartitions(ctx, &milvuspb.LoadPartitionsRequest{})
	assert.False(t, handled)
	assert.NoError(t, err)
	handled, err = e.InterceptReleasePartitions(ctx, &milvuspb.ReleasePartitionsRequest{})
	assert.False(t, handled)
	assert.NoError(t, err)
	state, err := e.InterceptGetLoadState(ctx, &milvuspb.GetLoadStateRequest{})
	assert.Nil(t, state)
	assert.NoError(t, err)
	progress, err := e.InterceptGetLoadingProgress(ctx, &milvuspb.GetLoadingProgressRequest{})
	assert.Nil(t, progress)
	assert.NoError(t, err)
}

// The op constants are the whole vocabulary a seam may use, and their values
// are the RPC names, which is what the seams and the access log print.
func TestOpConstantsAreTheRPCNames(t *testing.T) {
	assert.Equal(t, DMLOp("Insert"), DMLInsert)
	assert.Equal(t, DMLOp("Import"), DMLImport)
	assert.Equal(t, AdminOp("OperatePrivilegeV2"), AdminOperatePrivilegeV2)
	assert.Equal(t, AdminOp("CreateReplicateStream"), AdminCreateReplicateStream)
}
