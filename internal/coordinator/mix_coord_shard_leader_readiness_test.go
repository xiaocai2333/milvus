package coordinator

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/querycoordv2"
	"github.com/milvus-io/milvus/pkg/v3/extension"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// TestGetShardLeaderReadinessByResourceGroupReachesQueryCoord asserts the
// coordinator's per-resource-group shard-leader readiness is answered by
// querycoord itself, not by a stub: a Server that has not been initialized
// refuses with ErrServiceNotReady, carrying the coordinator-not-ready reason
// and no panic, which is querycoord's own documented behavior for that state.
// When err != nil the struct is unspecified beyond what the producer chose to
// fill in, so the reason is checked here only because this producer does set
// it - a caller classifies on the error first.
func TestGetShardLeaderReadinessByResourceGroupReachesQueryCoord(t *testing.T) {
	qc := &querycoordv2.Server{}
	// The per-resource-group entry points are gated on querycoord's OWN health,
	// not on the mixcoord's: an uninitialized Server refuses rather than
	// dereferencing stores it has not built yet.
	qc.UpdateStateCode(commonpb.StateCode_Healthy)
	s := &mixCoordImpl{queryCoordServer: qc}
	s.UpdateStateCode(commonpb.StateCode_Healthy)

	readiness, err := s.GetShardLeaderReadinessByResourceGroup(context.Background(), 1, "rg-target")
	assert.ErrorIs(t, err, merr.ErrServiceNotReady)
	assert.False(t, readiness.Ready)
	assert.Equal(t, extension.ShardLeadersReasonCoordinatorNotReady, readiness.Reason)
}

// TestGetShardLeaderReadinessByResourceGroupForwardsArgumentsAndResult asserts
// the collection id, the resource group name, the verdict and the error all
// cross the delegation untouched. Answering about a different resource group
// here is precisely the admission bug this method exists to close: it would
// let a caller admit a query to a resource group on the strength of another
// one's shard leaders.
func TestGetShardLeaderReadinessByResourceGroupForwardsArgumentsAndResult(t *testing.T) {
	mockey.PatchConvey("readiness and arguments are forwarded", t, func() {
		var (
			seenCollectionID int64
			seenRG           string
		)
		want := extension.ShardLeaderReadiness{
			Reason:        extension.ShardLeadersReasonShardsWithoutLeader,
			TotalShards:   2,
			UnreadyShards: []string{"coll-dmc1"},
		}
		mockey.Mock((*querycoordv2.Server).GetShardLeaderReadinessByResourceGroup).
			To(func(_ *querycoordv2.Server, _ context.Context, collectionID int64, rgName string) (extension.ShardLeaderReadiness, error) {
				seenCollectionID = collectionID
				seenRG = rgName
				return want, nil
			}).Build()

		s := &mixCoordImpl{queryCoordServer: &querycoordv2.Server{}}
		s.UpdateStateCode(commonpb.StateCode_Healthy)
		readiness, err := s.GetShardLeaderReadinessByResourceGroup(context.Background(), 42, "rg-a")

		assert.NoError(t, err)
		assert.Equal(t, want, readiness)
		assert.EqualValues(t, 42, seenCollectionID)
		assert.Equal(t, "rg-a", seenRG)
	})

	mockey.PatchConvey("a querycoord error is not swallowed", t, func() {
		want := errors.New("collection failed to load")
		mockey.Mock((*querycoordv2.Server).GetShardLeaderReadinessByResourceGroup).
			Return(extension.ShardLeaderReadiness{}, want).Build()

		s := &mixCoordImpl{queryCoordServer: &querycoordv2.Server{}}
		s.UpdateStateCode(commonpb.StateCode_Healthy)
		readiness, err := s.GetShardLeaderReadinessByResourceGroup(context.Background(), 42, "rg-a")

		assert.ErrorIs(t, err, want)
		assert.False(t, readiness.Ready)
	})
}
