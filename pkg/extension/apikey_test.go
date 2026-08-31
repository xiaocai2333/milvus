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
)

type fakeVerifier struct {
	user     string
	err      error
	external bool
	seen     []string
}

func (f *fakeVerifier) Verify(_ context.Context, rawToken string) (string, error) {
	f.seen = append(f.seen, rawToken)
	return f.user, f.err
}

func (f *fakeVerifier) RequireAPIKeyOnExternalListener() bool { return f.external }

func TestCapabilitiesReportsAPIKeyPresence(t *testing.T) {
	assert.False(t, Capabilities{}.has(CapAPIKey),
		"an empty table must not claim to supply the api key capability")
	assert.True(t, Capabilities{APIKey: &fakeVerifier{}}.has(CapAPIKey))
}

func TestSetProviderRejectsMissingAPIKeyCapability(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	err := SetProvider(fakeProvider{
		name:     "testprovider",
		requires: []CapabilityID{CapAPIKey},
		caps:     Capabilities{},
	})
	assert.ErrorContains(t, err, string(CapAPIKey))
}

func TestInstalledVerifierIsReachableThroughCaps(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	v := &fakeVerifier{user: "alice", external: true}
	assert.NoError(t, SetProvider(fakeProvider{name: "testprovider", caps: Capabilities{APIKey: v}}))

	got := Caps().APIKey
	assert.NotNil(t, got)

	user, err := got.Verify(context.Background(), "tok-1")
	assert.NoError(t, err)
	assert.Equal(t, "alice", user)
	assert.Equal(t, []string{"tok-1"}, v.seen, "the raw token must reach the verifier unchanged")
	assert.True(t, got.RequireAPIKeyOnExternalListener())
}

func TestVerifierErrorIsPropagated(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	want := errors.New("boom")
	v := &fakeVerifier{err: want}
	assert.NoError(t, SetProvider(fakeProvider{name: "testprovider", caps: Capabilities{APIKey: v}}))

	_, err := Caps().APIKey.Verify(context.Background(), "tok")
	assert.ErrorIs(t, err, want, "an error from Verify must survive install, Caps, and the call unwrapped and unreplaced")
}
