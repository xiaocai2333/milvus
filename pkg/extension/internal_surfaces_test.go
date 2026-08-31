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
	"testing"

	"github.com/stretchr/testify/assert"
)

type stubInternalSurfaces struct{ grpcPort, restPort int }

func (s stubInternalSurfaces) InternalDomainPorts() (int, int) { return s.grpcPort, s.restPort }

func TestInternalSurfacesAbsentWithoutProvider(t *testing.T) {
	ResetForTest()
	assert.Nil(t, Caps().InternalSurfaces,
		"with no provider installed no internal listener may be opened")
}

func TestInternalSurfacesIsInstalledAndRequirable(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	surfaces := stubInternalSurfaces{grpcPort: 19531, restPort: 9092}
	assert.NoError(t, SetProvider(fakeProvider{
		name:     "testprovider",
		requires: []CapabilityID{CapInternalSurfaces},
		caps:     Capabilities{InternalSurfaces: surfaces},
	}))

	got := Caps().InternalSurfaces
	assert.Equal(t, surfaces, got)
	grpcPort, restPort := got.InternalDomainPorts()
	assert.Equal(t, 19531, grpcPort, "the gRPC port must reach the seam unchanged")
	assert.Equal(t, 9092, restPort, "the REST port must reach the seam unchanged")
}

func TestSetProviderRejectsMissingInternalSurfaces(t *testing.T) {
	ResetForTest()
	t.Cleanup(ResetForTest)

	err := SetProvider(fakeProvider{
		name:     "testprovider",
		requires: []CapabilityID{CapInternalSurfaces},
		caps:     Capabilities{},
	})
	assert.ErrorContains(t, err, string(CapInternalSurfaces),
		"a form whose control plane needs the internal listeners must not start without them")
	assert.Nil(t, Caps().InternalSurfaces, "a failed install must leave no trace")
}
