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

import "context"

// APIKeyVerifier resolves an API key token to the milvus username it maps onto.
//
// A distribution that issues its own API keys installs one; with none installed
// milvus keeps its native token path - which includes the hookutil
// VerifyAPIKey of an out-of-tree plug-in, consulted only when this capability
// is nil - so a stock binary is unaffected.
//
// FROZEN under the package evolution policy: Verify has no inert answer (a
// default that accepted every token would be a hole, one that rejected every
// token would lock every client out), so there is no Noop base and no method
// is ever added. A need this interface cannot express is a new Capabilities
// field.
type APIKeyVerifier interface {
	// Verify maps a raw token to a username. ctx is the authenticating RPC's
	// context: a verifier that asks its own control plane must respect the
	// deadline on it, and a trace started there continues into the call.
	//
	// An error means the token is not valid and the caller reports an
	// authentication failure; it must be or wrap a merr sentinel (see the
	// package # Errors convention on ProxyExtension), and it must not contain
	// the raw token: milvus logs it, and a token is a credential.
	Verify(ctx context.Context, rawToken string) (username string, err error)

	// RequireAPIKeyOnExternalListener reports whether the external listener must
	// refuse username and password authentication. A distribution whose users
	// authenticate only through its own keys returns true, so that a milvus
	// credential cannot be used to bypass the key system.
	RequireAPIKeyOnExternalListener() bool
}
