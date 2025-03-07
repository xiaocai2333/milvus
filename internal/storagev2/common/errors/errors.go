// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package errors

import "github.com/cockroachdb/errors"

var (
	ErrSchemaIsNil      = errors.New("schema is nil")
	ErrBlobAlreadyExist = errors.New("blob already exist")
	ErrBlobNotExist     = errors.New("blob not exist")
	ErrSchemaNotMatch   = errors.New("schema not match")
	ErrColumnNotExist   = errors.New("column not exist")
	ErrInvalidPath      = errors.New("invalid path")
	ErrNoEndpoint       = errors.New("no endpoint is specified")
)
