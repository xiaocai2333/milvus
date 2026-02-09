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

package storage

import (
	"context"
	"io"
	"math"
	"os"
	"path"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/mocks/flushcommon/mock_util"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

func TestPackedBinlogRecordSuite(t *testing.T) {
	suite.Run(t, new(PackedBinlogRecordSuite))
}

type PackedBinlogRecordSuite struct {
	suite.Suite

	ctx          context.Context
	mockID       atomic.Int64
	logIDAlloc   allocator.Interface
	mockBinlogIO *mock_util.MockBinlogIO

	collectionID  UniqueID
	partitionID   UniqueID
	segmentID     UniqueID
	schema        *schemapb.CollectionSchema
	maxRowNum     int64
	chunkSize     uint64
	storageConfig *indexpb.StorageConfig
}

func (s *PackedBinlogRecordSuite) SetupTest() {
	ctx := context.Background()
	s.ctx = ctx
	logIDAlloc := allocator.NewLocalAllocator(1, math.MaxInt64)
	s.logIDAlloc = logIDAlloc
	// initcore.InitLocalArrowFileSystem("/tmp")
	s.mockID.Store(time.Now().UnixMilli())
	s.mockBinlogIO = mock_util.NewMockBinlogIO(s.T())
	s.collectionID = UniqueID(0)
	s.partitionID = UniqueID(0)
	s.segmentID = UniqueID(0)
	s.schema = generateTestSchema()
	// s.rootPath = "/tmp"
	// s.bucketName = "a-bucket"
	s.maxRowNum = int64(1000)
	s.chunkSize = uint64(1024)
	s.storageConfig = &indexpb.StorageConfig{
		StorageType: "local",
		RootPath:    "/tmp",
		BucketName:  "a-bucket",
	}
}

func (s *PackedBinlogRecordSuite) TestPackedBinlogRecordIntegration() {
	paramtable.Get().Save(paramtable.Get().CommonCfg.StorageType.Key, "local")
	s.mockBinlogIO.EXPECT().Upload(mock.Anything, mock.Anything).Return(nil)
	rows := 10000
	readBatchSize := 1024
	columnGroups := []storagecommon.ColumnGroup{
		{
			GroupID: 0,
			Columns: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			Fields:  []int64{0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101},
		},
		{
			GroupID: 102,
			Columns: []int{13},
			Fields:  []int64{102},
		},
		{
			GroupID: 103,
			Columns: []int{14},
			Fields:  []int64{103},
		},
		{
			GroupID: 104,
			Columns: []int{15},
			Fields:  []int64{104},
		},
		{
			GroupID: 105,
			Columns: []int{16},
			Fields:  []int64{105},
		},
		{
			GroupID: 106,
			Columns: []int{17},
			Fields:  []int64{106},
		},
	}
	wOption := []RwOption{
		WithUploader(func(ctx context.Context, kvs map[string][]byte) error {
			return s.mockBinlogIO.Upload(ctx, kvs)
		}),
		WithVersion(StorageV2),
		WithMultiPartUploadSize(0),
		WithBufferSize(1 * 1024 * 1024), // 1MB
		WithColumnGroups(columnGroups),
		WithStorageConfig(s.storageConfig),
	}

	w, err := NewBinlogRecordWriter(s.ctx, s.collectionID, s.partitionID, s.segmentID, s.schema, s.logIDAlloc, s.chunkSize, s.maxRowNum, wOption...)
	s.NoError(err)

	blobs, err := generateTestData(rows)
	s.NoError(err)

	reader, err := NewBinlogDeserializeReader(generateTestSchema(), MakeBlobsReader(blobs), false)
	s.NoError(err)
	defer reader.Close()

	for i := 1; i <= rows; i++ {
		value, err := reader.NextValue()
		s.NoError(err)
		rec, err := ValueSerializer([]*Value{*value}, s.schema)
		s.NoError(err)
		err = w.Write(rec)
		s.NoError(err)
	}
	err = w.Close()
	s.NoError(err)
	writtenUncompressed := w.GetWrittenUncompressed()
	s.Positive(writtenUncompressed)

	rowNum := w.GetRowNum()
	s.Equal(rowNum, int64(rows))

	fieldBinlogs, statsLog, bm25StatsLog, _ := w.GetLogs()
	s.Equal(len(fieldBinlogs), len(columnGroups))
	for _, columnGroup := range fieldBinlogs {
		s.Equal(len(columnGroup.Binlogs), 1)
		s.Equal(columnGroup.Binlogs[0].EntriesNum, int64(rows))
		s.Positive(columnGroup.Binlogs[0].MemorySize)
	}

	s.Equal(len(statsLog.Binlogs), 1)
	s.Equal(statsLog.Binlogs[0].EntriesNum, int64(rows))

	s.Equal(len(bm25StatsLog), 0)

	binlogs := SortFieldBinlogs(fieldBinlogs)
	rOption := []RwOption{
		WithVersion(StorageV2),
		WithStorageConfig(s.storageConfig),
	}
	r, err := NewBinlogRecordReader(s.ctx, binlogs, s.schema, rOption...)
	s.NoError(err)
	defer r.Close()
	for i := 0; i < rows/readBatchSize+1; i++ {
		rec, err := r.Next()
		s.NoError(err)
		if i < rows/readBatchSize {
			s.Equal(rec.Len(), readBatchSize)
		} else {
			s.Equal(rec.Len(), rows%readBatchSize)
		}
	}

	_, err = r.Next()
	s.Equal(err, io.EOF)
	err = r.Close()
	s.NoError(err)
}

func (s *PackedBinlogRecordSuite) TestGenerateBM25Stats() {
	s.mockBinlogIO.EXPECT().Upload(mock.Anything, mock.Anything).Return(nil)
	s.schema = genCollectionSchemaWithBM25()
	columnGroups := []storagecommon.ColumnGroup{
		{
			GroupID: 0,
			Columns: []int{0, 1, 2},
		},
		{
			GroupID: 101,
			Columns: []int{3},
		},
		{
			GroupID: 102,
			Columns: []int{4},
		},
	}
	wOption := []RwOption{
		WithUploader(func(ctx context.Context, kvs map[string][]byte) error {
			return s.mockBinlogIO.Upload(ctx, kvs)
		}),
		WithVersion(StorageV2),
		WithMultiPartUploadSize(0),
		WithBufferSize(10 * 1024 * 1024), // 10MB
		WithColumnGroups(columnGroups),
		WithStorageConfig(s.storageConfig),
	}

	v := &Value{
		PK:        NewVarCharPrimaryKey("0"),
		Timestamp: int64(tsoutil.ComposeTSByTime(getMilvusBirthday(), 0)),
		Value:     genRowWithBM25(0),
	}
	rec, err := ValueSerializer([]*Value{v}, s.schema)
	s.NoError(err)

	w, err := NewBinlogRecordWriter(s.ctx, s.collectionID, s.partitionID, s.segmentID, s.schema, s.logIDAlloc, s.chunkSize, s.maxRowNum, wOption...)
	s.NoError(err)
	err = w.Write(rec)
	s.NoError(err)
	err = w.Close()
	s.NoError(err)
	fieldBinlogs, statsLog, bm25StatsLog, _ := w.GetLogs()
	s.Equal(len(fieldBinlogs), len(columnGroups))

	s.Equal(statsLog.Binlogs[0].EntriesNum, int64(1))
	s.Positive(statsLog.Binlogs[0].MemorySize)

	s.Equal(len(bm25StatsLog), 1)
	s.Equal(bm25StatsLog[102].Binlogs[0].EntriesNum, int64(1))
	s.Positive(bm25StatsLog[102].Binlogs[0].MemorySize)
}

func (s *PackedBinlogRecordSuite) TestUnsuportedStorageVersion() {
	wOption := []RwOption{
		WithVersion(-1),
		WithStorageConfig(s.storageConfig),
	}
	_, err := NewBinlogRecordWriter(s.ctx, s.collectionID, s.partitionID, s.segmentID, s.schema, s.logIDAlloc, s.chunkSize, s.maxRowNum, wOption...)
	s.Error(err)

	rOption := []RwOption{
		WithVersion(-1),
	}
	_, err = NewBinlogRecordReader(s.ctx, []*datapb.FieldBinlog{{}}, s.schema, rOption...)
	s.Error(err)
}

func (s *PackedBinlogRecordSuite) TestNoPrimaryKeyError() {
	s.schema = &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 13, Name: "field12", DataType: schemapb.DataType_JSON},
	}}
	columnGroups := []storagecommon.ColumnGroup{
		{
			GroupID: 0,
			Columns: []int{0},
		},
	}
	wOption := []RwOption{
		WithVersion(StorageV2),
		WithColumnGroups(columnGroups),
		WithStorageConfig(s.storageConfig),
	}
	_, err := NewBinlogRecordWriter(s.ctx, s.collectionID, s.partitionID, s.segmentID, s.schema, s.logIDAlloc, s.chunkSize, s.maxRowNum, wOption...)
	s.Error(err)
}

func (s *PackedBinlogRecordSuite) TestConvertArrowSchemaError() {
	s.schema = &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 14, Name: "field13", DataType: schemapb.DataType_Float16Vector, TypeParams: []*commonpb.KeyValuePair{}},
	}}
	columnGroups := []storagecommon.ColumnGroup{
		{
			GroupID: 0,
			Columns: []int{0},
		},
	}
	wOption := []RwOption{
		WithVersion(StorageV2),
		WithColumnGroups(columnGroups),
		WithStorageConfig(s.storageConfig),
	}
	_, err := NewBinlogRecordWriter(s.ctx, s.collectionID, s.partitionID, s.segmentID, s.schema, s.logIDAlloc, s.chunkSize, s.maxRowNum, wOption...)
	s.Error(err)
}

func (s *PackedBinlogRecordSuite) TestEmptyBinlog() {
	rOption := []RwOption{
		WithVersion(StorageV2),
		WithStorageConfig(s.storageConfig),
	}
	_, err := NewBinlogRecordReader(s.ctx, []*datapb.FieldBinlog{}, s.schema, rOption...)
	s.Error(err)
}

func (s *PackedBinlogRecordSuite) TestAllocIDExhausedError() {
	columnGroups := []storagecommon.ColumnGroup{
		{
			GroupID: 0,
			Columns: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
		},
	}
	wOption := []RwOption{
		WithVersion(StorageV2),
		WithColumnGroups(columnGroups),
		WithStorageConfig(s.storageConfig),
		WithUploader(func(ctx context.Context, kvs map[string][]byte) error {
			return nil
		}),
	}
	logIDAlloc := allocator.NewLocalAllocator(1, 1)
	w, err := NewBinlogRecordWriter(s.ctx, s.collectionID, s.partitionID, s.segmentID, s.schema, logIDAlloc, s.chunkSize, s.maxRowNum, wOption...)
	s.NoError(err)

	size := 10
	blobs, err := generateTestData(size)
	s.NoError(err)

	reader, err := NewBinlogDeserializeReader(generateTestSchema(), MakeBlobsReader(blobs), false)
	s.NoError(err)
	defer reader.Close()

	for i := 0; i < size; i++ {
		value, err := reader.NextValue()
		s.NoError(err)

		rec, err := ValueSerializer([]*Value{*value}, s.schema)
		s.NoError(err)
		err = w.Write(rec)
		s.Error(err)
	}
}

func genRowWithBM25(magic int64) map[int64]interface{} {
	ts := tsoutil.ComposeTSByTime(getMilvusBirthday(), 0)
	return map[int64]interface{}{
		common.RowIDField:     magic,
		common.TimeStampField: int64(ts),
		100:                   strconv.FormatInt(magic, 10),
		101:                   "varchar",
		102:                   typeutil.CreateAndSortSparseFloatRow(map[uint32]float32{1: 1}),
	}
}

func genCollectionSchemaWithBM25() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Name:        "schema",
		Description: "schema",
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:  common.RowIDField,
				Name:     "row_id",
				DataType: schemapb.DataType_Int64,
			},
			{
				FieldID:  common.TimeStampField,
				Name:     "Timestamp",
				DataType: schemapb.DataType_Int64,
			},
			{
				FieldID:      100,
				Name:         "pk",
				DataType:     schemapb.DataType_VarChar,
				IsPrimaryKey: true,
			},
			{
				FieldID:  101,
				Name:     "text",
				DataType: schemapb.DataType_VarChar,
				TypeParams: []*commonpb.KeyValuePair{
					{
						Key:   common.MaxLengthKey,
						Value: "8",
					},
				},
			},
			{
				FieldID:  102,
				Name:     "sparse",
				DataType: schemapb.DataType_SparseFloatVector,
			},
		},
		Functions: []*schemapb.FunctionSchema{{
			Name:             "BM25",
			Id:               100,
			Type:             schemapb.FunctionType_BM25,
			InputFieldNames:  []string{"text"},
			InputFieldIds:    []int64{101},
			OutputFieldNames: []string{"sparse"},
			OutputFieldIds:   []int64{102},
		}},
	}
}

func getMilvusBirthday() time.Time {
	return time.Date(2019, time.Month(5), 30, 0, 0, 0, 0, time.UTC)
}

func Test_makeBlobsReader(t *testing.T) {
	ctx := context.Background()
	downloader := func(ctx context.Context, paths []string) ([][]byte, error) {
		return lo.Map(paths, func(item string, index int) []byte {
			return []byte{}
		}), nil
	}

	tests := []struct {
		name    string
		binlogs []*datapb.FieldBinlog
		want    [][]*Blob
		wantErr bool
	}{
		{
			name: "test full",
			binlogs: []*datapb.FieldBinlog{
				{
					FieldID: 100,
					Binlogs: []*datapb.Binlog{
						{LogPath: "x/1/1/1/100/1"},
					},
				},
				{
					FieldID: 101,
					Binlogs: []*datapb.Binlog{
						{LogPath: "x/1/1/1/101/2"},
					},
				},
				{
					FieldID: 102,
					Binlogs: []*datapb.Binlog{
						{LogPath: "x/1/1/1/102/3"},
					},
				},
			},
			want: [][]*Blob{
				{
					{
						Key:   "x/1/1/1/100/1",
						Value: []byte{},
					},
					{
						Key:   "x/1/1/1/101/2",
						Value: []byte{},
					},
					{
						Key:   "x/1/1/1/102/3",
						Value: []byte{},
					},
				},
			},
			wantErr: false,
		},

		{
			name: "test added field",
			binlogs: []*datapb.FieldBinlog{
				{
					FieldID: 100,
					Binlogs: []*datapb.Binlog{
						{LogPath: "x/1/1/1/100/1"},
						{LogPath: "x/1/1/1/100/3"},
					},
				},
				{
					FieldID: 101,
					Binlogs: []*datapb.Binlog{
						{LogPath: "x/1/1/1/101/2"},
						{LogPath: "x/1/1/1/101/4"},
					},
				},
				{
					FieldID: 102,
					Binlogs: []*datapb.Binlog{
						{LogPath: "x/1/1/1/102/5"},
					},
				},
			},
			want: [][]*Blob{
				{
					{
						Key:   "x/1/1/1/100/1",
						Value: []byte{},
					},
					{
						Key:   "x/1/1/1/101/2",
						Value: []byte{},
					},
				},
				{
					{
						Key:   "x/1/1/1/100/3",
						Value: []byte{},
					},
					{
						Key:   "x/1/1/1/101/4",
						Value: []byte{},
					},
					{
						Key:   "x/1/1/1/102/5",
						Value: []byte{},
					},
				},
			},
			wantErr: false,
		},

		// {
		// 	name: "test error",
		// 	binlogs: []*datapb.FieldBinlog{
		// 		{
		// 			FieldID: 100,
		// 			Binlogs: []*datapb.Binlog{
		// 				{LogPath: "x/1/1/1/100/1"},
		// 				{LogPath: "x/1/1/1/100/3"},
		// 			},
		// 		},
		// 		{
		// 			FieldID: 101,
		// 			Binlogs: []*datapb.Binlog{
		// 				{LogPath: "x/1/1/1/101/2"},
		// 				{LogPath: "x/1/1/1/101/4"},
		// 			},
		// 		},
		// 		{
		// 			FieldID: 102,
		// 			Binlogs: []*datapb.Binlog{
		// 				{LogPath: "x/1/1/1/102/5"},
		// 				{LogPath: "x/1/1/1/102/6"},
		// 			},
		// 		},
		// 	},
		// 	want:    nil,
		// 	wantErr: true,
		// },
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader, err := makeBlobsReader(ctx, tt.binlogs, downloader)
			if err != nil {
				if !tt.wantErr {
					t.Errorf("makeBlobsReader() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			got := make([][]*Blob, 0)
			for {
				bs, err := reader()
				if err == io.EOF {
					break
				}
				if err != nil {
					assert.Fail(t, err.Error())
				}
				got = append(got, bs)
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestRwOptionValidate(t *testing.T) {
	testCases := []struct {
		tag         string
		input       *rwOptions
		expectError bool
	}{
		{
			tag: "normal_case",
			input: &rwOptions{
				version:       StorageV1,
				storageConfig: &indexpb.StorageConfig{},
				op:            OpRead,
				downloader:    func(ctx context.Context, paths []string) ([][]byte, error) { return nil, nil },
			},
			expectError: false,
		},
		{
			tag: "normal_case_v2",
			input: &rwOptions{
				version:       StorageV2,
				storageConfig: &indexpb.StorageConfig{},
				op:            OpRead,
			},
			expectError: false,
		},
		{
			tag: "bad_version",
			input: &rwOptions{
				version:       -1,
				storageConfig: &indexpb.StorageConfig{},
				downloader:    func(ctx context.Context, paths []string) ([][]byte, error) { return nil, nil },
				op:            OpRead,
			},
			expectError: true,
		},
		{
			tag: "missing_config",
			input: &rwOptions{
				version:       StorageV2,
				storageConfig: nil,
				op:            OpRead,
			},
			expectError: true,
		},
		{
			tag: "v1eader_missing_downloader",
			input: &rwOptions{
				version:       StorageV1,
				storageConfig: &indexpb.StorageConfig{},
				op:            OpRead,
			},
			expectError: true,
		},
		{
			tag: "writer_missing_uploader",
			input: &rwOptions{
				version:       StorageV2,
				storageConfig: &indexpb.StorageConfig{},
				op:            OpWrite,
			},
			expectError: true,
		},
		{
			tag: "v1_with_prefetch",
			input: &rwOptions{
				version:       StorageV1,
				storageConfig: &indexpb.StorageConfig{},
				op:            OpRead,
				downloader:    func(ctx context.Context, paths []string) ([][]byte, error) { return nil, nil },
				prefetch:      true,
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.tag, func(t *testing.T) {
			err := tc.input.validate()
			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func Test_prepareBinlogChunks(t *testing.T) {
	t.Run("full binlogs", func(t *testing.T) {
		binlogs := []*datapb.FieldBinlog{
			{
				FieldID: 102,
				Binlogs: []*datapb.Binlog{
					{LogPath: "x/1/1/1/102/3"},
				},
			},
			{
				FieldID: 100,
				Binlogs: []*datapb.Binlog{
					{LogPath: "x/1/1/1/100/1"},
				},
			},
			{
				FieldID: 101,
				Binlogs: []*datapb.Binlog{
					{LogPath: "x/1/1/1/101/2"},
				},
			},
		}
		chunks := prepareBinlogChunks(binlogs)
		assert.Equal(t, [][]string{
			{"x/1/1/1/100/1", "x/1/1/1/101/2", "x/1/1/1/102/3"},
		}, chunks)
	})

	t.Run("added field (uneven binlogs)", func(t *testing.T) {
		binlogs := []*datapb.FieldBinlog{
			{
				FieldID: 100,
				Binlogs: []*datapb.Binlog{
					{LogPath: "x/1/1/1/100/1"},
					{LogPath: "x/1/1/1/100/3"},
				},
			},
			{
				FieldID: 101,
				Binlogs: []*datapb.Binlog{
					{LogPath: "x/1/1/1/101/2"},
					{LogPath: "x/1/1/1/101/4"},
				},
			},
			{
				FieldID: 102,
				Binlogs: []*datapb.Binlog{
					{LogPath: "x/1/1/1/102/5"},
				},
			},
		}
		chunks := prepareBinlogChunks(binlogs)
		assert.Equal(t, [][]string{
			{"x/1/1/1/100/1", "x/1/1/1/101/2"},
			{"x/1/1/1/100/3", "x/1/1/1/101/4", "x/1/1/1/102/5"},
		}, chunks)
	})
}

func Test_prefetchToLocal(t *testing.T) {
	ctx := context.Background()

	t.Run("downloads files to temp dir", func(t *testing.T) {
		downloader := func(ctx context.Context, paths []string) ([][]byte, error) {
			return lo.Map(paths, func(p string, _ int) []byte {
				return []byte("data-" + p)
			}), nil
		}
		paths := [][]string{
			{"file1.parquet", "file2.parquet"},
			{"file3.parquet"},
		}
		localPaths, tempDir, err := prefetchToLocal(ctx, downloader, paths)
		assert.NoError(t, err)
		assert.DirExists(t, tempDir)
		assert.Contains(t, tempDir, "/tmp/prefetchReader/")
		defer os.RemoveAll(tempDir)

		// Verify structure matches input
		assert.Equal(t, len(paths), len(localPaths))
		assert.Equal(t, 2, len(localPaths[0]))
		assert.Equal(t, 1, len(localPaths[1]))

		// Verify paths are absolute (under tempDir)
		assert.Equal(t, path.Join(tempDir, "chunk_0_0.parquet"), localPaths[0][0])
		assert.Equal(t, path.Join(tempDir, "chunk_1_0.parquet"), localPaths[1][0])

		// Verify file contents
		content, err := os.ReadFile(localPaths[0][0])
		assert.NoError(t, err)
		assert.Equal(t, []byte("data-file1.parquet"), content)

		content, err = os.ReadFile(localPaths[1][0])
		assert.NoError(t, err)
		assert.Equal(t, []byte("data-file3.parquet"), content)
	})

	t.Run("cleans up on download error", func(t *testing.T) {
		callCount := 0
		downloader := func(ctx context.Context, paths []string) ([][]byte, error) {
			callCount++
			if callCount > 1 {
				return nil, assert.AnError
			}
			return lo.Map(paths, func(p string, _ int) []byte {
				return []byte("data")
			}), nil
		}
		paths := [][]string{
			{"file1.parquet"},
			{"file2.parquet"},
		}
		_, _, err := prefetchToLocal(ctx, downloader, paths)
		assert.Error(t, err)
		// Temp dir should have been cleaned up (we can't easily check since it's removed)
	})
}

// mockRecordReader is a simple mock for RecordReader used in prefetchRecordReader tests.
type mockRecordReader struct {
	closed bool
}

func (m *mockRecordReader) Next() (Record, error) {
	return nil, io.EOF
}

func (m *mockRecordReader) Close() error {
	m.closed = true
	return nil
}

func Test_prefetchRecordReader(t *testing.T) {
	t.Run("close cleans up temp dir", func(t *testing.T) {
		tempDir, err := os.MkdirTemp("", "test-prefetch-*")
		assert.NoError(t, err)
		// Verify dir exists
		_, err = os.Stat(tempDir)
		assert.NoError(t, err)

		inner := &mockRecordReader{}
		reader := &prefetchRecordReader{inner: inner, tempDir: tempDir}

		err = reader.Close()
		assert.NoError(t, err)
		assert.True(t, inner.closed)

		// Verify temp dir was removed
		_, err = os.Stat(tempDir)
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("close without temp dir", func(t *testing.T) {
		inner := &mockRecordReader{}
		reader := &prefetchRecordReader{inner: inner, tempDir: ""}
		err := reader.Close()
		assert.NoError(t, err)
		assert.True(t, inner.closed)
	})

	t.Run("next delegates to inner", func(t *testing.T) {
		inner := &mockRecordReader{}
		reader := &prefetchRecordReader{inner: inner}
		_, err := reader.Next()
		assert.Equal(t, io.EOF, err)
	})
}

func Test_WithPrefetch(t *testing.T) {
	opts := DefaultReaderOptions()
	assert.False(t, opts.prefetch)

	WithPrefetch()(opts)
	assert.True(t, opts.prefetch)
}
