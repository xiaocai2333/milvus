package planparserv2

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/milvus-io/milvus/internal/proto/planpb"

	"github.com/stretchr/testify/suite"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func Test_ConvertFieldDataToGenericValue(t *testing.T) {
	toJSONBytes := func(v any) []byte {
		jsonBytes, err := json.Marshal(v)
		assert.Nil(t, err)
		return jsonBytes
	}
	scalarDatas := []*schemapb.FieldData{
		{
			Type:      schemapb.DataType_Int32,
			FieldName: "int",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{1, 2, 3, 4, 5, 6},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Int64,
			FieldName: "int64",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{1, 2, 3, 4, 5, 6},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Bool,
			FieldName: "bool",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_BoolData{
						BoolData: &schemapb.BoolArray{
							Data: []bool{true, false, true, false},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Float,
			FieldName: "float",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_FloatData{
						FloatData: &schemapb.FloatArray{
							Data: []float32{1.1, 2.2, 3.3, 4.4, 5.6},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Double,
			FieldName: "float",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_DoubleData{
						DoubleData: &schemapb.DoubleArray{
							Data: []float64{1.1, 2.2, 3.3, 4.4, 5.6},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_String,
			FieldName: "string",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{
						StringData: &schemapb.StringArray{
							Data: []string{"abc", "def", "ghi"},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_String,
			FieldName: "string",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{
						StringData: &schemapb.StringArray{
							Data: []string{"abc", "def", "ghi"},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Array,
			FieldName: "int64_array",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_ArrayData{
						ArrayData: &schemapb.ArrayArray{
							ElementType: schemapb.DataType_Int64,
							Data: []*schemapb.ScalarField{
								{
									Data: &schemapb.ScalarField_LongData{
										LongData: &schemapb.LongArray{
											Data: []int64{1, 2, 3, 4, 5, 6},
										},
									},
								},
								{
									Data: &schemapb.ScalarField_LongData{
										LongData: &schemapb.LongArray{
											Data: []int64{2, 3, 4, 5, 6, 7},
										},
									},
								},
								{
									Data: &schemapb.ScalarField_LongData{
										LongData: &schemapb.LongArray{
											Data: []int64{3, 4, 5, 6, 7, 8},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Array,
			FieldName: "string_array",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_ArrayData{
						ArrayData: &schemapb.ArrayArray{
							ElementType: schemapb.DataType_String,
							Data: []*schemapb.ScalarField{
								{
									Data: &schemapb.ScalarField_StringData{
										StringData: &schemapb.StringArray{
											Data: []string{"abc", "def", "ghi"},
										},
									},
								},
								{
									Data: &schemapb.ScalarField_StringData{
										StringData: &schemapb.StringArray{
											Data: []string{"jkl", "opq", "rst"},
										},
									},
								},
								{
									Data: &schemapb.ScalarField_StringData{
										StringData: &schemapb.StringArray{
											Data: []string{"uvw", "xyz"},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			Type:      schemapb.DataType_Array,
			FieldName: "json_array",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_ArrayData{
						ArrayData: &schemapb.ArrayArray{
							ElementType: schemapb.DataType_JSON,
							Data: []*schemapb.ScalarField{
								{
									Data: &schemapb.ScalarField_JsonData{
										JsonData: &schemapb.JSONArray{
											Data: [][]byte{toJSONBytes("abc"), toJSONBytes(1), toJSONBytes(2.2)},
										},
									},
								},
								{
									Data: &schemapb.ScalarField_JsonData{
										JsonData: &schemapb.JSONArray{
											Data: [][]byte{toJSONBytes("def"), toJSONBytes(100), toJSONBytes(22.2)},
										},
									},
								},
								{
									Data: &schemapb.ScalarField_JsonData{
										JsonData: &schemapb.JSONArray{
											Data: [][]byte{toJSONBytes("100"), toJSONBytes(100), toJSONBytes(99.99)},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, data := range scalarDatas {
		values, err := ConvertFieldDataToGenericValue(data.GetScalars(), data.GetType())
		assert.Nil(t, err)
		fmt.Println(values)
	}
}

type FillExpressionValueSuite struct {
	suite.Suite
}

func (s *FillExpressionValueSuite) SetupTest() {

}

func TestFillExpressionValue(t *testing.T) {
	suite.Run(t, new(FillExpressionValueSuite))
}

type testcase struct {
	expr   string
	values map[string]interface{}
}

func (s *FillExpressionValueSuite) TestTermExpr() {
	testcases := []testcase{
		{`Int64Field in {age}`, map[string]interface{}{"age": []int{1, 2, 3}}},
		{`A in {list}`, map[string]interface{}{"list": []interface{}{1, "abc", 2.2, false}}},
	}
	schemaH := newTestSchemaHelper(s.T())

	for _, c := range testcases {
		values, err := json.Marshal(c.values)
		s.NoError(err)
		plan, err := CreateSearchPlan(schemaH, c.expr, "FloatVectorField", &planpb.QueryInfo{
			Topk:         0,
			MetricType:   "",
			SearchParams: "",
			RoundDecimal: 0,
		}, values)

		s.NoError(err)
		fmt.Println(plan)
	}
}
