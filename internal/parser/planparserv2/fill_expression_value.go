package planparserv2

import (
	"encoding/json"
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
)

func ConvertFieldDataToGenericValue(data *schemapb.ScalarField, dataType schemapb.DataType) ([]*planpb.GenericValue, error) {
	if data == nil {
		return nil, fmt.Errorf("convert field data is nil")
	}
	values := make([]*planpb.GenericValue, 0)
	switch dataType {
	case schemapb.DataType_Bool:
		elements := data.GetBoolData().GetData()
		for _, element := range elements {
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_BoolVal{
					BoolVal: element,
				},
			})
		}
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		elements := data.GetIntData().GetData()
		for _, element := range elements {
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_Int64Val{
					Int64Val: int64(element),
				},
			})
		}
	case schemapb.DataType_Int64:
		elements := data.GetLongData().GetData()
		for _, element := range elements {
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_Int64Val{
					Int64Val: element,
				},
			})
		}
	case schemapb.DataType_Float:
		elements := data.GetFloatData().GetData()
		for _, element := range elements {
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_FloatVal{
					FloatVal: float64(element),
				},
			})
		}
	case schemapb.DataType_Double:
		elements := data.GetDoubleData().GetData()
		for _, element := range elements {
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_FloatVal{
					FloatVal: element,
				},
			})
		}
	case schemapb.DataType_String:
		elements := data.GetStringData().GetData()
		for _, element := range elements {
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_StringVal{
					StringVal: element,
				},
			})
		}
	case schemapb.DataType_Array:
		elements := data.GetArrayData().GetData()
		for _, element := range elements {
			arrayElements, err := ConvertFieldDataToGenericValue(element, data.GetArrayData().GetElementType())
			if err != nil {
				return nil, err
			}
			values = append(values, &planpb.GenericValue{
				Val: &planpb.GenericValue_ArrayVal{
					ArrayVal: &planpb.Array{
						Array:       arrayElements,
						SameType:    data.GetArrayData().GetElementType() != schemapb.DataType_JSON,
						ElementType: data.GetArrayData().GetElementType(),
					},
				},
			})
		}
	case schemapb.DataType_JSON:
		elements := data.GetJsonData().GetData()
		for _, element := range elements {
			var j interface{}
			err := json.Unmarshal(element, &j)
			if err != nil {
				return nil, err
			}
			switch v := j.(type) {
			case bool:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_BoolVal{
						BoolVal: v,
					},
				})
			case int:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_Int64Val{
						Int64Val: int64(v),
					},
				})
			case int32:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_Int64Val{
						Int64Val: int64(v),
					},
				})
			case int64:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_Int64Val{
						Int64Val: v,
					},
				})
			case float32:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_FloatVal{
						FloatVal: float64(v),
					},
				})
			case float64:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_FloatVal{
						FloatVal: v,
					},
				})
			case string:
				values = append(values, &planpb.GenericValue{
					Val: &planpb.GenericValue_StringVal{
						StringVal: v,
					},
				})
			default:
				return nil, fmt.Errorf("unknown element type: %s", v)
			}
		}
	default:
		return nil, fmt.Errorf("expression elements can only be scalars")

	}

	return values, nil
}

func FillExpressionValue(expr *planpb.Expr, data map[string]*planpb.GenericValue) error {
	switch e := expr.GetExpr().(type) {
	case *planpb.Expr_TermExpr:
		return FillTermExpressionValue(e.TermExpr, data)
	case *planpb.Expr_UnaryExpr:
	case *planpb.Expr_BinaryExpr:
	case *planpb.Expr_CompareExpr:
	case *planpb.Expr_UnaryRangeExpr:
	case *planpb.Expr_BinaryRangeExpr:
	case *planpb.Expr_BinaryArithOpEvalRangeExpr:
	case *planpb.Expr_BinaryArithExpr:
	case *planpb.Expr_ValueExpr:
	case *planpb.Expr_ColumnExpr:
	case *planpb.Expr_AlwaysTrueExpr:
	case *planpb.Expr_ExistsExpr:
	case *planpb.Expr_JsonContainsExpr:

	default:

	}
	return nil
}

func FillTermExpressionValue(expr *planpb.TermExpr, data map[string]*planpb.GenericValue) error {
	if expr.GetValues() == nil {
		value, ok := data[expr.GetPlaceholderName()]
		if !ok && expr.GetValues() == nil {
			return fmt.Errorf("placeholder %s is not found", expr.GetPlaceholderName())
		}

		if value == nil || value.GetArrayVal() == nil || len(value.GetArrayVal().GetArray()) == 0 {
			return fmt.Errorf("expression value: %s is nil", expr.GetPlaceholderName())
		}

		expr.Values = value.GetArrayVal().GetArray()
	}

	return nil
}

func FillUnaryRangeExpressionValue(expr *planpb.UnaryRangeExpr, data map[string]*planpb.GenericValue) error {
	if expr.GetValue() == nil {
		value, ok := data[expr.GetPlaceholderName()]
		if !ok {
			return fmt.Errorf("placeholder %s is not found", expr.GetPlaceholderName())
		}

		castedValue, err := castValue(expr.GetColumnInfo().GetDataType(), value)
		if err != nil {
			return err
		}
		expr.Value = castedValue
	}
	return nil
}

func FillBinaryRangeExpressionValue(expr *planpb.BinaryRangeExpr, data map[string]*planpb.GenericValue) error {
	if expr.GetLowerValue() == nil {
		lowerValue, ok := data[expr.GetLowerPlaceholderName()]
		if !ok {
			return fmt.Errorf("lower placeholder %s is not found", expr.GetLowerPlaceholderName())
		}

		castedLowerValue, err := castValue(expr.GetColumnInfo().GetDataType(), lowerValue)
		if err != nil {
			return err
		}
		expr.LowerValue = castedLowerValue
	}

	if expr.GetUpperValue() == nil {
		upperValue, ok := data[expr.GetUpperPlaceholderName()]
		if !ok {
			return fmt.Errorf("upper placeholder %s is not found", expr.GetUpperPlaceholderName())
		}

		castedUpperValue, err := castValue(expr.GetColumnInfo().GetDataType(), upperValue)
		if err != nil {
			return err
		}
		expr.UpperValue = castedUpperValue
	}

	return nil
}

func FillBinaryArithOpEvalRangeExpressionValue(expr *planpb.BinaryArithOpEvalRangeExpr, data map[string]*planpb.GenericValue) error {
	if expr.GetRightOperand() == nil {
		operand, ok := data[expr.GetOperandPlaceholderName()]
		if !ok {
			return fmt.Errorf("right operand: %s of BinaryArithOpEvalRangeExpression is not found", expr.GetOperandPlaceholderName())
		}
		castedOperand, err := castValue(expr.GetColumnInfo().GetDataType(), operand)
		if err != nil {
			return err
		}
		expr.RightOperand = castedOperand
	}

	if expr.GetValue() == nil {
		value, ok := data[expr.GetValuePlaceholderName()]
		if !ok {
			return fmt.Errorf("value: %s of BinaryArithOpEvalRangeExpressionis not found", expr.GetValuePlaceholderName())
		}
		castedValue, err := castValue(expr.GetColumnInfo().GetDataType(), value)
		if err != nil {
			return err
		}
		expr.Value = castedValue
	}
	return nil
}

func FillJSONContainsExpressionValue(expr *planpb.JSONContainsExpr, data map[string]*planpb.GenericValue) error {
	if expr.GetElements() == nil {
		value, ok := data[expr.GetPlaceholderName()]
		if !ok {
			return fmt.Errorf("value: %s of JSONContains is not found", expr.GetPlaceholderName())
		}
		if expr.GetOp() == planpb.JSONContainsExpr_Contains {
			castedValue, err := castValue(expr.GetColumnInfo().GetDataType(), value)
			if err != nil {
				return err
			}
			expr.Elements = append(expr.Elements, castedValue)
		} else {
			if value.GetArrayVal() == nil {
				return fmt.Errorf("expression value: %s of JSONContains is not array or nil array", expr.GetPlaceholderName())
			}
			for _, e := range value.GetArrayVal().GetArray() {
				castedValue, err := castValue(expr.GetColumnInfo().GetDataType(), e)
				if err != nil {
					return err
				}
				expr.Elements = append(expr.Elements, castedValue)
			}
		}
	}
	return nil
}
