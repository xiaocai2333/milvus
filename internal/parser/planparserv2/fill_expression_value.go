package planparserv2

import (
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
)

func FillExpressionValue(expr *planpb.Expr, values map[string]*milvuspb.GenericValue) error {

	panic("implement me")
}

func FillTermExpressionValue(expr *planpb.TermExpr, values map[string]*milvuspb.GenericValue) error {
	value, ok := values[expr.GetPlaceholderName()]
	if !ok {
		return fmt.Errorf("placeholder %s is not found", expr.GetPlaceholderName())
	}

	arrays := value.GetArrayVal()
	if arrays == nil {
		return fmt.Errorf("term elements is zero")
	}
	return nil
}
