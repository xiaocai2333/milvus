// Code generated by protoc-gen-go. DO NOT EDIT.
// source: plan.proto

package planpb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	schemapb "github.com/milvus-io/milvus/internal/proto/schemapb"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type RangeExpr_OpType int32

const (
	RangeExpr_Invalid      RangeExpr_OpType = 0
	RangeExpr_GreaterThan  RangeExpr_OpType = 1
	RangeExpr_GreaterEqual RangeExpr_OpType = 2
	RangeExpr_LessThan     RangeExpr_OpType = 3
	RangeExpr_LessEqual    RangeExpr_OpType = 4
	RangeExpr_Equal        RangeExpr_OpType = 5
	RangeExpr_NotEqual     RangeExpr_OpType = 6
)

var RangeExpr_OpType_name = map[int32]string{
	0: "Invalid",
	1: "GreaterThan",
	2: "GreaterEqual",
	3: "LessThan",
	4: "LessEqual",
	5: "Equal",
	6: "NotEqual",
}

var RangeExpr_OpType_value = map[string]int32{
	"Invalid":      0,
	"GreaterThan":  1,
	"GreaterEqual": 2,
	"LessThan":     3,
	"LessEqual":    4,
	"Equal":        5,
	"NotEqual":     6,
}

func (x RangeExpr_OpType) String() string {
	return proto.EnumName(RangeExpr_OpType_name, int32(x))
}

func (RangeExpr_OpType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{3, 0}
}

type UnaryExpr_UnaryOp int32

const (
	UnaryExpr_Invalid UnaryExpr_UnaryOp = 0
	UnaryExpr_Not     UnaryExpr_UnaryOp = 1
)

var UnaryExpr_UnaryOp_name = map[int32]string{
	0: "Invalid",
	1: "Not",
}

var UnaryExpr_UnaryOp_value = map[string]int32{
	"Invalid": 0,
	"Not":     1,
}

func (x UnaryExpr_UnaryOp) String() string {
	return proto.EnumName(UnaryExpr_UnaryOp_name, int32(x))
}

func (UnaryExpr_UnaryOp) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{5, 0}
}

type BinaryExpr_BinaryOp int32

const (
	BinaryExpr_Invalid    BinaryExpr_BinaryOp = 0
	BinaryExpr_LogicalAnd BinaryExpr_BinaryOp = 1
	BinaryExpr_LogicalOr  BinaryExpr_BinaryOp = 2
)

var BinaryExpr_BinaryOp_name = map[int32]string{
	0: "Invalid",
	1: "LogicalAnd",
	2: "LogicalOr",
}

var BinaryExpr_BinaryOp_value = map[string]int32{
	"Invalid":    0,
	"LogicalAnd": 1,
	"LogicalOr":  2,
}

func (x BinaryExpr_BinaryOp) String() string {
	return proto.EnumName(BinaryExpr_BinaryOp_name, int32(x))
}

func (BinaryExpr_BinaryOp) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{6, 0}
}

type GenericValue struct {
	// Types that are valid to be assigned to Val:
	//	*GenericValue_BoolVal
	//	*GenericValue_Int64Val
	//	*GenericValue_FloatVal
	Val                  isGenericValue_Val `protobuf_oneof:"val"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *GenericValue) Reset()         { *m = GenericValue{} }
func (m *GenericValue) String() string { return proto.CompactTextString(m) }
func (*GenericValue) ProtoMessage()    {}
func (*GenericValue) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{0}
}

func (m *GenericValue) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_GenericValue.Unmarshal(m, b)
}
func (m *GenericValue) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_GenericValue.Marshal(b, m, deterministic)
}
func (m *GenericValue) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GenericValue.Merge(m, src)
}
func (m *GenericValue) XXX_Size() int {
	return xxx_messageInfo_GenericValue.Size(m)
}
func (m *GenericValue) XXX_DiscardUnknown() {
	xxx_messageInfo_GenericValue.DiscardUnknown(m)
}

var xxx_messageInfo_GenericValue proto.InternalMessageInfo

type isGenericValue_Val interface {
	isGenericValue_Val()
}

type GenericValue_BoolVal struct {
	BoolVal bool `protobuf:"varint,1,opt,name=bool_val,json=boolVal,proto3,oneof"`
}

type GenericValue_Int64Val struct {
	Int64Val int64 `protobuf:"varint,2,opt,name=int64_val,json=int64Val,proto3,oneof"`
}

type GenericValue_FloatVal struct {
	FloatVal float64 `protobuf:"fixed64,3,opt,name=float_val,json=floatVal,proto3,oneof"`
}

func (*GenericValue_BoolVal) isGenericValue_Val() {}

func (*GenericValue_Int64Val) isGenericValue_Val() {}

func (*GenericValue_FloatVal) isGenericValue_Val() {}

func (m *GenericValue) GetVal() isGenericValue_Val {
	if m != nil {
		return m.Val
	}
	return nil
}

func (m *GenericValue) GetBoolVal() bool {
	if x, ok := m.GetVal().(*GenericValue_BoolVal); ok {
		return x.BoolVal
	}
	return false
}

func (m *GenericValue) GetInt64Val() int64 {
	if x, ok := m.GetVal().(*GenericValue_Int64Val); ok {
		return x.Int64Val
	}
	return 0
}

func (m *GenericValue) GetFloatVal() float64 {
	if x, ok := m.GetVal().(*GenericValue_FloatVal); ok {
		return x.FloatVal
	}
	return 0
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*GenericValue) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*GenericValue_BoolVal)(nil),
		(*GenericValue_Int64Val)(nil),
		(*GenericValue_FloatVal)(nil),
	}
}

type QueryInfo struct {
	Topk                 int64    `protobuf:"varint,1,opt,name=topk,proto3" json:"topk,omitempty"`
	MetricType           string   `protobuf:"bytes,3,opt,name=metric_type,json=metricType,proto3" json:"metric_type,omitempty"`
	SearchParams         string   `protobuf:"bytes,4,opt,name=search_params,json=searchParams,proto3" json:"search_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *QueryInfo) Reset()         { *m = QueryInfo{} }
func (m *QueryInfo) String() string { return proto.CompactTextString(m) }
func (*QueryInfo) ProtoMessage()    {}
func (*QueryInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{1}
}

func (m *QueryInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_QueryInfo.Unmarshal(m, b)
}
func (m *QueryInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_QueryInfo.Marshal(b, m, deterministic)
}
func (m *QueryInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_QueryInfo.Merge(m, src)
}
func (m *QueryInfo) XXX_Size() int {
	return xxx_messageInfo_QueryInfo.Size(m)
}
func (m *QueryInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_QueryInfo.DiscardUnknown(m)
}

var xxx_messageInfo_QueryInfo proto.InternalMessageInfo

func (m *QueryInfo) GetTopk() int64 {
	if m != nil {
		return m.Topk
	}
	return 0
}

func (m *QueryInfo) GetMetricType() string {
	if m != nil {
		return m.MetricType
	}
	return ""
}

func (m *QueryInfo) GetSearchParams() string {
	if m != nil {
		return m.SearchParams
	}
	return ""
}

type ColumnInfo struct {
	FieldId              int64             `protobuf:"varint,1,opt,name=field_id,json=fieldId,proto3" json:"field_id,omitempty"`
	DataType             schemapb.DataType `protobuf:"varint,2,opt,name=data_type,json=dataType,proto3,enum=milvus.proto.schema.DataType" json:"data_type,omitempty"`
	IsPrimaryKey         bool              `protobuf:"varint,3,opt,name=is_primary_key,json=isPrimaryKey,proto3" json:"is_primary_key,omitempty"`
	IsAutoID             bool              `protobuf:"varint,4,opt,name=is_autoID,json=isAutoID,proto3" json:"is_autoID,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *ColumnInfo) Reset()         { *m = ColumnInfo{} }
func (m *ColumnInfo) String() string { return proto.CompactTextString(m) }
func (*ColumnInfo) ProtoMessage()    {}
func (*ColumnInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{2}
}

func (m *ColumnInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ColumnInfo.Unmarshal(m, b)
}
func (m *ColumnInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ColumnInfo.Marshal(b, m, deterministic)
}
func (m *ColumnInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ColumnInfo.Merge(m, src)
}
func (m *ColumnInfo) XXX_Size() int {
	return xxx_messageInfo_ColumnInfo.Size(m)
}
func (m *ColumnInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_ColumnInfo.DiscardUnknown(m)
}

var xxx_messageInfo_ColumnInfo proto.InternalMessageInfo

func (m *ColumnInfo) GetFieldId() int64 {
	if m != nil {
		return m.FieldId
	}
	return 0
}

func (m *ColumnInfo) GetDataType() schemapb.DataType {
	if m != nil {
		return m.DataType
	}
	return schemapb.DataType_None
}

func (m *ColumnInfo) GetIsPrimaryKey() bool {
	if m != nil {
		return m.IsPrimaryKey
	}
	return false
}

func (m *ColumnInfo) GetIsAutoID() bool {
	if m != nil {
		return m.IsAutoID
	}
	return false
}

type RangeExpr struct {
	ColumnInfo           *ColumnInfo        `protobuf:"bytes,1,opt,name=column_info,json=columnInfo,proto3" json:"column_info,omitempty"`
	Ops                  []RangeExpr_OpType `protobuf:"varint,2,rep,packed,name=ops,proto3,enum=milvus.proto.plan.RangeExpr_OpType" json:"ops,omitempty"`
	Values               []*GenericValue    `protobuf:"bytes,3,rep,name=values,proto3" json:"values,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *RangeExpr) Reset()         { *m = RangeExpr{} }
func (m *RangeExpr) String() string { return proto.CompactTextString(m) }
func (*RangeExpr) ProtoMessage()    {}
func (*RangeExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{3}
}

func (m *RangeExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RangeExpr.Unmarshal(m, b)
}
func (m *RangeExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RangeExpr.Marshal(b, m, deterministic)
}
func (m *RangeExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RangeExpr.Merge(m, src)
}
func (m *RangeExpr) XXX_Size() int {
	return xxx_messageInfo_RangeExpr.Size(m)
}
func (m *RangeExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_RangeExpr.DiscardUnknown(m)
}

var xxx_messageInfo_RangeExpr proto.InternalMessageInfo

func (m *RangeExpr) GetColumnInfo() *ColumnInfo {
	if m != nil {
		return m.ColumnInfo
	}
	return nil
}

func (m *RangeExpr) GetOps() []RangeExpr_OpType {
	if m != nil {
		return m.Ops
	}
	return nil
}

func (m *RangeExpr) GetValues() []*GenericValue {
	if m != nil {
		return m.Values
	}
	return nil
}

type TermExpr struct {
	ColumnInfo           *ColumnInfo     `protobuf:"bytes,1,opt,name=column_info,json=columnInfo,proto3" json:"column_info,omitempty"`
	Values               []*GenericValue `protobuf:"bytes,2,rep,name=values,proto3" json:"values,omitempty"`
	XXX_NoUnkeyedLiteral struct{}        `json:"-"`
	XXX_unrecognized     []byte          `json:"-"`
	XXX_sizecache        int32           `json:"-"`
}

func (m *TermExpr) Reset()         { *m = TermExpr{} }
func (m *TermExpr) String() string { return proto.CompactTextString(m) }
func (*TermExpr) ProtoMessage()    {}
func (*TermExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{4}
}

func (m *TermExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TermExpr.Unmarshal(m, b)
}
func (m *TermExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TermExpr.Marshal(b, m, deterministic)
}
func (m *TermExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TermExpr.Merge(m, src)
}
func (m *TermExpr) XXX_Size() int {
	return xxx_messageInfo_TermExpr.Size(m)
}
func (m *TermExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_TermExpr.DiscardUnknown(m)
}

var xxx_messageInfo_TermExpr proto.InternalMessageInfo

func (m *TermExpr) GetColumnInfo() *ColumnInfo {
	if m != nil {
		return m.ColumnInfo
	}
	return nil
}

func (m *TermExpr) GetValues() []*GenericValue {
	if m != nil {
		return m.Values
	}
	return nil
}

type UnaryExpr struct {
	Op                   UnaryExpr_UnaryOp `protobuf:"varint,1,opt,name=op,proto3,enum=milvus.proto.plan.UnaryExpr_UnaryOp" json:"op,omitempty"`
	Child                *Expr             `protobuf:"bytes,2,opt,name=child,proto3" json:"child,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *UnaryExpr) Reset()         { *m = UnaryExpr{} }
func (m *UnaryExpr) String() string { return proto.CompactTextString(m) }
func (*UnaryExpr) ProtoMessage()    {}
func (*UnaryExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{5}
}

func (m *UnaryExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_UnaryExpr.Unmarshal(m, b)
}
func (m *UnaryExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_UnaryExpr.Marshal(b, m, deterministic)
}
func (m *UnaryExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_UnaryExpr.Merge(m, src)
}
func (m *UnaryExpr) XXX_Size() int {
	return xxx_messageInfo_UnaryExpr.Size(m)
}
func (m *UnaryExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_UnaryExpr.DiscardUnknown(m)
}

var xxx_messageInfo_UnaryExpr proto.InternalMessageInfo

func (m *UnaryExpr) GetOp() UnaryExpr_UnaryOp {
	if m != nil {
		return m.Op
	}
	return UnaryExpr_Invalid
}

func (m *UnaryExpr) GetChild() *Expr {
	if m != nil {
		return m.Child
	}
	return nil
}

type BinaryExpr struct {
	Op                   BinaryExpr_BinaryOp `protobuf:"varint,1,opt,name=op,proto3,enum=milvus.proto.plan.BinaryExpr_BinaryOp" json:"op,omitempty"`
	Left                 *Expr               `protobuf:"bytes,2,opt,name=left,proto3" json:"left,omitempty"`
	Right                *Expr               `protobuf:"bytes,3,opt,name=right,proto3" json:"right,omitempty"`
	XXX_NoUnkeyedLiteral struct{}            `json:"-"`
	XXX_unrecognized     []byte              `json:"-"`
	XXX_sizecache        int32               `json:"-"`
}

func (m *BinaryExpr) Reset()         { *m = BinaryExpr{} }
func (m *BinaryExpr) String() string { return proto.CompactTextString(m) }
func (*BinaryExpr) ProtoMessage()    {}
func (*BinaryExpr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{6}
}

func (m *BinaryExpr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BinaryExpr.Unmarshal(m, b)
}
func (m *BinaryExpr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BinaryExpr.Marshal(b, m, deterministic)
}
func (m *BinaryExpr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BinaryExpr.Merge(m, src)
}
func (m *BinaryExpr) XXX_Size() int {
	return xxx_messageInfo_BinaryExpr.Size(m)
}
func (m *BinaryExpr) XXX_DiscardUnknown() {
	xxx_messageInfo_BinaryExpr.DiscardUnknown(m)
}

var xxx_messageInfo_BinaryExpr proto.InternalMessageInfo

func (m *BinaryExpr) GetOp() BinaryExpr_BinaryOp {
	if m != nil {
		return m.Op
	}
	return BinaryExpr_Invalid
}

func (m *BinaryExpr) GetLeft() *Expr {
	if m != nil {
		return m.Left
	}
	return nil
}

func (m *BinaryExpr) GetRight() *Expr {
	if m != nil {
		return m.Right
	}
	return nil
}

type Expr struct {
	// Types that are valid to be assigned to Expr:
	//	*Expr_RangeExpr
	//	*Expr_TermExpr
	//	*Expr_UnaryExpr
	//	*Expr_BinaryExpr
	Expr                 isExpr_Expr `protobuf_oneof:"expr"`
	XXX_NoUnkeyedLiteral struct{}    `json:"-"`
	XXX_unrecognized     []byte      `json:"-"`
	XXX_sizecache        int32       `json:"-"`
}

func (m *Expr) Reset()         { *m = Expr{} }
func (m *Expr) String() string { return proto.CompactTextString(m) }
func (*Expr) ProtoMessage()    {}
func (*Expr) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{7}
}

func (m *Expr) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Expr.Unmarshal(m, b)
}
func (m *Expr) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Expr.Marshal(b, m, deterministic)
}
func (m *Expr) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Expr.Merge(m, src)
}
func (m *Expr) XXX_Size() int {
	return xxx_messageInfo_Expr.Size(m)
}
func (m *Expr) XXX_DiscardUnknown() {
	xxx_messageInfo_Expr.DiscardUnknown(m)
}

var xxx_messageInfo_Expr proto.InternalMessageInfo

type isExpr_Expr interface {
	isExpr_Expr()
}

type Expr_RangeExpr struct {
	RangeExpr *RangeExpr `protobuf:"bytes,1,opt,name=range_expr,json=rangeExpr,proto3,oneof"`
}

type Expr_TermExpr struct {
	TermExpr *TermExpr `protobuf:"bytes,2,opt,name=term_expr,json=termExpr,proto3,oneof"`
}

type Expr_UnaryExpr struct {
	UnaryExpr *UnaryExpr `protobuf:"bytes,3,opt,name=unary_expr,json=unaryExpr,proto3,oneof"`
}

type Expr_BinaryExpr struct {
	BinaryExpr *BinaryExpr `protobuf:"bytes,4,opt,name=binary_expr,json=binaryExpr,proto3,oneof"`
}

func (*Expr_RangeExpr) isExpr_Expr() {}

func (*Expr_TermExpr) isExpr_Expr() {}

func (*Expr_UnaryExpr) isExpr_Expr() {}

func (*Expr_BinaryExpr) isExpr_Expr() {}

func (m *Expr) GetExpr() isExpr_Expr {
	if m != nil {
		return m.Expr
	}
	return nil
}

func (m *Expr) GetRangeExpr() *RangeExpr {
	if x, ok := m.GetExpr().(*Expr_RangeExpr); ok {
		return x.RangeExpr
	}
	return nil
}

func (m *Expr) GetTermExpr() *TermExpr {
	if x, ok := m.GetExpr().(*Expr_TermExpr); ok {
		return x.TermExpr
	}
	return nil
}

func (m *Expr) GetUnaryExpr() *UnaryExpr {
	if x, ok := m.GetExpr().(*Expr_UnaryExpr); ok {
		return x.UnaryExpr
	}
	return nil
}

func (m *Expr) GetBinaryExpr() *BinaryExpr {
	if x, ok := m.GetExpr().(*Expr_BinaryExpr); ok {
		return x.BinaryExpr
	}
	return nil
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*Expr) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*Expr_RangeExpr)(nil),
		(*Expr_TermExpr)(nil),
		(*Expr_UnaryExpr)(nil),
		(*Expr_BinaryExpr)(nil),
	}
}

type VectorANNS struct {
	IsBinary             bool       `protobuf:"varint,1,opt,name=is_binary,json=isBinary,proto3" json:"is_binary,omitempty"`
	FieldId              int64      `protobuf:"varint,2,opt,name=field_id,json=fieldId,proto3" json:"field_id,omitempty"`
	Predicates           *Expr      `protobuf:"bytes,3,opt,name=predicates,proto3" json:"predicates,omitempty"`
	QueryInfo            *QueryInfo `protobuf:"bytes,4,opt,name=query_info,json=queryInfo,proto3" json:"query_info,omitempty"`
	PlaceholderTag       string     `protobuf:"bytes,5,opt,name=placeholder_tag,json=placeholderTag,proto3" json:"placeholder_tag,omitempty"`
	XXX_NoUnkeyedLiteral struct{}   `json:"-"`
	XXX_unrecognized     []byte     `json:"-"`
	XXX_sizecache        int32      `json:"-"`
}

func (m *VectorANNS) Reset()         { *m = VectorANNS{} }
func (m *VectorANNS) String() string { return proto.CompactTextString(m) }
func (*VectorANNS) ProtoMessage()    {}
func (*VectorANNS) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{8}
}

func (m *VectorANNS) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_VectorANNS.Unmarshal(m, b)
}
func (m *VectorANNS) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_VectorANNS.Marshal(b, m, deterministic)
}
func (m *VectorANNS) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VectorANNS.Merge(m, src)
}
func (m *VectorANNS) XXX_Size() int {
	return xxx_messageInfo_VectorANNS.Size(m)
}
func (m *VectorANNS) XXX_DiscardUnknown() {
	xxx_messageInfo_VectorANNS.DiscardUnknown(m)
}

var xxx_messageInfo_VectorANNS proto.InternalMessageInfo

func (m *VectorANNS) GetIsBinary() bool {
	if m != nil {
		return m.IsBinary
	}
	return false
}

func (m *VectorANNS) GetFieldId() int64 {
	if m != nil {
		return m.FieldId
	}
	return 0
}

func (m *VectorANNS) GetPredicates() *Expr {
	if m != nil {
		return m.Predicates
	}
	return nil
}

func (m *VectorANNS) GetQueryInfo() *QueryInfo {
	if m != nil {
		return m.QueryInfo
	}
	return nil
}

func (m *VectorANNS) GetPlaceholderTag() string {
	if m != nil {
		return m.PlaceholderTag
	}
	return ""
}

type PlanNode struct {
	// Types that are valid to be assigned to Node:
	//	*PlanNode_VectorAnns
	Node                 isPlanNode_Node `protobuf_oneof:"node"`
	OutputFieldIds       []int64         `protobuf:"varint,2,rep,packed,name=output_field_ids,json=outputFieldIds,proto3" json:"output_field_ids,omitempty"`
	XXX_NoUnkeyedLiteral struct{}        `json:"-"`
	XXX_unrecognized     []byte          `json:"-"`
	XXX_sizecache        int32           `json:"-"`
}

func (m *PlanNode) Reset()         { *m = PlanNode{} }
func (m *PlanNode) String() string { return proto.CompactTextString(m) }
func (*PlanNode) ProtoMessage()    {}
func (*PlanNode) Descriptor() ([]byte, []int) {
	return fileDescriptor_2d655ab2f7683c23, []int{9}
}

func (m *PlanNode) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_PlanNode.Unmarshal(m, b)
}
func (m *PlanNode) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_PlanNode.Marshal(b, m, deterministic)
}
func (m *PlanNode) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PlanNode.Merge(m, src)
}
func (m *PlanNode) XXX_Size() int {
	return xxx_messageInfo_PlanNode.Size(m)
}
func (m *PlanNode) XXX_DiscardUnknown() {
	xxx_messageInfo_PlanNode.DiscardUnknown(m)
}

var xxx_messageInfo_PlanNode proto.InternalMessageInfo

type isPlanNode_Node interface {
	isPlanNode_Node()
}

type PlanNode_VectorAnns struct {
	VectorAnns *VectorANNS `protobuf:"bytes,1,opt,name=vector_anns,json=vectorAnns,proto3,oneof"`
}

func (*PlanNode_VectorAnns) isPlanNode_Node() {}

func (m *PlanNode) GetNode() isPlanNode_Node {
	if m != nil {
		return m.Node
	}
	return nil
}

func (m *PlanNode) GetVectorAnns() *VectorANNS {
	if x, ok := m.GetNode().(*PlanNode_VectorAnns); ok {
		return x.VectorAnns
	}
	return nil
}

func (m *PlanNode) GetOutputFieldIds() []int64 {
	if m != nil {
		return m.OutputFieldIds
	}
	return nil
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*PlanNode) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*PlanNode_VectorAnns)(nil),
	}
}

func init() {
	proto.RegisterEnum("milvus.proto.plan.RangeExpr_OpType", RangeExpr_OpType_name, RangeExpr_OpType_value)
	proto.RegisterEnum("milvus.proto.plan.UnaryExpr_UnaryOp", UnaryExpr_UnaryOp_name, UnaryExpr_UnaryOp_value)
	proto.RegisterEnum("milvus.proto.plan.BinaryExpr_BinaryOp", BinaryExpr_BinaryOp_name, BinaryExpr_BinaryOp_value)
	proto.RegisterType((*GenericValue)(nil), "milvus.proto.plan.GenericValue")
	proto.RegisterType((*QueryInfo)(nil), "milvus.proto.plan.QueryInfo")
	proto.RegisterType((*ColumnInfo)(nil), "milvus.proto.plan.ColumnInfo")
	proto.RegisterType((*RangeExpr)(nil), "milvus.proto.plan.RangeExpr")
	proto.RegisterType((*TermExpr)(nil), "milvus.proto.plan.TermExpr")
	proto.RegisterType((*UnaryExpr)(nil), "milvus.proto.plan.UnaryExpr")
	proto.RegisterType((*BinaryExpr)(nil), "milvus.proto.plan.BinaryExpr")
	proto.RegisterType((*Expr)(nil), "milvus.proto.plan.Expr")
	proto.RegisterType((*VectorANNS)(nil), "milvus.proto.plan.VectorANNS")
	proto.RegisterType((*PlanNode)(nil), "milvus.proto.plan.PlanNode")
}

func init() { proto.RegisterFile("plan.proto", fileDescriptor_2d655ab2f7683c23) }

var fileDescriptor_2d655ab2f7683c23 = []byte{
	// 893 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xac, 0x54, 0x4f, 0x6f, 0x1b, 0x45,
	0x14, 0xcf, 0xee, 0x3a, 0xf6, 0xee, 0xb3, 0xeb, 0x9a, 0xb9, 0x60, 0x08, 0x55, 0xac, 0x6d, 0x05,
	0x96, 0x50, 0x1d, 0xe1, 0x96, 0x54, 0x2a, 0x2a, 0x22, 0xa1, 0xa5, 0x89, 0xa8, 0x9c, 0xb0, 0x84,
	0x1c, 0xb8, 0xac, 0xc6, 0xbb, 0x63, 0x7b, 0xd4, 0xf1, 0xcc, 0x64, 0x76, 0xd6, 0xaa, 0x2f, 0x5c,
	0xb8, 0x71, 0xe3, 0x4b, 0x70, 0xe1, 0x0b, 0x71, 0xe7, 0x8b, 0xa0, 0x99, 0xd9, 0xd8, 0x31, 0x72,
	0x82, 0x90, 0x7a, 0x7b, 0x7f, 0xe7, 0xfd, 0x7e, 0xef, 0xbd, 0x79, 0x00, 0x92, 0x61, 0x3e, 0x90,
	0x4a, 0x68, 0x81, 0x3e, 0x98, 0x53, 0xb6, 0x28, 0x0b, 0xa7, 0x0d, 0x8c, 0xe3, 0xe3, 0x56, 0x91,
	0xcd, 0xc8, 0x1c, 0x3b, 0x53, 0x2c, 0xa1, 0xf5, 0x9a, 0x70, 0xa2, 0x68, 0x76, 0x89, 0x59, 0x49,
	0xd0, 0x1e, 0x84, 0x63, 0x21, 0x58, 0xba, 0xc0, 0xac, 0xeb, 0xf5, 0xbc, 0x7e, 0x78, 0xb2, 0x93,
	0x34, 0x8c, 0xe5, 0x12, 0x33, 0xf4, 0x00, 0x22, 0xca, 0xf5, 0xe1, 0x53, 0xeb, 0xf5, 0x7b, 0x5e,
	0x3f, 0x38, 0xd9, 0x49, 0x42, 0x6b, 0xaa, 0xdc, 0x13, 0x26, 0xb0, 0xb6, 0xee, 0xa0, 0xe7, 0xf5,
	0x3d, 0xe3, 0xb6, 0xa6, 0x4b, 0xcc, 0x8e, 0x77, 0x21, 0x58, 0x60, 0x16, 0x13, 0x88, 0x7e, 0x28,
	0x89, 0x5a, 0x9e, 0xf2, 0x89, 0x40, 0x08, 0x6a, 0x5a, 0xc8, 0xb7, 0xb6, 0x54, 0x90, 0x58, 0x19,
	0xed, 0x43, 0x73, 0x4e, 0xb4, 0xa2, 0x59, 0xaa, 0x97, 0x92, 0xd8, 0x87, 0xa2, 0x04, 0x9c, 0xe9,
	0x62, 0x29, 0x09, 0x7a, 0x08, 0xf7, 0x0a, 0x82, 0x55, 0x36, 0x4b, 0x25, 0x56, 0x78, 0x5e, 0x74,
	0x6b, 0x36, 0xa4, 0xe5, 0x8c, 0xe7, 0xd6, 0x16, 0xff, 0xe1, 0x01, 0x7c, 0x2b, 0x58, 0x39, 0xe7,
	0xb6, 0xd0, 0x47, 0x10, 0x4e, 0x28, 0x61, 0x79, 0x4a, 0xf3, 0xaa, 0x58, 0xc3, 0xea, 0xa7, 0x39,
	0x7a, 0x0e, 0x51, 0x8e, 0x35, 0x76, 0xd5, 0x0c, 0xab, 0xf6, 0xf0, 0xc1, 0x60, 0xa3, 0x6f, 0x55,
	0xc7, 0x5e, 0x62, 0x8d, 0x0d, 0x80, 0x24, 0xcc, 0x2b, 0x09, 0x3d, 0x82, 0x36, 0x2d, 0x52, 0xa9,
	0xe8, 0x1c, 0xab, 0x65, 0xfa, 0x96, 0x2c, 0x2d, 0xdc, 0x30, 0x69, 0xd1, 0xe2, 0xdc, 0x19, 0xbf,
	0x27, 0x4b, 0xb4, 0x07, 0x11, 0x2d, 0x52, 0x5c, 0x6a, 0x71, 0xfa, 0xd2, 0x82, 0x0d, 0x93, 0x90,
	0x16, 0x47, 0x56, 0x8f, 0xff, 0xf4, 0x21, 0x4a, 0x30, 0x9f, 0x92, 0x57, 0xef, 0xa4, 0x42, 0x5f,
	0x43, 0x33, 0xb3, 0xa8, 0x53, 0xca, 0x27, 0xc2, 0x42, 0x6d, 0xfe, 0x1b, 0x8e, 0x9d, 0xef, 0x9a,
	0x5b, 0x02, 0xd9, 0x9a, 0xe7, 0x97, 0x10, 0x08, 0x59, 0x74, 0xfd, 0x5e, 0xd0, 0x6f, 0x0f, 0x1f,
	0x6e, 0xc9, 0x5b, 0x95, 0x1a, 0x9c, 0x49, 0x4b, 0xc6, 0xc4, 0xa3, 0x67, 0x50, 0x5f, 0x98, 0xf9,
	0x17, 0xdd, 0xa0, 0x17, 0xf4, 0x9b, 0xc3, 0xfd, 0x2d, 0x99, 0x37, 0xf7, 0x24, 0xa9, 0xc2, 0x63,
	0x0e, 0x75, 0xf7, 0x0e, 0x6a, 0x42, 0xe3, 0x94, 0x2f, 0x30, 0xa3, 0x79, 0x67, 0x07, 0xdd, 0x87,
	0xe6, 0x6b, 0x45, 0xb0, 0x26, 0xea, 0x62, 0x86, 0x79, 0xc7, 0x43, 0x1d, 0x68, 0x55, 0x86, 0x57,
	0x57, 0x25, 0x66, 0x1d, 0x1f, 0xb5, 0x20, 0x7c, 0x43, 0x8a, 0xc2, 0xfa, 0x03, 0x74, 0x0f, 0x22,
	0xa3, 0x39, 0x67, 0x0d, 0x45, 0xb0, 0xeb, 0xc4, 0x5d, 0x13, 0x37, 0x12, 0xda, 0x69, 0xf5, 0xf8,
	0x57, 0x0f, 0xc2, 0x0b, 0xa2, 0xe6, 0xef, 0xa5, 0x59, 0x6b, 0xd6, 0xfe, 0xff, 0x63, 0xfd, 0xbb,
	0x07, 0xd1, 0x4f, 0x1c, 0xab, 0xa5, 0x85, 0xf1, 0x14, 0x7c, 0x21, 0x6d, 0xf5, 0xf6, 0xf0, 0xd1,
	0x96, 0x27, 0x56, 0x91, 0x4e, 0x3a, 0x93, 0x89, 0x2f, 0x24, 0x7a, 0x0c, 0xbb, 0xd9, 0x8c, 0xb2,
	0xdc, 0xae, 0x5c, 0x73, 0xf8, 0xe1, 0x96, 0x44, 0x93, 0x93, 0xb8, 0xa8, 0x78, 0x1f, 0x1a, 0x55,
	0xf6, 0x66, 0xa7, 0x1b, 0x10, 0x8c, 0x84, 0xee, 0x78, 0xf1, 0x5f, 0x1e, 0xc0, 0x31, 0x5d, 0x81,
	0x3a, 0xbc, 0x01, 0xea, 0xd3, 0x2d, 0x6f, 0xaf, 0x43, 0x2b, 0xb1, 0x82, 0xf5, 0x39, 0xd4, 0x18,
	0x99, 0xe8, 0xff, 0x42, 0x65, 0x83, 0x0c, 0x07, 0x45, 0xa7, 0x33, 0x6d, 0xb7, 0xfe, 0x2e, 0x0e,
	0x36, 0x2a, 0x3e, 0x84, 0xf0, 0xba, 0xd6, 0x26, 0x89, 0x36, 0xc0, 0x1b, 0x31, 0xa5, 0x19, 0x66,
	0x47, 0x3c, 0xef, 0x78, 0x76, 0x1b, 0x9c, 0x7e, 0xa6, 0x3a, 0x7e, 0xfc, 0x9b, 0x0f, 0x35, 0x4b,
	0xea, 0x05, 0x80, 0x32, 0xfb, 0x9b, 0x92, 0x77, 0x52, 0x55, 0xf3, 0xfe, 0xe4, 0xae, 0x25, 0x3f,
	0xd9, 0x49, 0x22, 0xb5, 0xfa, 0x5c, 0xcf, 0x21, 0xd2, 0x44, 0xcd, 0x5d, 0xb6, 0x23, 0xb8, 0xb7,
	0x25, 0xfb, 0x7a, 0xbf, 0xcc, 0xf5, 0xd2, 0xd7, 0xbb, 0xf6, 0x02, 0xa0, 0x34, 0xd0, 0x5d, 0x72,
	0x70, 0x6b, 0xe9, 0xd5, 0xb0, 0x4d, 0xe9, 0x72, 0x35, 0x8e, 0x6f, 0xa0, 0x39, 0xa6, 0xeb, 0xfc,
	0xda, 0xad, 0xab, 0xba, 0x9e, 0xcb, 0xc9, 0x4e, 0x02, 0xe3, 0x95, 0x76, 0x5c, 0x87, 0x9a, 0x49,
	0x8d, 0xff, 0xf6, 0x00, 0x2e, 0x49, 0xa6, 0x85, 0x3a, 0x1a, 0x8d, 0x7e, 0xac, 0x6e, 0x8b, 0x8b,
	0x73, 0x17, 0xdb, 0xdc, 0x16, 0xf7, 0xca, 0xc6, 0xd5, 0xf3, 0x37, 0xaf, 0xde, 0x33, 0x00, 0xa9,
	0x48, 0x4e, 0x33, 0xac, 0xed, 0xaf, 0xbf, 0x73, 0x7e, 0x37, 0x42, 0xd1, 0x57, 0x00, 0x57, 0xe6,
	0x7e, 0xbb, 0x3f, 0x57, 0xbb, 0xb5, 0x11, 0xab, 0x23, 0x9f, 0x44, 0x57, 0xab, 0x7b, 0xff, 0x19,
	0xdc, 0x97, 0x0c, 0x67, 0x64, 0x26, 0x58, 0x4e, 0x54, 0xaa, 0xf1, 0xb4, 0xbb, 0x6b, 0x8f, 0x77,
	0xfb, 0x86, 0xf9, 0x02, 0x4f, 0xe3, 0x5f, 0x20, 0x3c, 0x67, 0x98, 0x8f, 0x44, 0x4e, 0x4c, 0xef,
	0x16, 0x96, 0x70, 0x8a, 0x39, 0x2f, 0xee, 0xf8, 0xe6, 0xeb, 0xb6, 0x98, 0xde, 0xb9, 0x9c, 0x23,
	0xce, 0x0b, 0xd4, 0x87, 0x8e, 0x28, 0xb5, 0x2c, 0x75, 0x7a, 0xdd, 0x0e, 0xf7, 0xe5, 0x83, 0xa4,
	0xed, 0xec, 0xdf, 0xb9, 0xae, 0x14, 0xa6, 0xcb, 0x5c, 0xe4, 0xe4, 0xf8, 0xc9, 0xcf, 0x5f, 0x4c,
	0xa9, 0x9e, 0x95, 0xe3, 0x41, 0x26, 0xe6, 0x07, 0xae, 0xd4, 0x63, 0x2a, 0x2a, 0xe9, 0x80, 0x72,
	0x4d, 0x14, 0xc7, 0xec, 0xc0, 0x56, 0x3f, 0x30, 0xd5, 0xe5, 0x78, 0x5c, 0xb7, 0xda, 0x93, 0x7f,
	0x02, 0x00, 0x00, 0xff, 0xff, 0xd5, 0x7c, 0x0d, 0xc8, 0x82, 0x07, 0x00, 0x00,
}
