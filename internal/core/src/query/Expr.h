// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once
#include <memory>
#include <vector>
#include <any>
#include <string>
#include <optional>
#include <map>
#include "common/Schema.h"

namespace milvus::query {
class ExprVisitor;

// Base of all Exprs
struct Expr {
 public:
    virtual ~Expr() = default;
    virtual void
    accept(ExprVisitor&) = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

struct BinaryExpr : Expr {
    ExprPtr left_;
    ExprPtr right_;
};

struct UnaryExpr : Expr {
    ExprPtr child_;
};

struct BoolUnaryExpr : UnaryExpr {
    enum class OpType { Invalid = 0, LogicalNot = 1 };
    OpType op_type_;

 public:
    void
    accept(ExprVisitor&) override;
};

struct BoolBinaryExpr : BinaryExpr {
    // Note: bitA - bitB == bitA & ~bitB, alias to LogicalMinus
    enum class OpType { Invalid = 0, LogicalAnd = 1, LogicalOr = 2, LogicalXor = 3, LogicalMinus = 4 };
    OpType op_type_;

 public:
    void
    accept(ExprVisitor&) override;
};

struct TermExpr : Expr {
    FieldOffset field_offset_;
    DataType data_type_ = DataType::NONE;
    // std::vector<std::any> terms_;

 protected:
    // prevent accidential instantiation
    TermExpr() = default;

 public:
    void
    accept(ExprVisitor&) override;
};

struct RangeExpr : Expr {
    FieldOffset field_offset_;
    DataType data_type_ = DataType::NONE;
    enum class OpType {
        Invalid = 0,
        GreaterThan = 1,
        GreaterEqual = 2,
        LessThan = 3,
        LessEqual = 4,
        Equal = 5,
        NotEqual = 6
    };
    static const std::map<std::string, OpType> mapping_;  // op_name -> op

    // std::vector<std::tuple<OpType, std::any>> conditions_;
 protected:
    // prevent accidential instantiation
    RangeExpr() = default;

 public:
    void
    accept(ExprVisitor&) override;
};
}  // namespace milvus::query
