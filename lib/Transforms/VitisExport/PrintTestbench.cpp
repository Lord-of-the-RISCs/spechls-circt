//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "Transforms/VitisExport/CFileContent.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include <fstream>

using namespace std;
using namespace mlir;
using namespace SpecHLS;

using namespace circt::hw;
using namespace circt::comb;

std::string quote(string s);
string parent(string s);
string replace_all(string str, const string &r, const string &ins);
string assign(CFileContent *p, Value lhs, string rhs);

string op2str(Operation *v);
string value2str(Value *v);
string type2str(Type type);
string attr2str(Attribute attr);
string predicate2str(circt::comb::ICmpPredicate predicate);

string valueList(CFileContent *p, OperandRange range, string sep);
string argList(CFileContent *p, OperandRange range, string sep);

void printHWModule(CFileContent *p, circt::hw::HWModuleOp hwop) {

  auto bodyBlock = hwop.getBodyBlock();
  string moduleDecl;
  llvm::outs() << "Generating Testbench code for " << hwop.getSymName().str()
               << "\n";

  p->appendIncludesUpdate("#include<ac_int.h>");
  p->appendIncludesUpdate("");
  //  if (hwop->getNumResults() == 0) {
  //    moduleDecl += "void ";
  //  } else {
  //    moduleDecl += type2str(hwop->getResult(0).getType()) + " ";
  //  }

  auto name = hwop.getSymName().str();
  moduleDecl += "struct " + name + "_res {\n";
  u_int32_t nbout = 0;
  for (auto portInfo : hwop.getPortList()) {
    if (portInfo.isOutput()) {
      moduleDecl += "\t" + type2str(portInfo.type) + " " +
                    portInfo.getName().str() + ";\n";
      nbout++;
    }
  }

  moduleDecl += "};\n\n ";

  moduleDecl += "struct " + name + "_res " + name + "(";

  if (!hwop.getBody().getBlocks().empty()) {
    auto nargs = hwop.getBody().getArguments().size();
    for (u_int32_t i = 0; i < nargs; i++) {
      auto arg = hwop.getBody().getArguments()[i];
      Type arg_type = arg.getType();
      moduleDecl +=
          (i == 0 ? "" : ", ") + type2str(arg_type) + " " + p->getValueId(&arg);
    }
  }
  moduleDecl += ") {";
  p->appendDeclarations(moduleDecl);

  llvm::outs() << moduleDecl << " {\n";

  for (Operation &op : *bodyBlock) {
    llvm::outs() << "\t- print for " << op << "\n";
    printOperation(p, &op);
  }
}
