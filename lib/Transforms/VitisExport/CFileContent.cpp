//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Transforms/VitisExport/CFileContent.h"

using namespace std;
using namespace mlir;

bool CFileContent::save() {
  ofstream oFile;

  llvm::outs() << "Saving C code to " << path + "/" + name + ".cpp";

  oFile.open(path + "/" + name + ".cpp");

  for (auto &i : includes)
    oFile << i << "\n";

  for (auto &d : declarations)
    oFile << "\t" << d << "\n";

  oFile << "\tbool exit;\n";

  oFile << "\t\t // Initialisation update\n\n";
  oFile << "void init_" << name << "() {\n";
  for (auto &i : init)
    oFile << "\t" << i << "\n";
  oFile << "} \n";

  oFile << "\t\t // Combinational update\n\n";
  oFile << "void comb_update_" << name << "() {\n";
  for (auto &c : combUpdate)
    oFile << "\t\t" << c << "\n";
  oFile << "} \n";

  oFile << "\t\t // Synchronous update\n\n";
  oFile << "void sync_update_" << name << "() {\n";
  for (auto &s : syncUpdate)
    oFile << "\t\t" << s << "\n";
  oFile << "}\n";

  oFile.close();
  return 0;
}

template <typename T>
string op2str(T *v) {
  std::string s;
  llvm::raw_string_ostream r(s);
  v->print(r);
  return r.str();
}

string CFileContent::getOpId(mlir::Operation *p) {
  auto key = op2str(p);
  auto it = opToId.find(key);
  string res = "";
  if (it == opToId.end()) {
    res = "op_" + to_string(id);
    id = id + 1;
    opToId[key] = res;
  } else {
    res = it->second;
  }
  return res;
}

string CFileContent::getValueId(mlir::Value *p) {
  // llvm::outs() << "searching for "<<  p<< " : " << *p << "\n";
  auto key = op2str(p);
  auto it = valueToId.find(key);
  string res = "";
  if (it == valueToId.end()) {
    res = "v_" + to_string(vid);
    vid = vid + 1;
    valueToId[key] = res;
    // llvm::outs() << "\t-adding "<< res << "->" << *p << "\n";
  } else {
    // llvm::outs() << "\t-found "<< it->second << " -> " << (it->first) <<
    // "\n";
    res = it->second;
  }
  llvm::outs() << "\t-found " << res << "\n";
  return res;
}

void CFileContent::appendIncludesUpdate(string line) {
  includes.push_back(line);
}
void CFileContent::appendDeclarations(string line) {
  declarations.push_back(line);
}
void CFileContent::appendCombUpdate(string line) { combUpdate.push_back(line); }
void CFileContent::appendSyncUpdate(string line) { syncUpdate.push_back(line); }
void CFileContent::appendInitUpdate(string line) { init.push_back(line); }
