//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_TRANSFORMS_VITIS_EXPORT_C_FILE_CONTENT_H
#define SPECHLS_INCLUDED_TRANSFORMS_VITIS_EXPORT_C_FILE_CONTENT_H

#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

using namespace std;
using namespace mlir;

struct CFileContent {

private:
  std::map<std::string, std::string> opToId;
  std::map<std::string, std::string> valueToId;

  u_int32_t id = 0;
  u_int32_t vid = 0;
  string path;
  string name;
  vector<string> includes;
  vector<string> declarations;
  vector<string> init;
  vector<string> syncUpdate;
  vector<string> combUpdate;

public:
  CFileContent(string path, string filename) { // Constructor
    this->name = filename;
    this->path = path;
  }

  bool save();
  string getOpId(mlir::Operation *op);
  string getValueId(mlir::Value *v);

  void appendIncludesUpdate(string line);
  void appendDeclarations(string line);
  void appendCombUpdate(string line);
  void appendSyncUpdate(string line);
  void appendInitUpdate(string line);
};

#endif // SPECHLS_INCLUDED_TRANSFORMS_VITIS_EXPORT_C_FILE_CONTENT_H
