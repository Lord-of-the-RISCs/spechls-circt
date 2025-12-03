#ifndef SPECHLS_DELAY_UTILS_H
#define SPECHLS_DELAY_UTILS_H

#include <mlir/IR/Dialect.h>

template <typename T, typename F>
void walkOnDelay(T &&block, F &&fun);

#endif
