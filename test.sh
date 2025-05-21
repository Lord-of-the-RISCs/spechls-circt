#!/bin/bash

./bin/spechls-opt --check-schedule $1 | grep 'Inconsistent'
if [[ $? -eq 1 ]]; then
  exit 0
else
  exit 1
fi
