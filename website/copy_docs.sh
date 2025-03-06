#!/bin/bash -eu

input_path=$1
output_path=$2

find $input_path -type d | grep -v includes/img | while read dir; do
  destdir=$(realpath --relative-to=$input_path "$dir")
  mkdir -p "${output_path}/$destdir"
  if ls $dir/*.md 1>/dev/null 2>&1; then
    cat >"${output_path}/$destdir/_index.md" <<EOF
---
 title: "${destdir##*/}"
 date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
 draft: false
---
EOF
  fi
done

cat >"${output_path}/_index.md" <<EOF
---
 title: "Code Documentation"
 date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
 draft: false
---
EOF

find $input_path -name "*.md" | while read file; do
  file=$(realpath --relative-to=$input_path $file)
  title=$(grep -m 1 "^# " $input_path/$file | sed 's/^# //') &&
    (echo "---" &&
      echo "title: \"$title\"" &&
      echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" &&
      echo "draft: false" &&
      echo "---" &&
      grep -v "^# $title" $input_path/$file | sed 's|\[TOC\]|<p/>{{< toc >}}|') >$output_path/$file &&
    echo "Processed $file"
done
