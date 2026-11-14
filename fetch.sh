#!/bin/sh
# The books in books.txt, as archive.org's OCR text, into data/raw/.
set -e
mkdir -p data/raw
grep -v '^#' books.txt | awk 'NF {print $1}' | while read -r id; do
  [ -s "data/raw/$id.txt" ] || curl -sL -o "data/raw/$id.txt" "https://archive.org/download/$id/${id}_djvu.txt"
  echo "$id: $(wc -w < "data/raw/$id.txt") words"
done
