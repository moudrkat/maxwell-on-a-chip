#!/bin/sh
# Maxwell's and Boltzmann's books, public domain, as archive.org's OCR text. Into data/raw/.
set -e
mkdir -p data/raw
for id in theoryofheat00maxwuoft scientificpapers01maxwuoft scientificpapers02maxwuoft treatiseonelectr01maxwrich \
          treatiseonelectr02maxw mattermotion00maxwuoft elementarytreati00maxwuoft \
          vorlesungenber01bolt vorlesungenber02bolt populreschrifte00boltgoog; do
  [ -s "data/raw/$id.txt" ] || curl -sL -o "data/raw/$id.txt" "https://archive.org/download/$id/${id}_djvu.txt"
  echo "$id: $(wc -w < "data/raw/$id.txt") words"
done
