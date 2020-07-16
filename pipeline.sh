#!/bin/bash

if [ "$#" -lt "3" ];then
  echo "usage: bash pipeline.sh <target-language> <source-corpus-enum> <target-corpus-enum>"
  exit 1
fi

target_language=$1
source_corpus_enum=$2
target_corpus_enum=$3

set -e

PYTHONPATH=mulan/ python mulan/transfer/1_retrieve_targets_manifesto.py --language $target_language

# vectorize source and target
PYTHONPATH=mulan/ python mulan/transfer/2_load.py $source_corpus_enum
PYTHONPATH=mulan/ python mulan/transfer/2_load.py $target_corpus_enum

# produce coordinates
coordinates_folder="coordinates/$source_corpus_enum-$target_corpus_enum"
PYTHONPATH=mulan/ python mulan/transfer/3_spot_targets.py --source-enum $source_corpus_enum --target-enum $target_corpus_enum --coordinates-folder $coordinates_folder
PYTHONPATH=mulan/ python mulan/transfer/4_compute_priorities.py --source-enum $source_corpus_enum --target-enum $target_corpus_enum --coordinates-folder $coordinates_folder

# transfer to xml
output_folder="transfer/$source_corpus_enum-$target_corpus_enum"
PYTHONPATH=mulan/ python mulan/transfer/5_word_fire.py \
    --language $target_language \
     --name "$source_corpus_enum-$target_corpus_enum" \
     --coordinates $coordinates_folder/coordinates.normalized.tsv,$source_corpus_enum,$target_corpus_enum \
     --output-folder $output_folder
