#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse
import pickle
import random
import statistics
from typing import Set, List, Dict, Tuple

from utils.nlp.wsd import read_lemma2synsets


def synset_stats(dataset: Set[str], test: Set[str], vocabulary: Dict):

    print(f'# dataset: {len(dataset)}')
    print(f'# test: {len(test)}')
    print(f'# dataset & test: {len(dataset & test)}')
    print(f'# test - dataset: {len(test - dataset)}')
    print()

    print(f'# non-mapped synsets: {len(test - dataset)}')
    for synset in random.sample(test - dataset, 10):
        print(f'* {synset}: {vocabulary.get(synset, set())}')


def lemma_stats(dataset: Dict[str, Set[str]], test: Dict[str, Set[str]]):

    print(f'# dataset: {len(dataset)}')
    print(f'# test: {len(test)}')
    print(f'# dataset & test: {len(set(dataset.keys()) & set(test.keys()))}')
    print(f'# test - dataset: {len(set(set(test.keys()) - dataset.keys()))}')
    print()

    print(f'# synsets per lemma:')
    print(f'\t- dataset: {statistics.mean([len(synsets) for synsets in dataset.values()])}')
    print(f'\t- test: {statistics.mean([len(synsets) for synsets in test.values()])}')
    print()

    print(f'# non-mapped lemmas: {len(set(test.keys()) - set(dataset.keys()))}')
    for lemma in random.sample(set(test.keys()) - set(dataset.keys()), 10):
        print(f'* {lemma}')


def lemma_synset_stats(dataset: Set[Tuple[str, str]], test: Set[Tuple[str, str]]):

    print(f'# dataset: {len(dataset)}')
    print(f'# test: {len(test)}')
    print(f'# dataset & test: {len(dataset & test)}')

    print(f'# non-mapped (synset, lemma): {len(test - dataset)}')
    for synset, lemma in random.sample(test - dataset, 10):
        print(f'* ({synset}, {lemma})')


def main(key_file: str, test_key_file: str, vocabulary_file: str):

    with open(vocabulary_file, 'rb') as f:
        vocabulary = pickle.load(f)

    dataset = read_lemma2synsets(key_file)
    test = read_lemma2synsets(test_key_file)

    print(f'###############')
    print(f'### synsets ###')
    print(f'###############')
    print()

    synset_stats(
        set([synset for synsets in dataset.values() for synset in synsets]),
        set([synset for synsets in test.values() for synset in synsets]),
        vocabulary
    )
    print()

    print(f'##############')
    print(f'### lemmas ###')
    print(f'##############')
    print()

    lemma_stats(dataset, test)
    print()

    print(f'#####################')
    print(f'### lemma2synsets ###')
    print(f'#####################')
    print()

    lemma_synset_stats(
        set([(lemma, synset) for lemma, synsets in dataset.items() for synset in synsets]),
        set([(lemma, synset) for lemma, synsets in test.items() for synset in synsets])
    )
    print()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-l", type=str, required=True, help='Language whose stats are to be computed')

    parser.add_argument("--dataset-key-file", type=str, required=False, help='Raganato key file')
    parser.add_argument("--test-key-file", type=str, required=False, help='Raganato key file')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    main(
        key_file=args.dataset_key_file,
        test_key_file=args.test_key_file,
        vocabulary_file=f'cache/synset2lemmas/{args.l}.pickle'
    )
