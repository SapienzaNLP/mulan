#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse
import statistics
import os
import pickle


def main(language):

    cache_folder = f'cache/synset2lemmas'
    vocabulary_path = f'vocabs/lemma2synsets.{language.lower()}.txt'

    language = language.lower()
    cache_path = f'{cache_folder}/{language}.pickle'

    assert not os.path.exists(cache_path)

    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    synset2language_lemmas = {}

    with open(vocabulary_path) as f:

        for line in f:

            parts = line.strip().split('\t')

            lemma = parts[0][: parts[0].rindex('#')].lower().replace('_', ' ')
            synsets = parts[1:]

            for synset in synsets:

                if synset not in synset2language_lemmas:
                    synset2language_lemmas[synset] = set()

                synset2language_lemmas[synset].add(lemma)

    with open(cache_path, 'wb') as f:
        pickle.dump(synset2language_lemmas, f)

    return synset2language_lemmas


def parse_args():
    parser = argparse.ArgumentParser(description='Load')
    parser.add_argument("--language", required=True, type=str, help='Language the mapping synset -> valid lemmas has to be built for')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    result = main(args.language)
    print(f'# {args.language.lower()}')
    print(f'\t # synsets: {len(result.keys())}')
    print(f'\t # lemmas: {len(set([_e for e in result.values() for _e in e]))}')
    print(f'\t # lemmas per synset: {statistics.mean([len(v) for k, v in result.items()])}')

