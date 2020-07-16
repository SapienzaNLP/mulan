#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import List, Tuple, Iterator, Any

from nltk import ngrams as backend_ngram


def pair_tokenizations(l1: List[str], l2: List[str]) -> List[Tuple[List[str], List[str]]]:

    def rec_pair(temp_l1: List[str], temp_l2: List[str], l1it: Iterator[str], l2it: Iterator[str]) -> Tuple[List[str], List[str]]:

        if ''.join(temp_l1) == ''.join(temp_l2):
            return temp_l1, temp_l2

        min_length = min(len(''.join(temp_l1)), len(''.join(temp_l2)))

        if ''.join(temp_l1)[: min_length] != ''.join(temp_l2)[: min_length]:
            raise ValueError(f'Alignment cannot be completed: {temp_l1} and {temp_l2} can never be aligned. State: pairing {pairing} l1 {l1} l2 {l2}')

        if len(''.join(temp_l1)) < len(''.join(temp_l2)):
            return rec_pair(temp_l1 + [next(l1it)], temp_l2, l1it, l2it)
        else:
            return rec_pair(temp_l1, temp_l2 + [next(l2it)], l1it, l2it)

    pairing = []

    l1it = iter(l1)
    l2it = iter(l2)

    while True:
        try:
            pairing.append(rec_pair([next(l1it)], [next(l2it)], l1it, l2it))
        except StopIteration:
            break

    if len(pairing) == 0 or l1[-1] != pairing[-1][0][-1] or l2[-1] != pairing[-1][1][-1]:
        raise ValueError(f'Alignment failed: {pairing} was reached but l1 was {l1} and l2 was {l2}')

    return pairing


def ngram(text: List[Any], n: int):
    return backend_ngram(text, n)
