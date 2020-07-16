#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import List, Tuple, Set


def flat_list(lst: list) -> list:
    return [_e for e in lst for _e in e]


def chunk(lst: list, n: int) -> List[list]:
    """Yield successive n-sized chunks from l."""
    chunks = []
    for i in range(0, len(lst), n):
        chunks.append(lst[i:i + n])
    return chunks


def index_collection(collection: list) -> dict:
    index = dict()
    for i, elem in enumerate(collection):
        index[i] = elem
    return index


def transpose(lst: List[tuple]) -> Tuple:
    return tuple(zip(*lst))


def load_set_from_file(set_file_path: str) -> Set[str]:
    set_store = set()
    with open(set_file_path) as f:
        for line in f:
            set_store.add(line.strip())
    return set_store
