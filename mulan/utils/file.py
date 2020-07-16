#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import pickle
import subprocess

import hashlib
import numpy as np
import os
import xml.etree.cElementTree as ET

from typing import Iterable, Tuple, List

from tqdm import tqdm

from db.level_db import LevelDB
from transfer.entities import TextSpan
from transfer.transfer_utils import to_text_span


def yield_batches_from_vector_folder(folder: str, batch_size=64) -> Iterable[Tuple[List[str], np.array]]:

    def extract_batch_number(file_path):
        return int(file_path.split('_')[-1][: -4])

    id_files = []
    vectors_files = []

    for f in os.listdir(folder):
        if f.endswith('.txt'):
            id_files.append(f'{folder}/{f}')
        elif f.endswith('.npy'):
            vectors_files.append(f'{folder}/{f}')

    id_files = {extract_batch_number(id_file): id_file for id_file in id_files}
    vectors_files = {extract_batch_number(vectors_file): vectors_file for vectors_file in vectors_files}

    pairing = []

    for n in set(id_files.keys()) & set(vectors_files.keys()):
        pairing.append((id_files[n], vectors_files[n]))

    batch_ids = []
    batch_vectors = []

    for id_file, vector_file in pairing:

        with open(id_file) as f:
            ids = []
            for line in f:
                line = line.strip()
                ids.append(line[: line.index('\t')])

        vectors = np.load(vector_file)

        assert len(ids) == len(vectors)

        for id, vector in zip(ids, vectors):

            batch_ids.append(id)
            batch_vectors.append(vector)

            if len(batch_ids) % batch_size == 0:
                yield batch_ids, np.array(batch_vectors)
                batch_ids = []
                batch_vectors = []

    if batch_ids != []:
        yield batch_ids, np.array(batch_vectors)


def count_lines_in_file(path):
    return int(subprocess.check_output(f"wc -l \"{path}\"", shell=True).split()[0])


def read_cache_iterable(iterable: Iterable, name: str, cache_folder='/tmp'):

    cache_path = f'{cache_folder}/{hashlib.md5(name.encode()).hexdigest()[:6]}.cache'

    if not os.path.exists(cache_path):

        with open(cache_path, 'wb') as f:
            pickle.dump(list(iterable), f)

    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def load_coordinates(path: str, source_db: LevelDB, target_db: LevelDB, column_idx: int = 2, use_tqdm: bool = False) -> Iterable[Tuple[TextSpan, List[Tuple[TextSpan, float]]]]:

    target_text_span_cache = {}

    with open(path) as f:

        if use_tqdm:
            iterator = tqdm(enumerate(f), total=count_lines_in_file(path), desc='Reading coordinates')
        else:
            iterator = enumerate(f)

        for i, line in iterator:

            line = line.strip()

            parts = line.split('\t')

            transfer_source_id = parts[0]
            transfer_source = to_text_span(transfer_source_id, source_db)

            if transfer_source is None:
                continue

            transfer_targets = []

            if len(parts) < column_idx + 1:
                continue

            if parts[column_idx] == '':
                continue

            for part in parts[column_idx].split(' '):

                transfer_target_id, transfer_target_score = part.split('#')
                if transfer_target_id not in target_text_span_cache:
                    target_text_span_cache[transfer_target_id] = to_text_span(transfer_target_id, target_db)
                transfer_target = target_text_span_cache[transfer_target_id]

                if transfer_target is None:
                    continue

                transfer_target_score = float(transfer_target_score)
                transfer_targets.append((transfer_target, transfer_target_score))

            yield transfer_source, transfer_targets
