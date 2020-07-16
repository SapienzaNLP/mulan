#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import os
import pickle
from pathlib import Path
from typing import Union, Dict, Set

from tqdm import tqdm

from db.level_db import LevelDB
from encoding.base import AlreadyTokenizedAligner
from encoding.transformers_encoders import TransformerEncoder
from transfer.entities import TextSpan


def normalize_line(line: str) -> str:
    return line.strip() \
        .replace('´', '\'') \
        .replace('`', '\'') \
        .replace('‘', '\'') \
        .replace('’', '\'') \
        .replace('“', '"') \
        .replace('”', '"') \
        .replace('—', '-') \
        .replace('–', '-') \
        .replace('…', '...')


def to_text_span(id: str, db: LevelDB) -> Union[TextSpan, None]:

    document_id, sentence_id, start_index, end_index = id.split('.')
    start_index = int(start_index)
    end_index = int(end_index)

    sentence = db[f'{document_id}.{sentence_id}']

    if start_index == -1:
        return TextSpan(None, None, None, -1, -1, None, sentence, sentence_id, document_id)

    tokens = sentence.split('\t')
    annotated_tokens = [token.split(' ') for token in tokens]

    span_tokens = annotated_tokens[start_index: end_index]

    if len(span_tokens) > 1 or len(span_tokens[0]) < 4:
        annotation = None
    else:
        annotation = span_tokens[0][3]

    return TextSpan(
        [e[0] for e in span_tokens],
        '_'.join([e[2] for e in span_tokens]),
        '_'.join([e[1] for e in span_tokens]),
        start_index,
        end_index,
        annotation,
        sentence,
        sentence_id,
        document_id
    )


def load_corpus(corpus_folder: str, cache_folder: str) -> LevelDB:

    db = LevelDB.get_instance(
        Path(cache_folder),
        key_transform=lambda x: x.encode(),
        key_back_transform=lambda x: x.decode(),
        value_transform=lambda x: x.encode(),
        value_back_transform=lambda x: x.decode()
    )

    # todo better approach

    try:

        next(db.db.iterator())

    except StopIteration:

        entries_added = 0
        iterator = tqdm(list(os.listdir(corpus_folder)), desc='Building corpus db')

        for corpus_instance in iterator:

            corpus_instance = f'{corpus_folder}/{corpus_instance}'

            with open(corpus_instance) as f:

                for line in f:

                    try:

                        line = normalize_line(line)

                        idx = line.index('\t')
                        id = line[: idx]
                        text = line[idx + 1:]

                        db[id] = text

                        entries_added += 1
                        iterator.set_postfix(entries_added=entries_added)

                    except ValueError:

                        continue

    return db
