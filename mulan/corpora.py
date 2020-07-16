#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from enum import Enum

from config import VECTORIZATION_FOLDER, CORPORA_FOLDER
from db.level_db import LevelDB
from transfer.transfer_utils import load_corpus


class _Corpus:

    def __init__(self, name: str, language: str, used_model: str, is_annotated: bool = False):
        self.name = name
        self.language = language
        self.used_model = used_model
        self.is_annotated = is_annotated

    def get_db(self) -> LevelDB:
        return load_corpus(self.data_folder, f'cache/{self.name}')

    @property
    def full_name(self):
        return f'{self.name}-{self.used_model}'

    @property
    def data_folder(self):
        return CORPORA_FOLDER / self.name

    @property
    def vectorization_folder(self):
        return VECTORIZATION_FOLDER / self.full_name


class Corpus(Enum):

    SAMPLE_SOURCE_MBERT = _Corpus('sample-source', 'en', 'mbert', is_annotated=True)
    SAMPLE_TARGET_MBERT = _Corpus('sample-target', 'it', 'mbert')

    SEMCOR_MBERT = _Corpus('semcor', 'en', 'mbert', is_annotated=True)
    WNGT_MBERT = _Corpus('wngt', 'en', 'mbert', is_annotated=True)

    SEMCOR_BERT_BASE = _Corpus('semcor', 'en', 'bert-base', is_annotated=True)
    WIKI_EN_BERT_BASE = _Corpus('wiki-en', 'en', 'bert-base')

    WIKI_EN_MBERT = _Corpus('wiki-en', 'en', 'mbert')
    WIKI_IT_MBERT = _Corpus('wiki-it', 'it', 'mbert')
    WIKI_ES_MBERT = _Corpus('wiki-es', 'es', 'mbert')
    WIKI_FR_MBERT = _Corpus('wiki-fr', 'fr', 'mbert')
    WIKI_DE_MBERT = _Corpus('wiki-de', 'de', 'mbert')

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Corpus[s.upper()]
        except KeyError:
            return s
