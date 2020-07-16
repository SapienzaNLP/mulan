#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from dataclasses import dataclass
from typing import List, Union


@dataclass
class TextSpan:
    tokens: List[str]
    pos: str
    lemma: str
    start_sentence_index: int
    end_sentence_index: int
    annotation: Union[str, None]
    sentence: str
    sentence_id: str
    document_id: str

    @property
    def full_id(self):
        return f'{self.document_id}.{self.sentence_id}'
