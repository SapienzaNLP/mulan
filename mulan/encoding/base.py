#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np

from utils.nlp.commons import pair_tokenizations


class Encoder(ABC):

    @abstractmethod
    def __call__(self, sentences: List[str], batch_size=4) -> List[List[Tuple[str, np.array]]]:
        pass


class AlreadyTokenizedAligner(Encoder):

    def __init__(self, encoder: Encoder, agglomeration_strategy: Callable[[List[np.array]], np.array]):
        self.encoder = encoder
        self.agglomeration_strategy = agglomeration_strategy

    def __call__(self, sentences: List[str], batch_size=4, tokenized_sentences: List[List[str]]=None) -> List[List[Tuple[str, np.array]]]:

        if tokenized_sentences is None:
            raise ValueError('tokenized_sentences is None but AlreadyTokenizerAligner requires it')

        encoding = self.encoder(sentences, batch_size=batch_size)
        aligned_encoding = []

        for tokenized_sentence, encoded_sentence in zip(tokenized_sentences, encoding):

            encoded_tokens = [e[0] for e in encoded_sentence]

            vectors = [e[1] for e in encoded_sentence]
            vectors_it = iter(vectors)

            align = pair_tokenizations([token.replace(' ', '') for token in tokenized_sentence], encoded_tokens)
            aligned_sentence = []

            for original_tokens, encoded_tokens in align:

                assert len(original_tokens) == 1, f'len(original tokens) > 1: ({original_tokens}, {encoded_tokens})'

                original_token = original_tokens[0]
                aligned_sentence.append((
                    original_token,
                    self.agglomeration_strategy([next(vectors_it) for _ in encoded_tokens])
                ))

            aligned_encoding.append(aligned_sentence)

        return aligned_encoding
