#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
logging.basicConfig(level=logging.DEBUG)
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from encoding.base import Encoder


logger = logging.getLogger(__name__)


class TransformerEncoder(Encoder):

    def __init__(self, model_name_or_path: str, agglomeration_strategy: Callable[[List[np.array]], np.array], bpe_is_start: Callable[[str], bool], device: str = 'cpu'):
        self.agglomeration_strategy = agglomeration_strategy
        self.bpe_is_start = bpe_is_start
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path)
        self._model.to(device)
        self._model.eval()

    def __call__(self, sentences: List[str], batch_size=4) -> List[List[Tuple[str, np.array]]]:

        encoding = self.__encode(sentences, batch_size=batch_size)
        agglomerated_encoding = []

        for sentence, encoded_sentence in zip(sentences, encoding):

            bpes = [e[0] for e in encoded_sentence]
            vectors = [e[1] for e in encoded_sentence]

            bpe_groups = []

            for bpe in bpes:
                if self.bpe_is_start(bpe):
                    group = []
                    bpe_groups.append(group)
                group.append(bpe)

            offset = 0

            encoded_sentence = []

            for bpe_group in bpe_groups:
                token = self._tokenizer.convert_tokens_to_string(bpe_group)
                vector_group = self.agglomeration_strategy(vectors[offset: offset + len(bpe_group)])
                encoded_sentence.append((token, vector_group))

                offset += len(bpe_group)

            assert vectors[offset:] == []

            agglomerated_encoding.append(encoded_sentence)

        return agglomerated_encoding

    def __encode(self, sentences: List[str], batch_size) -> List[List[Tuple[str, np.array]]]:

        feedable_sentences = []

        for sentence in sentences:
            sentence = self._tokenizer.tokenize(sentence)
            assert len(sentence) <= 512  # todo improve with blocking
            sentence = self._tokenizer.convert_tokens_to_ids(sentence)
            sentence = self._tokenizer.build_inputs_with_special_tokens(sentence)
            feedable_sentences.append(sentence)

        outputs = []

        for i in range(0, len(feedable_sentences), batch_size):

            batch = [torch.tensor(fs) for fs in feedable_sentences[i: i + batch_size]]
            batch = pad_sequence(batch, batch_first=True, padding_value=self._tokenizer.pad_token_id)

            attention_mask = torch.ones(batch.shape)
            attention_mask[batch == self._tokenizer.pad_token_id] = 0

            batch = batch.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                batch_out = self._model(batch, attention_mask=attention_mask)[0]

            for _batch, _batch_out in zip(batch, batch_out):

                output = []

                for e1, e2 in zip(_batch, _batch_out):

                    if e1 == self._tokenizer.pad_token_id:
                        continue

                    e1 = self._tokenizer.convert_ids_to_tokens(e1.item())
                    e2 = e2.cpu().numpy()

                    if e1 in self._tokenizer.all_special_tokens:
                        continue

                    output.append((e1, e2))

                outputs.append(output)

        return outputs
