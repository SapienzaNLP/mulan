#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import logging
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel

from encoding.base import Encoder

logger = logging.getLogger(__name__)


class BertEncoder(Encoder):

    def __init__(self, model_path: str, agglomeration_strategy: Callable[[List[np.array]], np.array], device: str = 'cpu'):

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.agglomeration_strategy = agglomeration_strategy

        self._model = BertModel.from_pretrained(model_path)
        self._model.to(device)
        self._model.eval()

        logger.debug('Bert model initialized')

        self.device = device

    def __call__(self, sentences: List[str], batch_size=4) -> List[List[Tuple[str, np.array]]]:

        encoding = self.__encode(sentences, batch_size=batch_size)
        agglomerated_encoding = []

        for sentence, encoded_sentence in zip(sentences, encoding):

            # todo a truly awesome workaround
            tokens = self._tokenizer.basic_tokenizer.tokenize(sentence, never_split=self._tokenizer.all_special_tokens)

            bpes = [e[0] for e in encoded_sentence]
            vectors = [e[1] for e in encoded_sentence]

            bpe_groups = []

            for bpe in bpes:
                if not bpe.startswith('##'):
                    group = [bpe]
                    bpe_groups.append(group)
                else:
                    group.append(bpe[2:])

            offset = 0

            encoded_sentence = []

            assert len(tokens) == len(bpe_groups)

            for token, bpe_group in zip(tokens, bpe_groups):

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

                    # todo special token support (especially CLS)
                    if e1 == '[CLS]' or e1 == '[SEP]':
                        continue

                    output.append((e1, e2))

                outputs.append(output)

        return outputs
