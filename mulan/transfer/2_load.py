#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse
import json
import os
import pickle
import re
from typing import List, Tuple, Iterable, Set, Callable

import numpy as np
from tqdm import tqdm

from corpora import Corpus
from encoding.base import AlreadyTokenizedAligner
from encoding.transformers_encoders import TransformerEncoder
from transfer.entities import TextSpan
from transfer.transfer_utils import normalize_line
from utils.file import count_lines_in_file
from utils.nlp.commons import ngram


def load_sentences(sentences_folder: str, batch_size: int = 10_000, limit: int = -1, sentence_max_length: int = 100) -> Iterable[Tuple[List[str], List[List[TextSpan]]]]:

    n_lines_read = 0

    # shuffle (to avoid biasing a top k pick)

    for sentences_path in os.listdir(sentences_folder):

        sentences_path = f'{sentences_folder}/{sentences_path}'

        with open(sentences_path) as f:

            batch_sentences_indices = []
            batch_sentences = []

            for line in f:

                sentence = normalize_line(line)
                parts = sentence.split('\t')

                if len(parts) == 1:
                    continue

                sentence_index, *sentence_annotated_tokens = parts
                document_id, sentence_id = sentence_index.split('.')

                sentence_tokens = []

                for i, sentence_token in enumerate(sentence_annotated_tokens):

                    sentence_token = sentence_token.split(' ')

                    sentence_tokens.append(
                        TextSpan(
                            re.split(r'(?<=[^_])_(?=[^_])', sentence_token[0]),
                            sentence_token[2],
                            sentence_token[1],
                            i,
                            i + 1,
                            sentence_token[3] if len(sentence_token) == 4 else None,
                            sentence,
                            sentence_id,
                            document_id
                        )
                    )

                if sentence_max_length != -1 and len(sentence_tokens) > sentence_max_length:
                    continue

                batch_sentences_indices.append(sentence_index)
                batch_sentences.append(sentence_tokens)

                if len(batch_sentences) == batch_size:
                    yield batch_sentences_indices, batch_sentences
                    batch_sentences_indices = []
                    batch_sentences = []

                n_lines_read += 1

                if limit != -1 and n_lines_read == limit:
                    return

    if len(batch_sentences) > 0:
        yield batch_sentences_indices, batch_sentences


class VectorPointProducer:

    def produce(self, text_spans: List[List[TextSpan]], vectors: List[List[np.array]]):
        raise NotImplementedError


class SourceVectorPointProducer(VectorPointProducer):

    def produce(self, text_spans: List[List[TextSpan]], vectors: List[List[np.array]]) -> List[List[Tuple[TextSpan, np.array]]]:

        assert len(text_spans) == len(vectors)

        result = []

        for sentence_text_spans, sentence_vectors in zip(text_spans, vectors):

            _result = []
            result.append(_result)

            for sentence_text_span, sentence_vector in zip(sentence_text_spans, sentence_vectors):
                if sentence_text_span.annotation is not None and sentence_text_span.annotation != 'X':
                    _result.append((sentence_text_span, sentence_vector))

        return result


class TargetVectorPointProducer(VectorPointProducer):

    def __init__(self, vocabulary: Set[str], agglomeration_strategy: Callable[[List[np.array]], np.array], max_ngram_length: int = 3):
        self.vocabulary = vocabulary
        self.max_ngram_length = max_ngram_length
        self.agglomeration_strategy = agglomeration_strategy

    def produce(self, text_spans: List[List[TextSpan]], vectors: List[List[np.array]]):

        assert len(text_spans) == len(vectors)

        result = []

        for sentence_text_spans, sentence_vectors in zip(text_spans, vectors):

            _result = []
            result.append(_result)

            for n_gram_length in range(1, self.max_ngram_length + 1):

                n_grammed_sentence_text_spans = ngram(sentence_text_spans, n_gram_length)
                n_grammed_sentence_vectors = ngram(sentence_vectors, n_gram_length)

                for _n_grammed_sentence_text_spans, _n_grammed_sentence_vectors in zip(n_grammed_sentence_text_spans, n_grammed_sentence_vectors):

                    lemma_src_text_span = ' '.join(map(lambda x: x.lemma, _n_grammed_sentence_text_spans)).lower()

                    if lemma_src_text_span not in self.vocabulary and ' '.join([' '.join(e.tokens) for e in _n_grammed_sentence_text_spans]).lower() not in self.vocabulary:
                        continue

                    text_span = TextSpan(
                        [' '.join(e.tokens) for e in _n_grammed_sentence_text_spans],
                        None,  # todo
                        lemma_src_text_span,  # todo
                        _n_grammed_sentence_text_spans[0].start_sentence_index,
                        _n_grammed_sentence_text_spans[-1].start_sentence_index + 1,
                        None,
                        _n_grammed_sentence_text_spans[0].sentence,
                        _n_grammed_sentence_text_spans[0].sentence_id,
                        _n_grammed_sentence_text_spans[0].document_id
                    )

                    _result.append((text_span, self.agglomeration_strategy(_n_grammed_sentence_vectors)))

        return result


def main(encoder, sentences_folder, output_folder, is_annotated, language, vector_points_limit=5e7, sentences_batch_size=64, sentence_max_length=100):

    assert not os.path.exists(output_folder), f'Folder {output_folder} already exists'
    os.mkdir(output_folder)

    print(f'About to start processing language {language} on sentences {sentences_folder}')
    print(f'Saving to {output_folder}')

    # params

    sentences_limit = -1

    if sentences_limit == -1:
        n_sentences = 0
        for sentences_path in os.listdir(sentences_folder):
            n_sentences += count_lines_in_file(f'{sentences_folder}/{sentences_path}')
    else:
        n_sentences = sentences_limit

    # define producer

    if is_annotated:

        vector_point_producer = SourceVectorPointProducer()

    else:

        with open(f'cache/synset2lemmas/{language}.pickle', 'rb') as f:
            synset2lemmas = pickle.load(f)
            vocabulary = set([lemma for lemmas in synset2lemmas.values() for lemma in lemmas])

        vector_point_producer = TargetVectorPointProducer(
            vocabulary,
            agglomeration_strategy=lambda ngram_vectors: np.average(ngram_vectors, axis=0)
        )

    # save files

    index_file_path_format = output_folder + '/vectorization_index_{}.txt'
    vectors_file_path_format = output_folder + '/vectorization_{}'

    # script

    already_done_sentences = set()

    n_vector_points = 0
    crashed_batches = 0
    batch_index = 0

    iterator = load_sentences(sentences_folder, batch_size=sentences_batch_size, limit=sentences_limit, sentence_max_length=sentence_max_length)

    tqdm_bar = tqdm(total=int(vector_points_limit))

    for batch_sentences_indices, batch_sentences in iterator:

        if n_vector_points > vector_points_limit:
            break

        try:

            assert len(batch_sentences_indices) == len(batch_sentences), 'number of sentences different from sentences_indices'

            # deduplicate

            deduplicated_batch_sentences_indices = []
            deduplicated_batch_sentences = []

            str_tokenized_deduplicated_batch_sentences = []
            str_deduplicated_batch_sentences = []

            sentences_already_in_batch = set()

            for batch_sentence_index, batch_sentence in zip(batch_sentences_indices, batch_sentences):

                str_tokenized_batch_sentence = [' '.join(span.tokens) for span in batch_sentence]
                str_batch_sentence = ' '.join(str_tokenized_batch_sentence)

                if str_batch_sentence in already_done_sentences or str_batch_sentence in sentences_already_in_batch:
                    continue

                deduplicated_batch_sentences_indices.append(batch_sentence_index)
                deduplicated_batch_sentences.append(batch_sentence)
                str_tokenized_deduplicated_batch_sentences.append(str_tokenized_batch_sentence)
                str_deduplicated_batch_sentences.append(str_batch_sentence)

                sentences_already_in_batch.add(str_batch_sentence)

            if len(str_deduplicated_batch_sentences) == 0:
                continue

            # encode

            encoded_deduplicated_batch_sentences = encoder(str_deduplicated_batch_sentences, batch_size=sentences_batch_size, tokenized_sentences=str_tokenized_deduplicated_batch_sentences)
            encoded_deduplicated_batch_sentences = [[_e[1] for _e in e] for e in encoded_deduplicated_batch_sentences]

            # save files

            index_file = open(index_file_path_format.format(batch_index), 'w')

            # produce vector points

            vectors_stack = []
            vector_points = vector_point_producer.produce(deduplicated_batch_sentences, encoded_deduplicated_batch_sentences)

            for _sentence_index, sentence_vector_points in zip(deduplicated_batch_sentences_indices, vector_points):
                for sentence_text_span, sentence_vector in sentence_vector_points:
                    index_file.write('{}.{}.{}\t{}\n'.format(
                        _sentence_index,
                        sentence_text_span.start_sentence_index,
                        sentence_text_span.end_sentence_index,
                        '_'.join(sentence_text_span.tokens)
                    ))
                    vectors_stack.append(sentence_vector)

            # save

            index_file.close()

            np.save(vectors_file_path_format.format(batch_index), np.stack(vectors_stack))

            batch_index += 1
            batch_vector_points = sum(map(len, vector_points))
            n_vector_points += batch_vector_points

            for str_batch_sentence in str_deduplicated_batch_sentences:
                already_done_sentences.add(str_batch_sentence)

            tqdm_bar.set_postfix(done_sentences=len(already_done_sentences), batch_index=batch_index, crashed_batches=crashed_batches)
            tqdm_bar.update(batch_vector_points)

        except Exception as e:

            tqdm_bar.write(f'Skipping batch {batch_index} due to {e}')
            batch_index += 1
            crashed_batches += 1
            continue

    tqdm_bar.close()

    with open(f'{output_folder}/stats.json', 'w') as f:
        json.dump(
            dict(done_sentences=len(already_done_sentences), batch_index=batch_index),
            f,
            indent=4
        )


def parse_args():

    parser = argparse.ArgumentParser(description='Load')
    parser.add_argument("corpus_enum", type=Corpus.argparse, help='Corpus to vectorize', choices=list(Corpus))
    parser.add_argument("--sentences-batch-size", type=int, default=128, help='Vectorization batch size')
    parser.add_argument("--sentence-max-length", type=int, default=-1, help='Max sentence length; sentences longer than this threshold will be discarded. Use -1 to disable this behavior')
    parser.add_argument("--cuda-device", type=str, default='cpu', help='Device to use')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # todo make encoder a arg parameter

    main(
        encoder=AlreadyTokenizedAligner(
            encoder=TransformerEncoder(
                model_name_or_path='bert-base-multilingual-cased',
                device=args.cuda_device,
                bpe_is_start=lambda bpe: not bpe.startswith('##'),
                agglomeration_strategy=lambda bpe_vectors: np.average(bpe_vectors, axis=0)
            ),
            agglomeration_strategy=lambda token_vectors: np.average(token_vectors, axis=0)
        ),
        sentences_folder=args.corpus_enum.value.data_folder,
        output_folder=str(args.corpus_enum.value.vectorization_folder),
        is_annotated=args.corpus_enum.value.is_annotated,
        language=args.corpus_enum.value.language,
        sentences_batch_size=args.sentences_batch_size,
        sentence_max_length=args.sentence_max_length
    )
