#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse
import os
import pickle
from collections import Counter
from itertools import chain
from typing import Tuple, List, Iterable, Dict, Set

from tqdm import tqdm

from config import WSD_DATASET_FOLDER
from corpora import Corpus
from db.level_db import LevelDB
from transfer.entities import TextSpan
from transfer.transfer_utils import to_text_span, load_corpus
from utils.data_structures import PriorityQueue
from utils.file import count_lines_in_file, load_coordinates
from utils.nlp.kb import to_bn_id
from utils.nlp.wsd import RaganatoBuilder, read_from_raganato


def compute_annotation2lemma_occurrences(paths) -> Dict[str, Dict[str, int]]:

    annotation2lemma_occurrences = {}

    for xml_path, key_path in paths:
        for _, _, sentence in read_from_raganato(xml_path, key_path):
            for instance in sentence:
                if instance.labels is not None:
                    for annotation in instance.labels:
                        annotation = to_bn_id(annotation)
                        if annotation not in annotation2lemma_occurrences:
                            annotation2lemma_occurrences[annotation] = Counter()
                        annotation2lemma_occurrences[annotation][instance.lemma] += 1

    return annotation2lemma_occurrences


def raw_sentence(text_span: TextSpan) -> str:
    sentence_annotated_tokens = text_span.sentence.split(' ')
    sentence_tokens = [x.split(' ')[0] for x in sentence_annotated_tokens]
    return ' '.join(sentence_tokens)


def main(
        language: str,
        name: str,
        annotation2lemma_occurrences: Dict[str, Dict[str, int]],
        coordinates_it: Iterable[Tuple[TextSpan, List[Tuple[TextSpan, float]]]],
        coordinates_len: int,
        output_folder: str,
        synset2lemmas: Dict[str, Set[str]],
        hard_threshold: float = 0.0,
        alpha: float = 1.0
):

    assert not os.path.exists(output_folder)
    os.mkdir(output_folder)

    # logging path

    logging_path_src_tgt = f'{output_folder}/src-tgt.log'
    logging_file_src_tgt = open(logging_path_src_tgt, 'w')

    logging_path_tgt_src = f'{output_folder}/tgt-src.log'
    logging_file_tgt_src = open(logging_path_tgt_src, 'w')

    # take top k for each annotation lemma

    annotation2lemma_transfers = {}

    iterator = tqdm(
        enumerate(coordinates_it),
        total=coordinates_len,
        desc='Grouping transfers by annotation'
    )

    n_evaluated_transfers = 0
    n_evaluated_valid_transfers = 0
    n_current_transfers = 0

    for i, (transfer_source, transfer_targets) in iterator:

        n_evaluated_transfers += len(transfer_targets)

        if transfer_source.start_sentence_index == -1:
            continue

        for annotation in transfer_source.annotation.split(','):

            annotation = to_bn_id(annotation)

            if annotation not in annotation2lemma_transfers:
                annotation2lemma_transfers[annotation] = {}

            if transfer_source.lemma not in annotation2lemma_transfers[annotation]:
                annotation2lemma_transfers[annotation][transfer_source.lemma] = PriorityQueue(int(alpha * annotation2lemma_occurrences[annotation][transfer_source.lemma]))

            logging_file_src_tgt.write(f'{transfer_source.annotation}\t{transfer_source.full_id}\t{transfer_source.lemma}\t{transfer_source.pos}\t{raw_sentence(transfer_source)}\n')

            for transfer_target, transfer_score in transfer_targets:

                if transfer_target.start_sentence_index != -1:

                    n_evaluated_valid_transfers += 1

                    if annotation2lemma_transfers[annotation][transfer_source.lemma].add((transfer_source, transfer_target), transfer_score, f'{transfer_source.full_id}.{transfer_source.start_sentence_index}.{transfer_source.end_sentence_index}', f'{transfer_target.full_id}.{transfer_target.start_sentence_index}.{transfer_target.end_sentence_index}') is None:
                        n_current_transfers += 1

                    logging_file_src_tgt.write(f'\t{transfer_score}\t{transfer_target.full_id}\t{transfer_target.lemma}\t{transfer_target.pos}\t{raw_sentence(transfer_target)}\n')

        logging_file_src_tgt.write('\n')

        iterator.set_postfix(
            n_evaluated_transfers=n_evaluated_transfers,
            n_evaluated_valid_transfers=n_evaluated_valid_transfers,
            n_current_transfers=n_current_transfers,
            n_current_mean=n_current_transfers / len(annotation2lemma_transfers)
        )

    # merge lemmas

    annotation2transfers = {}

    iterator = tqdm(
        annotation2lemma_transfers.items(),
        desc='Merging lemmas'
    )

    for annotation, lemma_transfers in iterator:

        if annotation not in annotation2transfers:
            annotation2transfers[annotation] = []

        for lemma, lemma_transfer in lemma_transfers.items():
            for (transfer_source, transfer_target), transfer_score, _, _ in lemma_transfer:
                annotation2transfers[annotation].append((transfer_source, transfer_target, transfer_score))

    # fire at coordinates

    target_id2transfers = {}

    n_transfers = 0
    iterator = tqdm(annotation2transfers.items())

    for annotation, transfers in iterator:

        synset = to_bn_id(annotation)

        if synset not in synset2lemmas:
            continue

        valid_lemmas = synset2lemmas[synset]

        for (transfer_source, transfer_target, transfer_score) in transfers:

            if transfer_score < hard_threshold:
                continue

            if transfer_target.lemma.replace('_', ' ').lower() not in valid_lemmas and ' '.join(transfer_target.tokens).lower() not in valid_lemmas:
                continue

            if transfer_target.full_id not in target_id2transfers:
                target_id2transfers[transfer_target.full_id] = []

            target_id2transfers[transfer_target.full_id].append((
                transfer_target,
                transfer_source,
                transfer_score
            ))
            n_transfers += 1

        iterator.set_postfix(n_transfers=n_transfers)

    # report

    raganato = RaganatoBuilder(language, name)

    iterator = tqdm(target_id2transfers.items())
    n_written_sentences = 0
    n_transfers = 0

    for target_id, transfers in iterator:

        if len(transfers) == 0:
            continue

        iterator.set_postfix(written_sentences=n_written_sentences, n_transfers=n_transfers)

        document_id, sentence_id = target_id.split('.')

        target_sentence = transfers[0][0].sentence.split('\t')
        target_tokens_it = iter(target_sentence)

        raganato.open_text_section(document_id)
        raganato.open_sentence_section(sentence_id)

        # transfers resolution

        resolved_transfers = []

        for (transfer_target, transfer_source, transfer_score) in sorted(transfers, key=lambda x: (len(x[0].tokens), x[2]), reverse=True):
            spanned_range = set(range(transfer_target.start_sentence_index, transfer_target.end_sentence_index))
            if all(len(spanned_range.intersection(range(resolved_transfer[0].start_sentence_index, resolved_transfer[0].end_sentence_index))) == 0 for resolved_transfer in resolved_transfers):
                resolved_transfers.append((transfer_target, transfer_source, transfer_score))

        for transfer_target, transfer_source, transfer_score in resolved_transfers:
            logging_file_tgt_src.write(f'\t{transfer_score}\t{transfer_source.annotation}\t{transfer_target.lemma}@@@{transfer_target.full_id} <- {transfer_source.lemma}@@@{transfer_source.full_id}\t{raw_sentence(transfer_target)}\t{raw_sentence(transfer_source)}\n')

        start_index2resolved_transfer = {}

        for i, (transfer_target, transfer_source, transfer_score) in enumerate(sorted(resolved_transfers, key=lambda x: x[0].start_sentence_index)):
            start_index2resolved_transfer[transfer_target.start_sentence_index] = (i, (transfer_target, transfer_source, transfer_score))

        i = 0

        while True:

            if i in start_index2resolved_transfer:

                idx, (transfer_target, transfer_source, transfer_score) = start_index2resolved_transfer[i]

                for _ in range(transfer_target.end_sentence_index - transfer_target.start_sentence_index):
                    i += 1
                    next(target_tokens_it)

                raganato.add_annotated_token(
                    '_'.join(transfer_target.tokens), transfer_target.lemma, transfer_target.pos,
                    str(idx), transfer_source.annotation.replace(',', ' ')
                )

                n_transfers += 1

            else:

                try:
                    target_token = next(target_tokens_it)
                    i += 1
                except StopIteration:
                    break

                target_token = target_token.split(' ')
                raganato.add_annotated_token(target_token[0], target_token[1], target_token[2])

        n_written_sentences += 1

    raganato.store(f'{output_folder}/transfer.data.xml', f'{output_folder}/transfer.gold.key.txt')
    logging_file_src_tgt.close()
    logging_file_tgt_src.close()


def parse_args():

    def coordinates_input(t):

        try:
            path, source_corpus, target_corpus = t.split(',')
        except:
            raise argparse.ArgumentTypeError('Coordinates input must be coordinates_path,source_corpus,target_corpus')

        try:
            source_corpus = Corpus.argparse(source_corpus)
        except:
            raise argparse.ArgumentTypeError(f'{source_corpus} is not a valid value for enum Corpus')

        try:
            target_corpus = Corpus.argparse(target_corpus)
        except:
            raise argparse.ArgumentTypeError(f'{target_corpus} is not a valid value for enum Corpus')

        return path, source_corpus, target_corpus

    parser = argparse.ArgumentParser(description='Fire at coordinates')

    parser.add_argument("--language", required=True, type=str, help='Language the mapping synset -> valid lemmas has to be built for')
    parser.add_argument("--name", required=True, type=str, help='Name of the generated dataset')

    parser.add_argument('--coordinates', required=True, type=coordinates_input, action='append', help='Triples (coordinates path, source enum, target enum)')
    parser.add_argument("--output-folder", required=True, type=str, help='Output folder')

    parser.add_argument("--hard-threshold", type=float, default=0.0, help='Proposed transfers below this threshold will be discarded')
    parser.add_argument("--alpha", type=float, default=1.0, help='Scales the number of transfers picked for each (source lemma, source annotation) pair: alpha * # (source lemma, source annotation) instances ')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    with open(f'cache/synset2lemmas/{args.language.lower()}.pickle', 'rb') as f:
        vocabulary = pickle.load(f)

    # todo put distribution as parameter

    main(
        language=args.language,
        name=args.name,
        annotation2lemma_occurrences=compute_annotation2lemma_occurrences([
            (WSD_DATASET_FOLDER / 'SemCor/semcor.data.xml', WSD_DATASET_FOLDER / 'SemCor/semcor.gold.key.txt'),
            (WSD_DATASET_FOLDER / 'WNGT/raganato/wngt.data.xml', WSD_DATASET_FOLDER / 'WNGT/raganato/wngt.gold.key.txt')
        ]),
        coordinates_it=chain(
            *[load_coordinates(_coordinates[0], _coordinates[1].value.get_db(), _coordinates[2].value.get_db(), column_idx=1) for _coordinates in args.coordinates]
        ),
        coordinates_len=sum([count_lines_in_file(_coordinates[0]) for _coordinates in args.coordinates]),
        output_folder=args.output_folder,
        synset2lemmas=vocabulary,
        hard_threshold=args.hard_threshold,
        alpha=args.alpha
    )
