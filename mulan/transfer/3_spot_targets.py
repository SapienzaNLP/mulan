#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse
import functools
import json
import pickle
import statistics
from collections import namedtuple
from typing import cast, Optional

import numpy as np
import os

from multiprocessing import Pool
from tqdm import tqdm

from corpora import Corpus
from index import CosineFaissIndex
from transfer.transfer_utils import load_corpus, to_text_span
from utils.commons import transpose, flat_list
from utils.data_structures import PriorityQueue
from utils.file import yield_batches_from_vector_folder


def get_valid_lemmas(annotation, annotation2target_lemmas):

    valid_lemmas = []

    for _annotation in annotation.split(','):
        if _annotation in annotation2target_lemmas:
            valid_lemmas += annotation2target_lemmas[_annotation]

    return valid_lemmas


SearchPostProcessingResult = namedtuple(
    'SearchPostProcessingResult',
    [
        'source_id',
        'below_min_threshold_attempted_transfers',
        'invalid_lemma_invalid_transfers',
        'invalid_lemma_invalid_transfers_log',
        'kb_cut_top_k_candidates',
        'raw_top_k_candidates'
    ]
)


def process_source_candidate_list(
        source_candidate_list,
        source_transfer_text_span_cache,
        target_transfer_text_span_cache,
        annotation2target_lemmas,
        raw_top_k,
        min_threshold) -> Optional[SearchPostProcessingResult]:

    source_id, (target_candidates, target_scores) = source_candidate_list

    if len(target_candidates) == 0:
        return None

    below_min_threshold_attempted_transfers = 0
    invalid_lemma_invalid_transfers = 0
    invalid_lemma_invalid_transfers_log = None
    kb_cut_top_k_candidates = []
    raw_top_k_candidates = []

    source_transfer = source_transfer_text_span_cache[source_id]
    top_invalid_lemma_reported = False

    for target_candidate, target_score in zip(target_candidates, target_scores):

        if target_score < min_threshold:
            below_min_threshold_attempted_transfers += 1
            continue

        if len(raw_top_k_candidates) < raw_top_k:
            raw_top_k_candidates.append((target_candidate, target_score))

        valid_lemmas = get_valid_lemmas(source_transfer.annotation, annotation2target_lemmas)
        target_transfer = target_transfer_text_span_cache[target_candidate]

        if target_transfer.lemma.replace('_', ' ').lower() not in valid_lemmas and ' '.join(target_transfer.tokens).lower() not in valid_lemmas:

            invalid_lemma_invalid_transfers += 1

            if not top_invalid_lemma_reported:
                invalid_lemma_invalid_transfers_log = f'{target_score:.5f}\t{source_transfer.annotation}\t{",".join(valid_lemmas)}\t{source_transfer.lemma}\t{target_transfer.lemma}\t{source_transfer.full_id}\t{target_transfer.full_id}\n'
                top_invalid_lemma_reported = True

            continue

        kb_cut_top_k_candidates.append((target_candidate, target_score))

    return SearchPostProcessingResult(
        source_id=source_id,
        below_min_threshold_attempted_transfers=below_min_threshold_attempted_transfers,
        invalid_lemma_invalid_transfers=invalid_lemma_invalid_transfers,
        invalid_lemma_invalid_transfers_log=invalid_lemma_invalid_transfers_log,
        kb_cut_top_k_candidates=kb_cut_top_k_candidates,
        raw_top_k_candidates=raw_top_k_candidates
    )


def main(source_corpus, target_corpus, annotation2target_lemmas, output_path):

    assert not os.path.exists(output_path), f'Folder {output_path} already exists'
    os.mkdir(output_path)

    # load dbs

    source_db = source_corpus.get_db()
    target_db = target_corpus.get_db()

    invalid_lemma_file = open(f'{output_path}/invalid_lemma.txt', 'w')
    wrong_lemma_file = open(f'{output_path}/wrong_lemma.txt', 'w')

    # params

    source_batch_size = 100_000
    target_batch_size = 100_000
    # target_batch_size = 250_000

    # n_target_sentences_to_consider = 5_000_000
    with open(f'{target_corpus.vectorization_folder}/stats.json') as f:
        n_target_sentences_to_consider = json.load(f)['done_sentences']

    min_threshold = 0.25
    local_k_nn = 100

    kb_cut_top_k = 1000
    raw_top_k = 4

    source2candidates = {}

    # loading source ids

    source_batches = []
    source_transfer_text_span_cache = {}
    considered_sources = 0

    iterator = tqdm(
        enumerate(yield_batches_from_vector_folder(f'{source_corpus.vectorization_folder}', source_batch_size)),
        desc='Loading source ids'
    )

    for source_batch_index, (batch_source_ids, batch_source_vectors) in iterator:

        # preprocessing stage

        actual_source_ids = []
        mask = np.full(batch_source_vectors.shape[0], False)

        for i, id in enumerate(batch_source_ids):

            ## populate text span cache
            source_transfer_text_span_cache[id] = to_text_span(id, source_db)

            ## discard ids with 0 lemmas in the target language
            
            annotation = source_transfer_text_span_cache[id].annotation
            valid_lemmas = get_valid_lemmas(annotation, annotation2target_lemmas)

            if valid_lemmas == []:
                invalid_lemma_file.write(f'{annotation}\t{source_transfer_text_span_cache[id].lemma}\t{source_transfer_text_span_cache[id].full_id}\n')
                continue

            actual_source_ids.append(id)
            mask[i] = True

        batch_source_ids = actual_source_ids
        batch_source_vectors = batch_source_vectors[mask]

        assert len(batch_source_ids) == len(batch_source_vectors)

        # normalize vectors
        batch_source_vectors = batch_source_vectors / np.linalg.norm(batch_source_vectors, axis=1).reshape(-1, 1)

        considered_sources += len(batch_source_ids)
        source_batches.append((batch_source_ids, batch_source_vectors))

        iterator.set_postfix(considered_sources=considered_sources)

    iterator.close()

    # re-applying source batch size

    source_ids, source_vectors = transpose(source_batches)
    source_ids = flat_list(source_ids)
    source_vectors = np.concatenate(source_vectors, axis=0)

    source_batches = []

    for i in range(0, len(source_ids), source_batch_size):
        source_batches.append((source_ids[i: i + source_batch_size], source_vectors[i: i + source_batch_size]))

    # aim

    iterator = enumerate(yield_batches_from_vector_folder(f'{target_corpus.vectorization_folder}', target_batch_size))
    tqdm_bar = tqdm(total=n_target_sentences_to_consider, desc='Aiming')
    below_min_threshold_attempted_transfers = 0
    invalid_lemma_invalid_transfers = 0.0
    transfer_count = 0

    n_target_sentences = 0

    already_seen_target_sentences = set()
    last_seen_sentence_id = None
    just_added_override = False

    for target_batch_index, (batch_target_ids, batch_target_vectors) in iterator:

        if n_target_sentences_to_consider != -1 and n_target_sentences > n_target_sentences_to_consider:
            break

        # preprocess target batch

        batch_n_target_sentences = 0
        target_transfer_text_span_cache = {}

        actual_target_ids = []
        mask = np.full(batch_target_vectors.shape[0], False)

        for i, (target_id, target_vector) in enumerate(zip(batch_target_ids, batch_target_vectors)):

            ## populate text span cache

            target_transfer_text_span_cache[target_id] = to_text_span(target_id, target_db)

            ## filter out sentences spans and avoid adding already seen sentences

            target_sentence = target_transfer_text_span_cache[target_id].sentence
            is_new_sentence = target_transfer_text_span_cache[target_id].full_id != last_seen_sentence_id

            if is_new_sentence:
                just_added_override = False
                last_seen_sentence_id = target_transfer_text_span_cache[target_id].full_id

            if not just_added_override and target_sentence in already_seen_target_sentences:
                continue

            already_seen_target_sentences.add(target_sentence)
            just_added_override = True

            if is_new_sentence:
                batch_n_target_sentences += 1

            actual_target_ids.append(target_id)
            mask[i] = True
                
        n_target_sentences += batch_n_target_sentences

        batch_target_ids = actual_target_ids
        batch_target_vectors = batch_target_vectors[mask]

        # normalize vectors

        batch_target_vectors = batch_target_vectors / np.linalg.norm(batch_target_vectors, axis=1).reshape(-1, 1)

        # populate space

        # todo make use_gpu a parameter
        faiss_index = CosineFaissIndex(batch_target_ids, batch_target_vectors, use_gpu=True)

        for source_batch_index, (source_ids, source_vectors) in enumerate(source_batches):

            target_search = faiss_index.get_near_elements(source_vectors, n_neighbors=local_k_nn)

            iterator = zip(source_ids, target_search)

            f = functools.partial(
                process_source_candidate_list,
                source_transfer_text_span_cache=source_transfer_text_span_cache,
                target_transfer_text_span_cache=target_transfer_text_span_cache,
                annotation2target_lemmas=annotation2target_lemmas,
                raw_top_k=raw_top_k,
                min_threshold=min_threshold
            )

            if len(source2candidates) == 0:
                buckets_mean = 0
            else:
                buckets_mean = statistics.mean([len(v[0]) for _, v in source2candidates.items()])

            for result_index, result in enumerate(map(f, iterator)):

                if result is None:
                    continue

                result = cast(SearchPostProcessingResult, result)

                source_id = result.source_id

                below_min_threshold_attempted_transfers += result.below_min_threshold_attempted_transfers
                invalid_lemma_invalid_transfers += result.invalid_lemma_invalid_transfers

                if result.invalid_lemma_invalid_transfers_log:
                    wrong_lemma_file.write(result.invalid_lemma_invalid_transfers_log)

                if source_id not in source2candidates:
                    source2candidates[source_id] = (PriorityQueue(kb_cut_top_k), PriorityQueue(raw_top_k))

                for target_candidate, target_score in result.raw_top_k_candidates[: raw_top_k]:
                    source2candidates[source_id][1].add(target_candidate, target_score)

                for target_candidate, target_score in result.kb_cut_top_k_candidates:
                    was_new_entry_added = source2candidates[source_id][0].add(target_candidate, target_score) is None
                    if was_new_entry_added:
                        transfer_count += 1

                if result_index % (source_batch_size // 20) == 0:

                    tqdm_bar.set_postfix(
                        source_batch=source_batch_index,
                        target_batch=target_batch_index,
                        below_min_threshold=below_min_threshold_attempted_transfers,
                        wrong_lemma=invalid_lemma_invalid_transfers,
                        buckets_mean=buckets_mean,
                        transfers=transfer_count
                    )

            tqdm_bar.set_postfix(
                source_batch=source_batch_index,
                target_batch=target_batch_index,
                below_min_threshold=below_min_threshold_attempted_transfers,
                wrong_lemma=invalid_lemma_invalid_transfers,
                buckets_mean=buckets_mean,
                transfers=transfer_count
            )

        tqdm_bar.update(n=batch_n_target_sentences)

    invalid_lemma_file.close()
    wrong_lemma_file.close()

    with open(f'{output_path}/stats.json', 'w') as stats_file:
        json.dump(
            dict(
                considered_sources=considered_sources,
                n_target_sentences=n_target_sentences,
                below_min_threshold=below_min_threshold_attempted_transfers,
                wrong_lemma=invalid_lemma_invalid_transfers,
                buckets_mean=statistics.mean([len(v[0]) for _, v in source2candidates.items()]),
                transfers=transfer_count
            ),
            stats_file,
            indent=4
        )

    tqdm_bar.close()

    with open(f'{output_path}/coordinates.tsv', 'w') as f:
        for source, candidates_lists in source2candidates.items():
            candidates = sorted(candidates_lists[0], key=lambda x: x[1], reverse=True)
            candidates = ' '.join([f'{e[0]}#{e[1]}' for e in candidates])
            marginalized_candidates = sorted(candidates_lists[1], key=lambda x: x[1], reverse=True)
            marginalized_candidates = ' '.join([f'{e[0]}#{e[1]}' for e in marginalized_candidates])
            parts = '\t'.join([source, candidates, marginalized_candidates])
            f.write(f'{parts}\n')


def parse_args():

    parser = argparse.ArgumentParser(description='Spot targets and dump possible firing coordinates')

    parser.add_argument("--source-enum", required=True, type=Corpus.argparse, help='Source corpus', choices=list(Corpus))
    parser.add_argument("--target-enum", required=True, type=Corpus.argparse, help='Target corpus', choices=list(Corpus))

    parser.add_argument("--coordinates-folder", required=True, type=str, help='Folder where produced coordinates are to be saved')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    print(args.source_enum)

    with open(f'cache/synset2lemmas/{args.target_enum.value.language.lower()}.pickle', 'rb') as f:
        vocabulary = pickle.load(f)

    main(
        args.source_enum.value,
        args.target_enum.value,
        vocabulary,
        args.coordinates_folder
    )
