#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse

import numpy as np
from tqdm import tqdm

from corpora import Corpus
from index import CosineFaissIndex
from utils.file import yield_batches_from_vector_folder, count_lines_in_file


def fecth_vectors(ids, batches_folder):

    result_ids = []
    result_vectors = np.zeros((len(ids), 768), dtype=np.float32)

    tqdm_bar = tqdm(total=len(ids), desc=f'Reading vectors from {batches_folder}')

    for batch_ids, batch_vectors in yield_batches_from_vector_folder(batches_folder, 128):

        if len(result_ids) == len(ids):
            break

        for batch_id, batch_vector in zip(batch_ids, batch_vectors):
            if batch_id in ids:
                result_vectors[len(result_ids)] = batch_vector / np.linalg.norm(batch_vector)
                result_ids.append(batch_id)
                tqdm_bar.update(1)

    tqdm_bar.close()

    return result_ids, result_vectors[: len(result_ids)]


def main(input_coordinates_path, output_coordinates_path, source_batches_folder, target_batches_folder, k=4, top_n=100):

    # read candidates

    forward_candidates = {}
    considered_target_ids = set()

    with open(input_coordinates_path) as f:

        for line in tqdm(f, total=count_lines_in_file(input_coordinates_path), desc='Reading coordinates'):

            parts = line.strip().split('\t')

            if len(parts) < 3:
                continue

            source_id = parts[0]
            source_target_ids = list(filter(lambda x: x != '', parts[1].split(' ')))

            if len(source_target_ids) == 0:
                continue

            forward_candidates[source_id] = [target_id.split('#')[0] for target_id in source_target_ids][: top_n]

            for target_id in forward_candidates[source_id]:
                considered_target_ids.add(target_id)

    # fetch vectors

    source_ids, source_vectors = fecth_vectors(set(forward_candidates.keys()), source_batches_folder)
    source_id2index = {source_id: index for index, source_id in enumerate(source_ids)}
    target_ids, target_vectors = fecth_vectors(set(considered_target_ids), target_batches_folder)
    target_id2index = {target_id: index for index, target_id in enumerate(target_ids)}

    # compute backward candidates

    ## approximate backward candidates

    approximated_backward_candidates = {}

    with open(input_coordinates_path) as f:

        for line in tqdm(f, total=count_lines_in_file(input_coordinates_path), desc='Approximating backward candidates'):

            parts = line.strip().split('\t')

            if len(parts) < 3:
                continue

            source_id = parts[0]
            source_target_ids = list(filter(lambda x: x != '', parts[1].split(' ')))

            if len(source_target_ids) == 0:
                continue

            for target_id in source_target_ids:

                target_id, target_score = target_id.split('#')
                target_score = float(target_score)

                if target_id not in considered_target_ids:
                    continue

                if target_id not in approximated_backward_candidates:
                    approximated_backward_candidates[target_id] = []

                if len(approximated_backward_candidates[target_id]) >= top_n and target_score < approximated_backward_candidates[target_id][-1][1]:
                    continue

                approximated_backward_candidates[target_id].append((source_id2index[source_id], target_score))

                if len(approximated_backward_candidates[target_id]) > 1.1 * top_n:

                    approximated_backward_candidates[target_id] = sorted(
                        approximated_backward_candidates[target_id],
                        key=lambda x: x[1],
                        reverse=True
                    )[: top_n]

    ## actual backward candidates

    backward_candidates = {}

    for target_id, target_source_ids in tqdm(approximated_backward_candidates.items(), desc='Computing backward candidates'):
        target_source_ids = sorted(target_source_ids, key=lambda x: x[1], reverse=True)[: top_n]
        backward_candidates[target_id] = set([e[0] for e in target_source_ids])

    del approximated_backward_candidates

    # intersect candidates

    # todo decomment if no longer debug: currently want 0.0 on backward deleted candidates

    # candidates = {}
    #
    # for source_id, source_target_ids in forward_candidates.items():
    #
    #     filtered_target_ids = []
    #
    #     for target_id in source_target_ids:
    #         if source_id in backward_candidates[target_id]:
    #             filtered_target_ids.append(target_id)
    #
    #     candidates[source_id] = filtered_target_ids

    candidates = forward_candidates

    # compute margins

    def compute_margin(x_ids, x_vectors, y_vectors, k, batch_size=250_000):

        margin = np.zeros((len(y_vectors), k))

        for i in range(0, len(y_vectors), batch_size):

            batch_y_vectors = y_vectors[i: i + batch_size]
            batch_margin = []

            for j in range(0, len(x_vectors), batch_size):

                batch_x_ids = x_ids[j: j + batch_size]
                batch_x_vectors = x_vectors[j: j + batch_size]

                batch_index = CosineFaissIndex(batch_x_ids, batch_x_vectors, use_gpu=True)
                batch_search_result = batch_index.get_near_elements(batch_y_vectors, min(k, len(x_vectors)))
                batch_margin.append([similarities for (_, similarities) in batch_search_result])

            batch_margin = np.concatenate(batch_margin, axis=1)
            batch_margin = np.sort(batch_margin, axis=1)[:, -k:][:, ::-1]

            margin[i: i + len(batch_margin)] = batch_margin

        return np.array(margin).mean(axis=1)

    target2source_mean = compute_margin(source_ids, source_vectors, target_vectors, k)

    source2target_mean = np.zeros((len(source_ids)))

    with open(input_coordinates_path) as f:

        for line in f:

            parts = line.strip().split('\t')

            if len(parts) < 3:
                continue

            source_id = parts[0]

            if source_id not in candidates:
                continue

            source_idx = source_id2index[source_id]
            target_ids = parts[2].split(' ')

            source2target_mean[source_idx] = np.array([float(source_candidate.split('#')[1]) for source_candidate in target_ids]).mean()

    # score candidates

    with open(output_coordinates_path, 'w') as f:

        for source_id, target_ids in tqdm(candidates.items(), desc='Scoring candidates'):

            source_index = source_id2index[source_id]
            source_vector = source_vectors[source_index]
            scored_target_ids = []

            for target_id in target_ids:

                target_index = target_id2index[target_id]
                target_vector = target_vectors[target_index]

                n = source_vector.T @ target_vector
                d = (source2target_mean[source_index] + target2source_mean[target_index]) / 2

                # scored_target_ids.append((target_id, n / d))
                scored_target_ids.append((target_id, n / d if source_id2index[source_id] in backward_candidates[target_id] else 0.0))

            scored_target_ids = sorted(scored_target_ids, key=lambda x: x[1], reverse=True)

            f.write(f'{source_id}\t{" ".join(["#".join(map(str, scored_target_id)) for scored_target_id in scored_target_ids])}\n')


def parse_args():

    parser = argparse.ArgumentParser(description='Aim or compute firing priorities')

    parser.add_argument("--source-enum", required=True, type=Corpus.argparse, help='Source corpus', choices=list(Corpus))
    parser.add_argument("--target-enum", required=True, type=Corpus.argparse, help='Target corpus', choices=list(Corpus))

    parser.add_argument("--coordinates-folder", required=True, type=str, help='Folder where coordinates have been dumped and normalized coordinates will be saved')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    main(
        f'{args.coordinates_folder}/coordinates.tsv',
        f'{args.coordinates_folder}/coordinates.normalized.tsv',
        args.source_enum.value.vectorization_folder,
        args.target_enum.value.vectorization_folder,
        k=4,
        top_n=100
    )
