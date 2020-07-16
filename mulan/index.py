#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from typing import List, Tuple, Any

import faiss
import numpy as np

from utils.commons import index_collection


class CosineFaissIndex:

    def __init__(self, elements: List[Any], x_vectors: np.array, use_gpu: bool = False):
        self.faiss_index = self.__load_faiss_index(x_vectors, use_gpu)
        self.elements_index = index_collection(elements)

    def get_near_elements(self, y_vectors: np.array, n_neighbors: int) -> List[Tuple[List[Any], List[float]]]:

        neighbors_s = []
        cosines_s, indices_s = self.faiss_index.search(y_vectors, n_neighbors)

        for cosines, indices in zip(cosines_s, indices_s):
            neighbors = [self.elements_index[s_idx] for s_idx in indices if s_idx >= 0]
            neighbors_scores = cosines
            neighbors_s.append(
                (neighbors, neighbors_scores)
            )

        return neighbors_s

    @staticmethod
    def __load_faiss_index(vectors: List[np.array], use_gpu: bool):
        vectors_dim = len(vectors[0])
        vector_stack = np.stack(vectors)
        index = faiss.IndexFlatIP(vectors_dim)
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(vector_stack)
        return index
