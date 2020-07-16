#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import heapq
from typing import Any, List, Callable, Tuple


class PriorityQueue:

    def __init__(self, max_size: int, reverse=False):
        self.max_size = max_size
        assert self.max_size > 0
        self.reverse = reverse
        self.backend_data_structure = []
        heapq.heapify(self.backend_data_structure)
        self.length = 0

    @property
    def min_value(self):
        *priority_scores, elem = self.backend_data_structure[0]
        return (elem, *priority_scores)

    def add(self, elem: Any, *priority_scores):

        if self.reverse:
            priority_scores = [-1 * priority_score for priority_score in priority_scores]

        if len(self.backend_data_structure) < self.max_size:
            heapq.heappush(self.backend_data_structure, (*priority_scores, elem))
            self.length += 1
            return
        else:

            if priority_scores < self.backend_data_structure[0][: -1]:
                return (*priority_scores, elem)

            return heapq.heapreplace(self.backend_data_structure, (*priority_scores, elem))

    def __iter__(self):
        for (*priority_scores, elem) in sorted(self.backend_data_structure):
            if self.reverse:
                priority_scores = [-1 * priority_score for priority_score in priority_scores]
            yield (elem, *priority_scores)

    def __len__(self) -> int:
        return self.length
