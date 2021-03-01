#
# Created by maks5507 (me@maksimeremeev.com)
#

from typing import Callable, List, Dict
import faiss
import numpy as np


class FaissWrapper:

    metrics = {
        'ip': faiss.METRIC_INNER_PRODUCT,
        'l2': faiss.METRIC_L2,
        'l1': faiss.METRIC_L1,  # cpu only
        'linf': faiss.METRIC_Linf,  # cpu only
        'canberra': faiss.METRIC_Canberra,  # cpu only
        'bray_curtis': faiss.METRIC_BrayCurtis,  # cpu only
        'jensen_shannon': faiss.METRIC_JensenShannon,  # cpu only
    }

    def __init__(self, vec_dimension: int, transformation: Callable = None, metric: str = 'l2',
                 num_clusters: int = None, num_probe: int = None, num_bytes: int = None):
        self.quantizer = faiss.IndexFlat(vec_dimension, self.metrics[metric])
        if num_bytes is not None:
            self.index = faiss.IndexIVFPQ(self.quantizer, vec_dimension, num_clusters, num_bytes,
                                          self.metrics[metric])
        else:
            self.index = faiss.IndexIVFFlat(self.quantizer, vec_dimension, num_clusters, self.metrics[metric])
        self.index.nprobe = num_probe
        self.index.make_direct_map()
        self.transformation = transformation
        self.mapper = {}
        self.inverted_mapper = {}

    def train_index(self, vectors: Dict[object, List[float]]):
        if self.transformation is None:
            self.index.train(np.float32(vectors))
        else:
            vectors_transformed = []
            for vector in vectors.values():
                vectors_transformed += [self.transformation(vector)]
            self.index.train(np.float32(vectors_transformed))

    def add_vectors_to_index(self, vectors: Dict[object, List[float]]):
        for vector_id, vector in vectors.items():
            vector_transformed = vector.copy()
            if self.transformation is not None:
                vector_transformed = self.transformation(vector_transformed)
            current_slot = len(self.mapper)
            self.mapper[vector_id] = current_slot
            self.inverted_mapper[current_slot] = vector_id
            self.index.add(np.float32([vector_transformed]))

    def get_vector(self, vector_id: object):
        return self.index.reconstruct(self.mapper[vector_id])

    def search(self, request_vectors: List[List[float]], num_to_return: int):
        request_vectors_transformed = []
        for request_vector in request_vectors:
            request_vectors_transformed += [self.transformation(request_vector)]
        distances, indices = self.index.search(np.float32(request_vectors_transformed), num_to_return)
        to_return = []
        for i in range(len(request_vectors)):
            current_state = {}
            for distance, index in zip(distances[i], indices[i]):
                current_state[self.inverted_mapper[index]] = distance
            to_return += [current_state]
        return to_return


