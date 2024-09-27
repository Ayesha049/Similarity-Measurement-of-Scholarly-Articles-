"""
Name: AYESHA SIDDIQUA and KHALID HASAN


How to run:
1. Place the source file "vector_space_model.py" in the same directory of "documents".
2. Open a terminal (or, command-prompt) in this directory and run the following command:
    python vector_space_model.py

"""
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from IR.base_IR import BaseIR
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


class VectorSpaceModel(BaseIR):
    def __init__(self, documents_path='./documents'):
        super().__init__()

        self.documents_path = documents_path
        self.terms_frequency = defaultdict(dict)

        self.vector_space = self._generate_vector_space()

    def _get_term_frequency_for_single_doc(self, doc_id, file_path) -> dict[str, dict]:
        """
        Generate term frequency with a unique doc_id for each term
        :param doc_id: a unique Document ID
        :param file_path: the path of the document file
        """
        text = self._read_document(file_path)
        terms = self.preprocess(text)

        _terms_frequency = defaultdict(dict)
        for _term in set(terms):
            _terms_frequency[_term][doc_id] = terms.count(_term)

        return _terms_frequency

    def _get_term_frequency_for_multiple_docs(self) -> dict[str, dict]:
        """
        Generate term frequency for multiple documents using multi-processes.
        :return: Term frequency of all documents
        """
        files = os.listdir(self.documents_path)

        # Using 4 workers to distribute the tasks, for each document file there is a unique file id
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(
                    self._get_term_frequency_for_single_doc,
                    file.split('.')[0],
                    f"{self.documents_path}/{file}")
                for file in files
            ]

            # Waiting for all tasks to complete
            results = [future.result() for future in as_completed(futures)]

        # Merge results received from the multi-processes
        merged_results = defaultdict(dict)
        for result in results:
            for key1, value1 in result.items():
                for key2, value2 in value1.items():
                    merged_results[key1][key2] = value2

        return merged_results

    def _generate_vector_space(self) -> pd.DataFrame:
        """
        Get term frequency for multiple docs in python dictionary and
        convert it into a vector space matrix using pandas dataframe
        :return: A pandas dataframe with term-frequency/vector-space matrix
        """
        try:
            with open("terms_frequency.json", "r") as fp:
                self.terms_frequency = json.load(fp)
        except:
            self.terms_frequency = self._get_term_frequency_for_multiple_docs()
            with open("terms_frequency.json", "w") as fp:
                json.dump(self.terms_frequency, fp)

        return pd.DataFrame.from_dict(self.terms_frequency, orient='index').fillna(0)

    def get_unique_words_count(self) -> int:
        """
        Get the count of unique words in all documents
        :return: Unique words count
        """
        return len(self.vector_space.index)

    def get_top_k_frequent_terms(self, _top_k):
        """
        Get top k most frequent terms in the collection by:
            1. Sum the appearance of terms in each document
            2. Sort the term occurrences in reverse order
            3. Select the top k most frequent terms
        :param _top_k: The given number of most frequent terms
        :return: The list of expected terms
        """
        collection_frequency: pd.DataFrame = self.vector_space.sum(axis=1).sort_values(ascending=False)

        return list(collection_frequency.index[:_top_k])

    @staticmethod
    def _get_cosine_similarity_measures(df: pd.DataFrame):
        """
        Get cosine similarity of a given dataframe matrix by:
            1. Normalize each element of each column using Euclidean distance
            2. For each column/doc, compute the dot product or cosine similarity with all other columns/docs
            3. Prioritize cosine similarities in descending order of each pair of columns/docs
        :param df: The given matrix
        :return: Cosine similarity measures for each pair of columns/docs in decreasing order
        """
        # Normalize each element of each column using Euclidean distance
        normalized_df = df.apply(
            lambda column: column / math.sqrt(sum(column ** 2))
        )

        similarity_measures = list()
        docs = normalized_df.columns

        # For each column/doc, compute the dot product or cosine similarity with all other columns/docs
        for _i in range(len(docs)):
            for _j in range(_i + 1, len(docs)):
                cos_similarity = normalized_df[docs[_i]].dot(normalized_df[docs[_j]])
                similarity_measures.append((round(cos_similarity, 2), docs[_i], docs[_j]))

        # Prioritize cosine similarities in descending order of each pair of columns/docs
        similarity_measures.sort(key=lambda obj: obj[0], reverse=True)

        return similarity_measures

    def get_top_k_closest_document_pairs_using_tf(self, _top_k):
        """Get the top k-closest document pairs using term frequency score"""
        similarity_measures = self._get_cosine_similarity_measures(self.vector_space)

        return similarity_measures[:_top_k]

    def get_idf(self):
        """
        Get the idf score by:
            1. Calculate document frequency: the number of documents in the corpus a term belongs to
            2. idf = Log_10(number of docs / document frequency)
        """
        document_frequency = self.vector_space.astype(bool).sum(axis=1)

        idf = np.log10(len(self.vector_space.columns) / document_frequency)

        return idf

    def get_top_k_closest_document_pairs_using_tf_idf(self, _top_k):
        """Get the top k-closest document pairs using tf-idf scheme"""
        idf = self.get_idf()

        # tf-idf = tf * idf
        tf_idf = self.vector_space.mul(idf, axis="index")

        similarity_measures = self._get_cosine_similarity_measures(tf_idf)

        return similarity_measures[:_top_k]

    def get_top_k_closest_document_pairs_using_wf_idf(self, _top_k):
        """Get the top k-closest document pairs using sublinear tf scaling (i.e., wf-idf scheme)"""
        # if tf>0, wf = 1 + log_10(tf) otherwise, wf = 0
        wf = self.vector_space.map(
            lambda value: (1 + np.log10(value)) if value > 0 else 0
        )

        idf = self.get_idf()

        # wf-idf = wf * idf
        wf_idf = wf.mul(idf, axis="index")

        similarity_measures = self._get_cosine_similarity_measures(wf_idf)

        return similarity_measures[:_top_k]

    def get_top_k_closest_documents(self, _top_k):
        # Using 3 workers to distribute the tasks
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.get_top_k_closest_document_pairs_using_tf, _top_k),
                executor.submit(self.get_top_k_closest_document_pairs_using_tf_idf, _top_k),
                executor.submit(self.get_top_k_closest_document_pairs_using_wf_idf, _top_k)
            ]

        # Wait until all tasks are completed
        return [future.result() for future in as_completed(futures)]

    def get_clustered_documents(self, using="euclidian", n_clusters=6) -> defaultdict[list]:
        """
        Get clustered documents using KMeans approach.
        :param using: The distance measure for grouping the documents -> euclidian (default) or cosine_similarity
        :param n_clusters: Number of clusters
        :return: A dict containing labels and documents -> {cluster_label: [documents]}
        """
        idf = vsmodel.get_idf()

        # tf-idf = tf * idf
        tf_idf = self.vector_space.mul(idf, axis="index")
        mat = tf_idf.T.values

        if using == "cosine_similarity":
            # Using sklearn's cosine_similarity function
            mat = cosine_similarity(mat)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(mat)

        labels = kmeans.predict(mat) if using == "cosine_similarity" else kmeans.labels_

        docs = tf_idf.columns.tolist()
        clusters = defaultdict(list)
        for index in range(len(labels)):
            clusters[labels[index]].append(docs[index])

        return clusters


if __name__ == '__main__':
    print("Generating Vector Space .....")

    # Set a timer to get the run time
    tic = time.perf_counter()
    vsmodel = VectorSpaceModel()
    toc = time.perf_counter()
    print(f"Generated the Vector Space in {toc - tic:0.4f} seconds")

    vsmodel.meta_data()

    print(f"The number of unique words is: {vsmodel.get_unique_words_count()}")

    top_k = 20
    print(f"The top {top_k} most frequent words are:")
    for i, term in enumerate(vsmodel.get_top_k_frequent_terms(top_k)):
        print(f"\t{i + 1}. {term}")

    print("========================================================")
    print("Generating K closest document pairs using tf, tf_idf, and wf_idf .....")

    # Set a timer to get the run time
    tic = time.perf_counter()
    using_tf, using_tf_idf, using_sublinear_tf_scaling = vsmodel.get_top_k_closest_documents(top_k)
    toc = time.perf_counter()
    print(f"Generated K closest document pairs using tf, tf_idf, and wf_idf in {toc - tic:0.4f} seconds")

    print(f"The top k closest document pairs are:")

    print("1. Using tf")
    for pair in using_tf:
        print(f"\t{pair[1]}, {pair[2]} with similarity of {pair[0]}")

    print("2. Using tf_idf")
    for pair in using_tf_idf:
        print(f"\t{pair[1]}, {pair[2]} with similarity of {pair[0]}")

    print("3. Using wf_idf")
    for pair in using_sublinear_tf_scaling:
        print(f"\t{pair[1]}, {pair[2]} with similarity of {pair[0]}")

    for i in range(10):
        pair1, pair2, pair3 = using_tf[i], using_tf_idf[i], using_sublinear_tf_scaling[i]
        print(
            f"{pair1[1]}, {pair1[2]} & {pair1[0]} & {pair2[1]}, {pair2[2]} & {pair2[0]} & {pair3[1]}, {pair3[2]} & {pair3[0]} \\\\ \hline")

    print("Create cluster using Euclidian distance measure")
    groups = vsmodel.get_clustered_documents()
    for k, v in groups.items():
        print(f"{k} & {', '.join(v)} \\\\")

    print("Create cluster using cosine similarity measure")
    groups = vsmodel.get_clustered_documents(using="cosine_similarity")
    for k, v in groups.items():
        print(f"{k} & {', '.join(v)} \\\\")
