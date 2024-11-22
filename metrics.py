from collections import defaultdict
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def novelty(recommendations: list[list[str]]) -> list[float]:
    """
        returns list of novelty score for each user based on list of each user recommendaions
        example:
        recs = [["a", "b"], ["a", "c"], ["b"], ["v", "y"], ["y"], ["z"]]
        print(novelty(recs))
    """
    counts = defaultdict(int)
    for user in recommendations:
        for recommendation in user:
            counts[recommendation] += 1

    N = len(recommendations)  # users count
    probs = {k: v / N for k, v in counts.items()}
    novelties = [sum(-math.log2(probs[recommendation])
                     for recommendation in user) / len(user) for user in recommendations]
    return novelties  # may also return mean


def ILS(recommendations: list[list[str]], embeddings: dict[str: np.array]) -> list[float]:
    """
        computes Intra List Similarity for each user using cosine similarity on embeddings of recommendations; smaller values means bigger diversity
        example:
        recs = [["a", "b"], ["a", "c"]]
        embeddings = {"a":  np.array([1, 2]), "b": np.array([2, 3]), "c": np.array([5, -1])}
        print(ILS(recs, embeddings))
    """
    similarities = []
    for user in recommendations:
        feature_vectors = np.array([embeddings[item] for item in user])
        similarity_matrix = cosine_similarity(feature_vectors)
        similarity = np.mean(similarity_matrix)
        similarities.append(similarity)

    return similarities


def personalization(recommendations: list[list[str]]) -> float:
    def cosine_similarity(A, B):
        dot_product = A @ B
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)

    unique_items = sorted(
        set(item for user in recommendations for item in user))
    N = len(recommendations)
    num_items = len(unique_items)
    mx = np.zeros((N, num_items))

    for i, user in enumerate(recommendations):
        for item in user:
            mx[i, unique_items.index(item)] = 1

    similarity_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            similarity_matrix[i][j] = cosine_similarity(
                mx[i], mx[j]) if i != j else 1

    personalization_score = 1 - np.mean(similarity_matrix)
    return personalization_score
