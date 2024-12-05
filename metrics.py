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
    """
        returns personalization score
        example:
        recs = [["a", "b"], ["a", "c"], ["b"], ["v", "y"], ["y"], ["z"]]
        print(personalization(recs))
    """
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


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    calculates precision at k

    Parameters:
    recommended: ranked list of model predictions
    relevant: list of relevant items (ground truth)
    k: number of top recommendations to consider
    """
    assert k > 0, "k should be a positive integer"

    k = min(k, len(recommended))
    recommended = recommended[:k].copy()

    relevant_count = len(set(relevant) & set(recommended))
    return relevant_count / k


def average_precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    calculates average precision at k

    Parameters:
    recommended: list of recommended items
    relevant: list of relevant items
    k: number of top recommendations to consider
    """
    assert k > 0, "k should be a positive integer"

    k = min(k, len(recommended))
    recommended = recommended[:k].copy()
    relevant_count = 0
    precision_sum = 0.0

    for i, item in enumerate(recommended, 1):
        if item in relevant:  # could use precision_at_k here, but this implementation is faster
            relevant_count += 1
            precision_sum += relevant_count / i

    return precision_sum / k


def mean_ap_at_k(all_recommended: list[list[str]], all_relevant: list[list[str]], k: int) -> float:
    """
    calculates mean average precision at k across multiple queries

    Parameters:
    all_recommended: list containing recommended items for each query.
    all_relevant: list containing relevant items for each query.
    k: number of top recommendations to consider.

    Returns:
    float: Mean Average Precision at K value.
    """
    ap_scores = [average_precision_at_k(recommended, relevant, k)
                 for recommended, relevant in zip(all_recommended, all_relevant)]
    return np.mean(ap_scores)


def cg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    calculates cumulative gain at k

    Parameters:
    recommended: ranked list of model predictions
    relevant: list of relevant items (ground truth)
    k: number of top recommendations to consider
    """
    assert k > 0, "k should be a positive integer"

    k = min(k, len(recommended))
    recommended = recommended[:k].copy()
    cumulative_gain = sum(1 for item in recommended if item in relevant)
    return cumulative_gain


def dcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    calculates discounted cumulative gain at k

    Parameters:
    recommended: ranked list of model predictions
    relevant: list of relevant items (ground truth)
    k: number of top recommendations to consider
    """
    assert k > 0, "k should be a positive integer"

    k = min(k, len(recommended))
    recommended = recommended[:k].copy()
    dcg = sum(1 / np.log2(i + 1)
              for i, item in enumerate(recommended, 1) if item in relevant)

    return dcg


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    calculates normalized discounted cumulative gain at k

    Parameters:
    recommended: ranked list of model predictions
    relevant: list of relevant items (ground truth)
    k: number of top recommendations to consider
    """
    assert k > 0, "k should be a positive integer"

    k = min(k, len(recommended))
    recommended = recommended[:k].copy()

    idcg = sum(1 / np.log2(i + 1) for i in range(k))
    dcg = dcg_at_k(recommended, relevant, k)

    return dcg / idcg


def mean_ndcg_at_k(all_recommended: list[list[str]], all_relevant: list[list[str]], k: int) -> float:
    """
    calculates mean average precision at k across multiple queries

    Parameters:
    all_recommended: list containing recommended items for each query.
    all_relevant: list containing relevant items for each query.
    k: number of top recommendations to consider.

    Returns:
    float: Mean Average Precision at K value.
    """
    ndgc_scores = [ndcg_at_k(recommended, relevant, k)
                   for recommended, relevant in zip(all_recommended, all_relevant)]
    return np.mean(ndgc_scores)
