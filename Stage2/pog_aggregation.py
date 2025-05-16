import numpy as np
from sklearn.cluster import KMeans
import time
import random
from collections import defaultdict

def generate_pog_example(num_clusters: int, max_group_size: int) -> str:
    """
    生成一个随机的POG字符串示例。

    Parameters:
    num_clusters (int): 总的聚类数量。
    max_group_size (int): 每个分组中最多包含的聚类数量。

    Returns:
    str: 生成的POG字符串，例如 '3=4=6>1=2=7>0>5=8=9>10'。
    """
    clusters = list(range(num_clusters))
    random.shuffle(clusters)  # 打乱顺序

    groups = []
    while clusters:
        group_size = random.randint(1, max_group_size)
        group = clusters[:group_size]
        groups.append(group)
        clusters = clusters[group_size:]

    pog_str = '>'.join(['='.join(map(str, sorted(group))) for group in groups])
    return pog_str

def calculate_rankings(pog_str):
    """
    解析POG字符串并计算排名和调整排名。

    Parameters:
    pog_str (str): 原始POG字符串，例如 '3=4=6>1=2=7>0>5=8=9>10'。

    Returns:
    tuple: rank_vec 和 adj_rank_vec，分别是排名向量和调整排名向量。
    """
    clusters = pog_str.split('>')
    rank = {}
    adj_rank = {}

    rank_count = 1
    for idx, group in enumerate(clusters):
        group_clusters = group.split('=')
        for cluster in group_clusters:
            rank[int(cluster)] = rank_count
        rank_count += 1

    rank_vec = [rank[cluster] for cluster in sorted(rank.keys())]
    reverse_rank = defaultdict(list)
    for cluster, r in rank.items():
        reverse_rank[r].append(cluster)

    for r in sorted(reverse_rank.keys()):
        for cluster in reverse_rank[r]:
            adj_rank[cluster] = rank[cluster] + 1 / 2 * (len(reverse_rank[r]) - 1)
    adj_rank_vec = [adj_rank[cluster] for cluster in sorted(adj_rank.keys())]
    return rank_vec, adj_rank_vec

def generate_pog_matrix(num_examples: int, num_clusters: int, max_group_size: int):
    """
    生成多个POG字符串并计算其排名矩阵。

    Parameters:
    num_examples (int): 要生成的POG字符串数量。
    num_clusters (int): 每个POG字符串的聚类数量。
    max_group_size (int): 每个分组中最多包含的聚类数量。

    Returns:
    np.ndarray: 计算出的调整排名矩阵。
    """
    pog_strings = [generate_pog_example(num_clusters, max_group_size) for _ in range(num_examples)]
    adj_ranks = []

    for pog_str in pog_strings:
        _, adj_rank_vec = calculate_rankings(pog_str)
        adj_ranks.append(adj_rank_vec)

    return np.array(adj_ranks)

def slow_pog_ensemble(pog_ranks: np.ndarray, sigma: float, epsilon: float, 
                       k: int, eta: float) -> np.ndarray:
    """
    较慢的POG聚合实现，包含显式循环和冗余计算。
    """
    start_time = time.time()
    M, N = pog_ranks.shape
    R_star = np.mean(pog_ranks, axis=0)

    while True:
        alphas = np.zeros(M)
        for m in range(M):
            dist = np.linalg.norm(pog_ranks[m] - R_star)
            alphas[m] = 1 - np.exp(-dist**2 / sigma**2)

        w_m = alphas / np.sum(alphas)

        R_star_new = np.zeros(N)
        for n in range(N):
            for m in range(M):
                R_star_new[n] += w_m[m] * pog_ranks[m, n]

        err = np.linalg.norm(R_star_new - R_star)
        if err < epsilon:
            break

        R_star = R_star_new

    print(f"Slow POG ensemble time: {time.time() - start_time:.2f} seconds")
    return R_star

def fast_pog_ensemble(pog_ranks: np.ndarray, sigma: float, epsilon: float, 
                       k: int, eta: float) -> np.ndarray:
    """
    快速的POG聚合实现。
    """
    start_time = time.time()
    M, N = pog_ranks.shape
    R_star = np.mean(pog_ranks, axis=0)

    while True:
        dists = np.linalg.norm(pog_ranks - R_star, axis=1)
        alphas = 1 - np.exp(-dists**2 / sigma**2)
        w_m = alphas / np.sum(alphas)
        R_star_new = np.dot(w_m, pog_ranks)
        err = np.linalg.norm(R_star_new - R_star)
        if err < epsilon:
            break
        R_star = R_star_new

    print(f"Fast POG ensemble time: {time.time() - start_time:.2f} seconds")
    return R_star

if __name__ == "__main__":
    num_examples = 100000  # POG矩阵的行数
    num_clusters = 22  # POG矩阵的列数
    max_group_size = 5  # 每个分组的最大大小

    pog_matrix = generate_pog_matrix(num_examples, num_clusters, max_group_size)

    sigma = 10.0
    epsilon = 1e-5
    k = 5
    eta = 0.1

    print("Running slow POG ensemble...")
    slow_result = slow_pog_ensemble(pog_matrix, sigma, epsilon, k, eta)

    print("Running fast POG ensemble...")
    fast_result = fast_pog_ensemble(pog_matrix, sigma, epsilon, k, eta)
