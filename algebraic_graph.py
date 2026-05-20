import numpy as np
import networkx as nx
from scipy.linalg import eigvals, eigvalsh
from scipy.sparse.linalg import eigs
from numpy.polynomial import Polynomial
import itertools
import config

def build_correlation_graph(returns_df, threshold=0.5, use_weights=True):
    """
    Build undirected weighted graph from ETF return correlations.
    """
    corr = returns_df.corr().abs()
    n = corr.shape[0]
    G = nx.Graph()
    tickers = returns_df.columns.tolist()
    G.add_nodes_from(tickers)
    for i in range(n):
        for j in range(i+1, n):
            w = corr.iloc[i, j]
            if w > threshold:
                if use_weights:
                    G.add_edge(tickers[i], tickers[j], weight=w)
                else:
                    G.add_edge(tickers[i], tickers[j])
    return G

def weisfeiler_lehman_coloring(G, iterations=10):
    """
    Return a colouring of nodes (list of ints) approximating automorphism orbits.
    """
    colors = {node: 1 for node in G.nodes}
    for _ in range(iterations):
        new_colors = {}
        for node in G.nodes:
            neighbor_colors = sorted([colors[nbr] for nbr in G.neighbors(node)])
            new_colors[node] = hash((colors[node], tuple(neighbor_colors)))
        # Normalise to integers
        unique_colors = {v: i for i, v in enumerate(sorted(set(new_colors.values())))}
        colors = {node: unique_colors[new_colors[node]] for node in G.nodes}
    return colors

def orbit_sizes(colors):
    """
    Return a dict {node: orbit_size} where orbit_size is the number of nodes with same colour.
    """
    color_counts = {}
    for col in colors.values():
        color_counts[col] = color_counts.get(col, 0) + 1
    orbit_sizes = {node: color_counts[colors[node]] for node in colors}
    return orbit_sizes

def adjacency_matrix(G):
    """Return adjacency matrix (weighted if available)."""
    A = nx.adjacency_matrix(G, weight='weight' if config.USE_WEIGHTS else None).toarray()
    return A

def characteristic_polynomial(A):
    """Return coefficients of characteristic polynomial det(λI - A)."""
    # Use numpy's polynomial from eigenvalues
    eigvals_array = eigvals(A)
    # The characteristic polynomial is ∏ (λ - λ_i)
    # Convert to polynomial coefficients (monic)
    coef = np.poly(eigvals_array)  # returns coefficients from highest degree to constant
    # Make it a Polynomial object
    return Polynomial(coef)

def spectral_gap(A):
    """Return λ1 - λ2 (largest minus second largest eigenvalue)."""
    ev = eigvalsh(A)
    ev = np.sort(ev)[::-1]
    return ev[0] - ev[1] if len(ev) > 1 else ev[0]

def is_ramanujan(G):
    """
    For a regular graph: test if all non-trivial eigenvalues |λ| ≤ 2√(k-1).
    For irregular, we use a relaxed condition: max non-trivial eigenvalue ≤ 2√(avg_degree).
    """
    A = adjacency_matrix(G)
    degs = [d for _, d in G.degree()]
    k = np.mean(degs)
    # Compute eigenvalues (largest few)
    ev = eigvals(A)
    ev = np.sort(np.abs(ev))[::-1]
    if len(ev) > 1:
        max_non_triv = ev[1] if ev[1] > ev[-1] else ev[-1]
    else:
        max_non_triv = 0
    bound = 2 * np.sqrt(k - 1) if k > 1 else 2
    return max_non_triv <= bound, max_non_triv, bound

def ihara_zeta_determinant(G, u):
    """
    Compute Ihara zeta function at point u using formula:
    Z_G(u)^{-1} = (1-u^2)^{χ(G)} det(I - u A + u^2 (D - I))
    where χ(G) = m - n (number of edges minus number of nodes).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    A = adjacency_matrix(G)
    D = np.diag([d for _, d in G.degree()])
    M = np.eye(n) - u * A + u**2 * (D - np.eye(n))
    det = np.linalg.det(M)
    chi = m - n
    Z_inv = (1 - u**2)**chi * det
    return 1.0 / Z_inv if Z_inv != 0 else np.inf

def compute_algebraic_invariants(G):
    """
    Compute all invariants:
        - orbit_sizes dict
        - characteristic_polynomial coeffs
        - spectral_gap
        - ramanujan test (bool, max_non_triv, bound)
        - Ihara zeta at a test point (e.g., u=0.1)
    """
    A = adjacency_matrix(G)
    colors = weisfeiler_lehman_coloring(G)
    orbits = orbit_sizes(colors)
    poly = characteristic_polynomial(A)
    gap = spectral_gap(A)
    ramanujan, max_non_triv, bound = is_ramanujan(G)
    # Evaluate Ihara zeta at a small u (0.1) as a proxy for its behaviour
    try:
        zeta = ihara_zeta_determinant(G, u=0.1)
    except:
        zeta = np.nan
    return {
        "orbit_sizes": orbits,
        "char_poly_coeffs": poly.coef.tolist(),
        "spectral_gap": float(gap),
        "is_ramanujan": ramanujan,
        "max_non_trivial_eigenvalue": float(max_non_triv),
        "ramanujan_bound": float(bound),
        "ihara_zeta_at_0.1": float(zeta) if not np.isnan(zeta) else None
    }

def score_etfs(G, invariants):
    """
    Score each ETF: combination of orbit_size (inverse, so smaller orbit = higher score)
    and eigenvector centrality.
    """
    eigen_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000) if config.USE_WEIGHTS else nx.eigenvector_centrality(G, max_iter=1000)
    orbit_sizes = invariants["orbit_sizes"]
    scores = {}
    for node in G.nodes:
        orbit_score = 1.0 / (orbit_sizes[node] + 1e-8)
        cent_score = eigen_cent[node]
        # Combine: higher is better for both (small orbit, high centrality)
        scores[node] = orbit_score * cent_score
    return scores
