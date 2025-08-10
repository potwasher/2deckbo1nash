# conquest_bo1_rps_style.py
import pandas as pd
import numpy as np
import itertools
from scipy.optimize import linprog


# ----- Helpers -----
def load_P_csv(path):
    df = pd.read_csv(path, index_col=0)
    decks = list(df.index)
    P = df.values.astype(float)
    # safety: fill diagonal with 0.5 if not already
    np.fill_diagonal(P, 0.5)
    return P, decks


def solve_2x2_zero_sum(a, b, c, d, eps=1e-12):
    """
    Solve 2x2 zero-sum game with payoff matrix for row player:
       [[a, b],
        [c, d]]
    Returns (x, y, value) where:
      x = prob row plays first row,
      y = prob col plays first col,
      value = game value (expected payoff for row).
    If matrix degenerate or numeric issues, falls back to pure best responses.
    """
    # Denominator for row mix formula: (a - b - c + d)
    denom = a - b - c + d
    if abs(denom) > eps:
        x = (d - b) / denom
        # column mix
        denom_col = a - c - b + d
        if abs(denom_col) > eps:
            y = (d - c) / (a - c - b + d)
        else:
            # symmetric fallback
            y = 0.5
    else:
        # degenerate: fallback to uniform/randomization
        x = 0.5
        y = 0.5

    # clamp to [0,1]
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)

    # compute value: expected payoff when row uses x and col uses y
    # value = x*y*a + x*(1-y)*b + (1-x)*y*c + (1-x)*(1-y)*d
    value = x * y * a + x * (1 - y) * b + (1 - x) * y * c + (1 - x) * (1 - y) * d
    return x, y, value


# ----- Build matchup payoff matrices -----
def build_matrices(P, decks):
    n = P.shape[0]
    lineups = list(itertools.combinations(range(n), 2))
    m = len(lineups)

    M_open = np.zeros(
        (m, m)
    )  # payoff matrix using 2x2 Nash inside lineup (open decklist)
    M_closed = np.zeros(
        (m, m)
    )  # payoff matrix using uniform 50/50 inside lineup (closed decklist)

    for a_idx, (i, j) in enumerate(lineups):
        for b_idx, (k, l) in enumerate(lineups):
            # closed: uniform 50/50 inside each lineup -> average of 4 matchups
            M_closed[a_idx, b_idx] = (P[i, k] + P[i, l] + P[j, k] + P[j, l]) / 4.0

            # open: solve 2x2 zero-sum inside the lineup
            a = P[i, k]
            b = P[i, l]
            c = P[j, k]
            d = P[j, l]
            x_row, y_col, val = solve_2x2_zero_sum(a, b, c, d)

            print(f"[{decks[i]}, {decks[j]}] : [{decks[k]}, {decks[l]}]")
            print(f"deck 1: {a} {b}\ndeck 2: {c} {d}")
            # x = chance of row selecting r1, y = chance of col selecting c1
            print(f"deck 1 pick% {abs(x_row):.2f} : {abs(y_col):.2f}")
            print(f"p1 win: {val:.2f}")
            print()

            # val is expected single-game winrate for the row-player under the internal mixed strategies
            M_open[a_idx, b_idx] = val

    return lineups, M_open, M_closed


# ----- Solve outer zero-sum (lineup selection) using LP -----
def solve_outer_nash(M):
    m = M.shape[0]
    # variables: x (m lineup probs) and v (value)
    c = np.zeros(m + 1)
    c[-1] = -1.0  # maximize v -> minimize -v

    # constraints: for each opponent column j: sum_i x_i * M[i,j] >= v  ->  -sum_i M[i,j]*x_i + v <= 0
    A_ub = np.hstack([-M.T, np.ones((m, 1))])
    b_ub = np.zeros(m)

    # equality: sum x_i = 1
    A_eq = np.hstack([np.ones((1, m)), [[0.0]]])
    b_eq = [1.0]

    bounds = [(0.0, 1.0)] * m + [(None, None)]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError("LP failed: " + str(res.message))
    x = res.x[:m]
    v = res.x[-1]
    return x, v


# ----- Utility: convert lineup dist -> deck shares (fraction of players bringing each deck) -----
def lineup_to_deck_shares(lineups, lineup_probs, n):
    # each lineup (a,b) contributes lineup_prob to each deck a and b. Sum and divide by 2 to get fraction of players.
    deck_counts = np.zeros(n)
    for prob, (a, b) in zip(lineup_probs, lineups):
        deck_counts[a] += prob
        deck_counts[b] += prob
    # deck_counts sum to 2 (since each player contributes 2 deck appearances); fraction of players bringing deck = deck_counts / 2
    deck_share = deck_counts / 2.0
    return deck_share


# ----- Main runner -----
def run(csv_path):
    P, decks = load_P_csv(csv_path)
    n = P.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 decks.")
    lineups, M_open, M_closed = build_matrices(P, decks)

    # solve outer Nash for both formats
    strat_open, val_open = solve_outer_nash(M_open)
    strat_closed, val_closed = solve_outer_nash(M_closed)

    deck_share_open = lineup_to_deck_shares(lineups, strat_open, n)
    deck_share_closed = lineup_to_deck_shares(lineups, strat_closed, n)

    # present results
    print("=== Open Decklist (internal 2x2 Nash picks) ===")
    print(f"Equilibrium expected winrate (first-player): {val_open:.4f}\n")
    print("Lineup distribution (probabilities):")
    for (a, b), p in zip(lineups, strat_open):
        if p > 1e-6:
            print(f"  {decks[a]} + {decks[b]} : {p:.4f}")
    print("\nPer-deck fraction of players bringing each deck (sum to 1 across decks):")
    for name, share in zip(decks, deck_share_open):
        print(f"  {name:15s} : {share:.4f}")

    print("\n\n=== Closed Decklist (internal uniform 50/50) ===")
    print(f"Equilibrium expected winrate (first-player): {val_closed:.4f}\n")
    print("Lineup distribution (probabilities):")
    for (a, b), p in zip(lineups, strat_closed):
        if p > 1e-6:
            print(f"  {decks[a]} + {decks[b]} : {p:.4f}")
    print("\nPer-deck fraction of players bringing each deck (sum to 1 across decks):")
    for name, share in zip(decks, deck_share_closed):
        print(f"  {name:15s} : {share:.4f}")

    return {
        "decks": decks,
        "lineups": lineups,
        "M_open": M_open,
        "M_closed": M_closed,
        "strat_open": strat_open,
        "val_open": val_open,
        "strat_closed": strat_closed,
        "val_closed": val_closed,
        "deck_share_open": deck_share_open,
        "deck_share_closed": deck_share_closed,
    }


# If run as script:
if __name__ == "__main__":
    # change filename as needed
    results = run("test.csv")
