import numpy as np
import pandas as pd
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")


def solve_2x2_nash(payoff_matrix):
    """
    Solve Nash equilibrium for a 2x2 game.
    Returns mixed strategy probabilities for both players.
    """
    # Payoff matrix is from P1's perspective
    a, b = payoff_matrix[0]
    c, d = payoff_matrix[1]

    # Check for pure strategy Nash equilibria
    # P1 best responses
    p1_best_vs_col0 = 0 if a >= c else 1
    p1_best_vs_col1 = 0 if b >= d else 1

    # P2 best responses (minimize P1's payoff)
    p2_best_vs_row0 = 0 if a <= b else 1
    p2_best_vs_row1 = 0 if c <= d else 1

    # Check for pure Nash
    if p1_best_vs_col0 == 0 and p2_best_vs_row0 == 0:
        return (1.0, 0.0), (1.0, 0.0)
    if p1_best_vs_col0 == 1 and p2_best_vs_row1 == 0:
        return (0.0, 1.0), (1.0, 0.0)
    if p1_best_vs_col1 == 0 and p2_best_vs_row0 == 1:
        return (1.0, 0.0), (0.0, 1.0)
    if p1_best_vs_col1 == 1 and p2_best_vs_row1 == 1:
        return (0.0, 1.0), (0.0, 1.0)

    # Mixed strategy Nash equilibrium
    # For P1: make P2 indifferent between columns
    # a*p + c*(1-p) = b*p + d*(1-p)
    # p(a - c - b + d) = d - c
    denom = a - b - c + d
    if abs(denom) < 1e-10:  # Degenerate case
        return (0.5, 0.5), (0.5, 0.5)

    p1_prob = (d - c) / denom
    p1_prob = max(0, min(1, p1_prob))  # Ensure in [0,1]
    p1_strategy = (p1_prob, 1 - p1_prob)

    # For P2: make P1 indifferent between rows
    # a*q + b*(1-q) = c*q + d*(1-q)
    # q(a - b - c + d) = d - b
    p2_prob = (d - b) / denom
    p2_prob = max(0, min(1, p2_prob))  # Ensure in [0,1]
    p2_strategy = (p2_prob, 1 - p2_prob)

    return p1_strategy, p2_strategy


def calculate_expected_value(p1_strategy, p2_strategy, payoff_matrix):
    """Calculate expected value for P1 given strategies."""
    ev = 0
    for i in range(2):
        for j in range(2):
            ev += p1_strategy[i] * p2_strategy[j] * payoff_matrix[i, j]
    return ev


def solve_tournament_nash(winrates_df):
    """
    Solve the Nash equilibrium for 2-deck tournament format.
    """
    decks = winrates_df.columns.tolist()
    n_decks = len(decks)

    # Generate all possible 2-deck lineups
    lineups = list(combinations(range(n_decks), 2))
    lineup_names = [f"{decks[i]}+{decks[j]}" for i, j in lineups]
    n_lineups = len(lineups)

    print(f"Found {n_decks} decks: {decks}")
    print(f"Generated {n_lineups} possible lineups:\n")
    for i, name in enumerate(lineup_names):
        print(f"  {i+1}. {name}")

    # Create the outer game payoff matrix
    outer_payoff = np.zeros((n_lineups, n_lineups))

    print("\n" + "=" * 80)
    print("SOLVING INNER GAMES (2x2 Matrices)")
    print("=" * 80)

    # For each pair of lineups, solve the inner game
    for i, lineup1 in enumerate(lineups):
        for j, lineup2 in enumerate(lineups):
            # Create 2x2 payoff matrix for this matchup
            inner_payoff = np.zeros((2, 2))

            # Lineup1 decks vs Lineup2 decks
            deck1_a, deck1_b = lineup1
            deck2_a, deck2_b = lineup2

            # Fill in the winrates
            # P1 chooses row (deck1_a or deck1_b)
            # P2 chooses column (deck2_a or deck2_b)
            # Payoff is P1's winrate when P1's deck faces P2's deck
            inner_payoff[0, 0] = winrates_df.iloc[
                deck1_a, deck2_a
            ]  # P1 plays deck1_a, P2 plays deck2_a
            inner_payoff[0, 1] = winrates_df.iloc[
                deck1_a, deck2_b
            ]  # P1 plays deck1_a, P2 plays deck2_b
            inner_payoff[1, 0] = winrates_df.iloc[
                deck1_b, deck2_a
            ]  # P1 plays deck1_b, P2 plays deck2_a
            inner_payoff[1, 1] = winrates_df.iloc[
                deck1_b, deck2_b
            ]  # P1 plays deck1_b, P2 plays deck2_b

            # Solve the inner Nash equilibrium
            p1_strat, p2_strat = solve_2x2_nash(inner_payoff)

            # Calculate expected value for the outer game
            ev = calculate_expected_value(p1_strat, p2_strat, inner_payoff)
            outer_payoff[i, j] = ev

            # Print details for each inner game
            if i <= j:  # Only print unique matchups
                print(f"\nMatchup: {lineup_names[i]} vs {lineup_names[j]}")
                print(f"Payoff Matrix (P1 perspective):")
                print(f"              {decks[deck2_a]:^10} {decks[deck2_b]:^10}")
                print(
                    f"  {decks[deck1_a]:10} {inner_payoff[0,0]:^10.2f} {inner_payoff[0,1]:^10.2f}"
                )
                print(
                    f"  {decks[deck1_b]:10} {inner_payoff[1,0]:^10.2f} {inner_payoff[1,1]:^10.2f}"
                )
                print(f"Nash Equilibrium:")
                print(
                    f"  P1 plays: {decks[deck1_a]} {p1_strat[0]:.1%}, {decks[deck1_b]} {p1_strat[1]:.1%}"
                )
                print(
                    f"  P2 plays: {decks[deck2_a]} {p2_strat[0]:.1%}, {decks[deck2_b]} {p2_strat[1]:.1%}"
                )
                print(f"  Expected value: {ev:.3f}")

    print("\n" + "=" * 80)
    print("OUTER GAME PAYOFF MATRIX")
    print("=" * 80)

    # Display outer payoff matrix
    outer_df = pd.DataFrame(outer_payoff, index=lineup_names, columns=lineup_names)
    print("\n", outer_df.round(3))

    print("\n" + "=" * 80)
    print("SOLVING OUTER NASH EQUILIBRIUM")
    print("=" * 80)

    # Solve outer Nash equilibrium using linear programming approach
    # This is more robust for larger games than the 2x2 formula
    from scipy.optimize import linprog

    # Solve for P1's strategy (row player)
    # Maximize v subject to: outer_payoff @ p2 >= v for all p2
    # Convert to minimization problem
    n = n_lineups
    c = np.zeros(n + 1)
    c[-1] = -1  # Maximize v

    # Constraints: -outer_payoff.T @ p1 + v <= 0
    A_ub = np.column_stack([-outer_payoff.T, np.ones(n)])
    b_ub = np.zeros(n)

    # Equality constraint: sum(p1) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1
    b_eq = np.array([1])

    # Bounds: 0 <= p1[i] <= 1, v unbounded
    bounds = [(0, 1) for _ in range(n)] + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if result.success:
        p1_strategy = result.x[:n]
        game_value = result.x[n]

        # Solve for P2's strategy similarly
        c2 = np.zeros(n + 1)
        c2[-1] = 1  # Minimize v

        A_ub2 = np.column_stack([outer_payoff, -np.ones(n)])
        b_ub2 = np.zeros(n)

        result2 = linprog(
            c2,
            A_ub=A_ub2,
            b_ub=b_ub2,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if result2.success:
            p2_strategy = result2.x[:n]

            print("\nOptimal Lineup Distribution (Nash Equilibrium):")
            print("-" * 50)

            # Sort by probability for better display
            p1_sorted = sorted(
                zip(lineup_names, p1_strategy), key=lambda x: x[1], reverse=True
            )

            print("\nPlayer 1 Optimal Strategy:")
            for lineup, prob in p1_sorted:
                if prob > 0.001:  # Only show lineups with >0.1% probability
                    print(f"  {lineup:20} {prob:>6.1%}")

            print(f"\nGame Value (Expected winrate for P1): {game_value:.3f}")

            # Verify equilibrium
            print("\nVerification:")
            print(f"  Sum of probabilities: {sum(p1_strategy):.6f}")
            print(f"  Min expected value for P1: {min(outer_payoff @ p2_strategy):.3f}")
            print(
                f"  Max expected value for P2: {1 - max(p1_strategy @ outer_payoff):.3f}"
            )

            return p1_strategy, lineup_names, game_value

    print("Failed to find Nash equilibrium")
    return None, None, None


def main():
    # Read the winrates CSV
    try:
        df = pd.read_csv("winrates.csv", index_col=0)
        print("Successfully loaded winrates.csv")
        print("\nWinrate Matrix (row beats column):")
        print(df)
        print()

        # Solve the tournament Nash equilibrium
        strategy, lineups, value = solve_tournament_nash(df)

        if strategy is not None:
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"\nIn a Nash equilibrium for this 2-deck best-of-1 format:")
            print(f"- Expected winrate at equilibrium: {value:.1%}")
            print(
                f"- Number of viable lineups: {sum(1 for p in strategy if p > 0.001)}"
            )

            # With a symmetric game, the value should be exactly 0.5
            if abs(value - 0.5) < 0.001:
                print(
                    "- The game is perfectly balanced (as expected with symmetric winrates)"
                )
            else:
                print(
                    f"- Warning: Game value is {value:.3f}, but should be 0.500 for symmetric winrates"
                )

    except FileNotFoundError:
        print("Error: winrates.csv not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
