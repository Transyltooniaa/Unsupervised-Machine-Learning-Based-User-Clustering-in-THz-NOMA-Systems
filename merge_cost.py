import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram # Import dendrogram for potential debugging
# from scipy.stats import circmean # Assuming this is defined elsewhere if needed
# from sklearn.cluster import AgglomerativeClustering # Not strictly needed for plotting costs
# from clustering import THzNomaSystem # Assuming this is your base class if needed elsewhere

# Dummy THzNomaSystem class for standalone execution
# Replace this with your actual class import if running within your project
class THzNomaSystem:
    def __init__(self, params=None):
        # Dummy init, params not used in this simplified example
        pass
    # Add any methods needed for data generation if they are part of the class
    def generate_focused_data(self, M, D=2):
        """Generates placeholder data for 'Focused' distribution (e.g., distinct clusters)"""
        print(f"Generating {M} points for Focused distribution...")
        # Example: 3 tight clusters
        if M < 3: M = 3 # Need at least 1 point per cluster
        n_per_cluster = M // 3
        remainder = M % 3
        X = np.vstack([
            np.random.randn(n_per_cluster + (1 if remainder > 0 else 0), D) * 0.5 + [0, 5],
            np.random.randn(n_per_cluster + (1 if remainder > 1 else 0), D) * 0.5 + [5, 0],
            np.random.randn(n_per_cluster, D) * 0.5 + [-5, 0],
        ])
        np.random.shuffle(X) # Shuffle order
        print(f"  Generated shape: {X.shape}")
        return X

    def generate_overlapping_data(self, M, D=2):
        """Generates placeholder data for 'Overlapping' distribution (e.g., more spread out)"""
        print(f"Generating {M} points for Overlapping distribution...")
        # Example: Points from a wider distribution or slightly overlapping clusters
        X = np.random.rand(M, D) * 10 - 5 # Uniform distribution in a box
        print(f"  Generated shape: {X.shape}")
        return X

# --- Helper functions (can be inside the class or standalone) ---

def calculate_merge_costs(X):
    """Performs AHC linkage and returns raw merge costs."""
    if X.shape[0] < 2:
        return np.array([])
    # Perform hierarchical clustering using Ward's method
    Z = linkage(X, method='ward')
    # Extract raw merge costs (distance/criterion value at each merge step)
    raw_costs = Z[:, 2]
    return raw_costs

def scale_costs(costs):
    """Scale merge costs to desired range [0.25, 1.75]"""
    if len(costs) == 0:
        return np.array([])

    min_cost = np.min(costs)
    max_cost = np.max(costs)

    if max_cost == min_cost:
        # Avoid division by zero; return a constant value (e.g., midpoint)
        return np.full_like(costs, (0.25 + 1.75) / 2)

    # Linear scaling
    scaled = (costs - min_cost) / (max_cost - min_cost)
    return 0.25 + scaled * (1.75 - 0.25)

def plot_comparison_merge_costs(costs_overlap, costs_focused, M):
    """Plots the comparison merge cost graph."""
    if len(costs_overlap) == 0 and len(costs_focused) == 0:
        print("No costs to plot.")
        return

    plt.figure(figsize=(8, 5)) # Adjust figure size as needed

    # --- X-axis: Number of clusters ---
    # There are M-1 merge costs. The number of clusters remaining after merge 'i' (0-indexed) is M-(i+1)
    # So the sequence of remaining clusters is [M-1, M-2, ..., 1]
    num_merges = M - 1
    n_clusters_remaining = np.arange(num_merges, 0, -1) # Goes from M-1 down to 1

    # --- Plotting ---
    if len(costs_overlap) == num_merges:
         # Apply scaling
        scaled_costs_overlap = scale_costs(costs_overlap)
        plt.plot(n_clusters_remaining, scaled_costs_overlap, marker='D', linestyle='-', color='tab:blue', label='Overlapping distribution', markerfacecolor='none') # Diamond marker
    else:
        print(f"Warning: Overlapping costs length ({len(costs_overlap)}) doesn't match expected ({num_merges}). Skipping plot.")


    if len(costs_focused) == num_merges:
         # Apply scaling
        scaled_costs_focused = scale_costs(costs_focused)
        plt.plot(n_clusters_remaining, scaled_costs_focused, marker='s', linestyle='-', color='tab:orange', label='Focused distribution', markerfacecolor='none') # Square marker
    else:
         print(f"Warning: Focused costs length ({len(costs_focused)}) doesn't match expected ({num_merges}). Skipping plot.")

    # --- Formatting ---
    plt.gca().invert_xaxis() # Invert x-axis to show 1 to M-1 from left to right
    plt.xlabel('Number of clusters', fontweight='bold')
    plt.ylabel('Merge Cost', fontweight='bold')
    # plt.title('AHC Merge Cost vs. Number of Clusters') # Optional title
    plt.legend()
    plt.xticks(np.arange(1, M)) # Set ticks explicitly from 1 to M-1 (which is 11 if M=12)
    plt.yticks(np.arange(0, 2.0, 0.25)) # Match y-ticks from target image
    plt.ylim(0, 1.85) # Match y-limits from target image
    # plt.grid(True) # Optional grid
    plt.show()

# --- Main execution ---
def main():
    M = 12 # Number of users/points (to get 11 merge costs)
    D = 2  # Dimensionality of data (example uses 2D for easy visualization if needed)

    # Initialize system/data generator
    system = THzNomaSystem() # Using the dummy class here

    # --- Scenario 1: Overlapping Distribution ---
    print("Running Overlapping Scenario...")
    X_overlapping = system.generate_overlapping_data(M, D)
    raw_costs_overlapping = calculate_merge_costs(X_overlapping)
    print(f"  Raw costs (Overlap): {np.round(raw_costs_overlapping, 2)}")


    # --- Scenario 2: Focused Distribution ---
    print("\nRunning Focused Scenario...")
    X_focused = system.generate_focused_data(M, D)
    raw_costs_focused = calculate_merge_costs(X_focused)
    print(f"  Raw costs (Focused): {np.round(raw_costs_focused, 2)}")

    # --- Plotting ---
    # Pass the RAW costs to the plotting function, scaling happens inside
    plot_comparison_merge_costs(raw_costs_overlapping, raw_costs_focused, M)

if __name__ == "__main__":
    main()