import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors

# Dummy THzNomaSystem class for standalone execution
class THzNomaSystem:
    def __init__(self, params=None):
        # Dummy init, params not used in this simplified example
        pass
    
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

# --- Functions for deployment visualization ---

def generate_focused_deployment_v_target(N_per_cluster=[8, 7, 10], D=2):
    """Generates placeholder data aiming to visually match the 'Focused' deployment target."""
    print(f"Generating {sum(N_per_cluster)} points for Focused deployment (Target Visual Match)...")
    means = {'blue': [1.8, 5.0], 'red': [4.8, 5.2], 'green':[3.3, 1.5]}
    stds = {'blue': 0.3, 'red': 0.25, 'green':0.4}
    colors = ['blue', 'red', 'green']
    n_points = N_per_cluster
    all_points, point_colors = [], []
    np.random.seed(40) # Keep seed for visual consistency
    for i, color in enumerate(colors):
        points = np.random.randn(n_points[i], D) * stds[color] + means[color]
        all_points.append(points)
        point_colors.extend([color] * n_points[i])
    X = np.vstack(all_points)
    marker_locations = np.array([[2.0, 5.0], [5.0, 5.0], [3.5, 1.5]])
    print(f"  Focused DEPLOYMENT generated shape: {X.shape}")
    return X, point_colors, marker_locations

def generate_overlapping_deployment_v_target(N_per_cluster=[8, 7, 10], D=2):
    """Generates placeholder data aiming to visually match the 'Overlapping' deployment target."""
    print(f"Generating {sum(N_per_cluster)} points for Overlapping deployment (Target Visual Match)...")
    means = {'green': [2.0, 5.5], 'blue': [5.5, 5.0], 'red': [3.5, 1.5]}
    stds = {'green': 0.9, 'blue': 0.8, 'red': 1.1}
    colors = ['green', 'blue', 'red'] # Match color order to means for generation
    n_points = N_per_cluster
    all_points, point_colors = [], []
    np.random.seed(42) # Keep seed for visual consistency
    for i, color in enumerate(colors):
        points = np.random.randn(n_points[i], D) * stds[color] + means[color]
        all_points.append(points)
        point_colors.extend([color] * n_points[i])
    X = np.vstack(all_points)
    marker_locations = np.array([[2.0, 5.0], [5.0, 5.0], [3.5, 1.5]])
    print(f"  Overlapping DEPLOYMENT generated shape: {X.shape}")
    return X, point_colors, marker_locations

# --- AHC Merge Cost functions ---

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

# --- Combined Plotting Function ---

def plot_deployments_and_merge_costs(
        # Data for deployment plots
        X_focused_deploy, colors_focused_deploy, markers_focused_deploy,
        X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy,
        # Data for merge costs plot
        costs_overlap, costs_focused, M):
    """
    Plots the deployment scenarios and merge costs in a combined figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5)) # 1 row, 3 columns

    unique_plot_colors = ['blue', 'red', 'green'] # Consistent plotting colors

    # --- Plot (a): Focused Deployment ---
    ax1 = axes[0]
    for color in unique_plot_colors:
        idx = [i for i, c in enumerate(colors_focused_deploy) if c == color]
        if idx: ax1.scatter(X_focused_deploy[idx, 0], X_focused_deploy[idx, 1], c=color, s=35, alpha=0.7)
    ax1.scatter(markers_focused_deploy[:, 0], markers_focused_deploy[:, 1], c='black', marker='D', s=60)
    ax1.set_xlim(1, 6); ax1.set_ylim(0, 6.2) # Limits from target deployment (a)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(False)
    ax1.set_title('(a) Focused user deployment.', fontsize=14, y=-0.20) # Caption

    # --- Plot (b): Overlapping Deployment ---
    ax2 = axes[1]
    for color in unique_plot_colors:
         idx = [i for i, c in enumerate(colors_overlap_deploy) if c == color]
         if idx: ax2.scatter(X_overlap_deploy[idx, 0], X_overlap_deploy[idx, 1], c=color, s=35, alpha=0.7)
    ax2.scatter(markers_overlap_deploy[:, 0], markers_overlap_deploy[:, 1], c='black', marker='D', s=60)
    ax2.set_xlim(0, 7.5); ax2.set_ylim(-1.0, 8.0) # Limits from corrected deployment (b)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(False)
    ax2.set_title('(b) Overlapping user deployment.', fontsize=14, y=-0.20) # Caption

    # --- Plot (c): Merge Cost Analysis ---
    ax3 = axes[2]
    
    # Prepare x-axis values (number of clusters)
    num_merges = M - 1
    n_clusters_remaining = np.arange(num_merges, 0, -1) # Goes from M-1 down to 1
    
    # Plot merge costs
    if len(costs_overlap) == num_merges:
        # Apply scaling
        scaled_costs_overlap = scale_costs(costs_overlap)
        ax3.plot(n_clusters_remaining, scaled_costs_overlap, marker='D', linestyle='-', 
                color='tab:blue', label='Overlapping distribution', markerfacecolor='none')
    else:
        print(f"Warning: Overlapping costs length ({len(costs_overlap)}) doesn't match expected ({num_merges}). Skipping plot.")

    if len(costs_focused) == num_merges:
        # Apply scaling
        scaled_costs_focused = scale_costs(costs_focused)
        ax3.plot(n_clusters_remaining, scaled_costs_focused, marker='s', linestyle='-', 
                color='tab:orange', label='Focused distribution', markerfacecolor='none')
    else:
        print(f"Warning: Focused costs length ({len(costs_focused)}) doesn't match expected ({num_merges}). Skipping plot.")
    
    # Formatting merge cost plot
    ax3.invert_xaxis() # Invert x-axis to show 1 to M-1 from left to right
    ax3.set_xlabel('Number of clusters', fontweight='bold')
    ax3.set_ylabel('Merge Cost', fontweight='bold')
    ax3.legend()
    ax3.set_xticks(np.arange(1, M)) # Set ticks explicitly from 1 to M-1
    ax3.set_yticks(np.arange(0, 2.0, 0.25)) # Match y-ticks from target image
    ax3.set_ylim(0, 1.85) # Match y-limits from target image
    ax3.grid(False)
    ax3.set_title('(c) Merge cost in AHC.', fontsize=14, y=-0.20) # Caption

    # Overall adjustments
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust rect bottom margin
    plt.show()

# --- Main execution ---
def main():
    M = 12 # Number of users/points (to get 11 merge costs)
    D = 2  # Dimensionality of data (example uses 2D for easy visualization if needed)
    N_users_per_cluster_deploy = [8, 7, 10] # Total 25 users for deployment visuals

    # Initialize system/data generator
    system = THzNomaSystem() # Using the dummy class here

    # --- Generate Data for Deployment Plots ---
    print("--- Generating Data for Deployment Plots ---")
    X_focused_deploy, colors_focused_deploy, markers_focused_deploy = generate_focused_deployment_v_target(N_users_per_cluster_deploy, D)
    X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy = generate_overlapping_deployment_v_target(N_users_per_cluster_deploy, D)

    # --- Generate Data for AHC Analysis ---
    print("\n--- Generating Data for AHC Analysis ---")
    
    # Scenario 1: Overlapping Distribution
    print("Running Overlapping Scenario...")
    X_overlapping = system.generate_overlapping_data(M, D)
    raw_costs_overlapping = calculate_merge_costs(X_overlapping)
    print(f"  Raw costs (Overlap): {np.round(raw_costs_overlapping, 2)}")

    # Scenario 2: Focused Distribution
    print("\nRunning Focused Scenario...")
    X_focused = system.generate_focused_data(M, D)
    raw_costs_focused = calculate_merge_costs(X_focused)
    print(f"  Raw costs (Focused): {np.round(raw_costs_focused, 2)}")

    # --- Plotting Combined Figure ---
    print("\n--- Generating Combined Plot ---")
    plot_deployments_and_merge_costs(
        # Deployment data
        X_focused_deploy, colors_focused_deploy, markers_focused_deploy,
        X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy,
        # Merge costs data
        raw_costs_overlapping, raw_costs_focused, M
    )

if __name__ == "__main__":
    main()