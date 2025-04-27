import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# --- Data Generation Set 1: For Visual Deployment Plots ---

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

# --- Data Generation Set 2: For K-Distance Plot Shape ---

class THzNomaSystem: # Using a class structure as in Code 1
    def __init__(self, params=None): pass

    def generate_focused_data_v3(self, N, D=2):
        """Generates data aiming for the target K-distance plot (ORANGE curve)."""
        print(f"Generating {N} points for Focused K-Distance (v3)...")
        n_per_cluster = N // 3
        remainder = N % 3
        n_points = [n_per_cluster + (1 if i < remainder else 0) for i in range(3)]
        if N < 3: return np.empty((0,D))
        cluster1 = np.random.randn(n_points[0], D) * 0.10 + [0, 0]
        cluster2 = np.random.randn(n_points[1], D) * 0.10 + [0.9, 0.9]
        cluster3 = np.random.randn(n_points[2], D) * 0.10 + [1.8, 1.8]
        X = np.vstack([cluster1, cluster2, cluster3])
        np.random.seed(101); np.random.shuffle(X)
        print(f"  Focused K-DIST generated shape: {X.shape}")
        return X

    def generate_overlapping_data_v3(self, N, D=2):
        """Generates data aiming for the target K-distance plot (BLUE curve)."""
        print(f"Generating {N} points for Overlapping K-Distance (v3)...")
        if N < 5: N = 5
        n_core = int(N * 0.80); n_transition = int(N*0.15); n_outliers = N - n_core - n_transition
        core_cluster = np.random.randn(n_core, D) * 0.06 + [0.5, 0.5]
        transition_points = np.random.randn(n_transition, D) * 0.12 + [0.8, 0.8]
        outliers = np.random.randn(n_outliers, D) * 0.15 + [1.5, 1.5]
        X = np.vstack([core_cluster, transition_points, outliers])
        np.random.seed(102); np.random.shuffle(X)
        print(f"  Overlapping K-DIST generated shape: {X.shape}")
        return X

# --- K-Distance Calculation Function (Unchanged) ---
def calculate_k_distances(X, k):
    """Calculates the distance to the k-th nearest neighbor for each point."""
    N = X.shape[0]
    if N <= k: return np.array([])
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = distances[:, k]
    sorted_k_distances = np.sort(k_distances)
    return sorted_k_distances


# --- Combined Plotting Function ---
def plot_deployments_and_kdist_target(
        # Data for deployment plots
        X_focused_deploy, colors_focused_deploy, markers_focused_deploy,
        X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy,
        # Calculated K-distances (from different data)
        distances_focused_kdist, distances_overlap_kdist, k,
        N_focused_kdist, N_overlap_kdist):
    """
    Plots the TARGET deployment scenarios and the TARGET K-distance graph.
    Uses separate data sources for deployments and K-distance calculations.
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

    # --- Plot (c): K-distance plot ---
    ax3 = axes[2]
    objects_overlap_kdist = np.arange(N_overlap_kdist)
    objects_focused_kdist = np.arange(N_focused_kdist)

    # Plot K-distances CALCULATED FROM K-DIST OPTIMIZED DATA
    if N_focused_kdist > 0 and len(distances_focused_kdist) == N_focused_kdist:
        ax3.plot(objects_focused_kdist, distances_focused_kdist, marker='x', linestyle='-', color='tab:orange', label='Focused distribution', markersize=6)
    else: print("Warning: Focused K-dist data issue. Skipping plot.")

    if N_overlap_kdist > 0 and len(distances_overlap_kdist) == N_overlap_kdist:
        ax3.plot(objects_overlap_kdist, distances_overlap_kdist, marker='s', linestyle='-', color='tab:blue', label='Overlapping distribution', markerfacecolor='none', markersize=6)
        print(f"  Plotting Overlapping K-Dist (Points: {N_overlap_kdist}, Min: {min(distances_overlap_kdist):.3f}, Max: {max(distances_overlap_kdist):.3f})")
    else: print("Warning: Overlapping K-dist data issue. Skipping plot.")

    # Formatting K-distance plot TO MATCH TARGET K-DIST IMAGE
    ax3.set_xlabel('Object', fontweight='bold') # Label from K-dist target
    ax3.set_ylabel('K-distance', fontweight='bold') # Label from K-dist target
    if N_focused_kdist > 0 or N_overlap_kdist > 0: ax3.legend()
    ax3.set_yticks(np.arange(0, 0.5, 0.1))  # Ticks from K-dist target
    ax3.set_ylim(-0.02, 0.45)                # Limits from K-dist target
    ax3.set_xticks(np.arange(0, 31, 5))     # Ticks from K-dist target
    ax3.set_xlim(-1, 30)                    # Limits from K-dist target
    ax3.grid(False)                         # Grid from K-dist target
    ax3.set_title('(c) NN in DBSCAN.', fontsize=14, y=-0.20) # Caption

    # Overall adjustments
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust rect bottom margin
    plt.show()


# --- Main Execution ---
def main():
    # Parameters for Deployment Plots
    N_users_per_cluster_deploy = [8, 7, 10] # Total 25 users for deployment visuals

    # Parameters for K-Distance Plot
    N_kdist = 30       # Number of points to generate for K-dist calculation (matches target axis)
    k_for_dbscan = 4   # Value of k for k-distance plot
    D = 2              # Data dimensionality

    # --- Generate Data Set 1 (For Deployment Plots) ---
    print("--- Generating Data for Deployment Plots ---")
    X_focused_deploy, colors_focused_deploy, markers_focused_deploy = generate_focused_deployment_v_target(N_users_per_cluster_deploy, D)
    X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy = generate_overlapping_deployment_v_target(N_users_per_cluster_deploy, D)

    # --- Generate Data Set 2 (For K-Distance Plot) ---
    print("\n--- Generating Data for K-Distance Plot ---")
    kdist_generator = THzNomaSystem()
    X_focused_kdist = kdist_generator.generate_focused_data_v3(N_kdist, D)
    X_overlap_kdist = kdist_generator.generate_overlapping_data_v3(N_kdist, D)
    N_focused_kdist_actual = X_focused_kdist.shape[0]
    N_overlap_kdist_actual = X_overlap_kdist.shape[0]

    # --- Calculate K-Distances (using Data Set 2) ---
    print(f"\n--- Calculating {k_for_dbscan}-Distances ---")
    k_dist_focused = calculate_k_distances(X_focused_kdist, k_for_dbscan)
    k_dist_overlap = calculate_k_distances(X_overlap_kdist, k_for_dbscan)

    # Print calculated ranges for debugging
    if len(k_dist_focused) > 0: print(f"  Focused K-Dist Range: Min={np.min(k_dist_focused):.3f}, Max={np.max(k_dist_focused):.3f}")
    if len(k_dist_overlap) > 0: print(f"  Overlap K-Dist Range: Min={np.min(k_dist_overlap):.3f}, Max={np.max(k_dist_overlap):.3f}")

    # --- Plotting ---
    print("\n--- Generating Combined Plot ---")
    plot_deployments_and_kdist_target(
        # Deployment data
        X_focused_deploy, colors_focused_deploy, markers_focused_deploy,
        X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy,
        # K-distance results (calculated from different data)
        k_dist_focused, k_dist_overlap, k_for_dbscan,
        N_focused_kdist_actual, N_overlap_kdist_actual
    )

if __name__ == "__main__":
    main()