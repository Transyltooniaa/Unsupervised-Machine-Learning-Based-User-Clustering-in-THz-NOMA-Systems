import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
from kneed import KneeLocator

# --- Data Generation Functions ---
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

class THzNomaSystem:
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

# --- K-Distance Calculation and Analysis ---
def calculate_k_distances(X, k):
    """Calculates the distance to the k-th nearest neighbor for each point."""
    N = X.shape[0]
    if N <= k: return np.array([])
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = distances[:, k]
    sorted_k_distances = np.sort(k_distances)
    return sorted_k_distances

def determine_optimal_eps(k_distances, dataset_type="standard"):
    """
    Determines the optimal epsilon value for DBSCAN based on the k-distance plot.
    Finds the "knee" or "elbow" point in the k-distance curve.
    
    Parameters:
    - k_distances: Array of sorted k-distances
    - dataset_type: String indicating the type of dataset ("focused" or "overlapping")
    """
    # If insufficient data, return a default value
    if len(k_distances) < 2:
        return 0.2  # Default fallback
    
    # Create the objects array (x-axis values)
    objects = np.arange(len(k_distances))
    
    # Plot the k-distance graph
    plt.figure(figsize=(12, 8))
    plt.plot(objects, k_distances, 'b-', marker='o', markersize=5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Points sorted by distance', fontsize=14)
    plt.ylabel(f'K-Distance', fontsize=14)
    plt.title(f'K-Distance Plot for {dataset_type.capitalize()} Dataset', fontsize=16)
    
    optimal_eps = None
    
    # Different strategies based on dataset type
    if dataset_type.lower() == "overlapping":
        # For overlapping data, use a more conservative approach
        # Use percentile method - find elbow in first third of curve
        
        # Look at the first third of the data for the knee
        third_idx = len(k_distances) // 3
        first_third = k_distances[:third_idx]
        first_third_objects = objects[:third_idx]
        
        try:
            # Try to find the knee in the first third
            kneedle = KneeLocator(
                first_third_objects, first_third, 
                S=1.0, curve="convex", direction="increasing"
            )
            knee_index = kneedle.knee
            
            if knee_index is not None:
                optimal_eps = first_third[knee_index]
                plt.axvline(x=knee_index, color='r', linestyle='--', linewidth=2,
                           label=f'Knee point: idx={knee_index}')
                plt.axhline(y=optimal_eps, color='g', linestyle='--', linewidth=2,
                           label=f'Optimal eps: {optimal_eps:.3f}')
            else:
                # If no knee found, use a stricter percentile
                pct_idx = int(len(k_distances) * 0.1)  # Use 10th percentile 
                optimal_eps = k_distances[pct_idx]
                plt.axvline(x=pct_idx, color='r', linestyle='--', linewidth=2,
                           label=f'Percentile point: idx={pct_idx}')
                plt.axhline(y=optimal_eps, color='g', linestyle='--', linewidth=2,
                           label=f'Percentile eps: {optimal_eps:.3f}')
        except Exception as e:
            print(f"Error in knee detection for overlapping data: {e}")
            # Fallback to a stricter percentile
            pct_idx = int(len(k_distances) * 0.1)  # Use 10th percentile 
            optimal_eps = k_distances[pct_idx]
            plt.axvline(x=pct_idx, color='r', linestyle='--', linewidth=2,
                       label=f'Fallback point: idx={pct_idx}')
            plt.axhline(y=optimal_eps, color='g', linestyle='--', linewidth=2,
                       label=f'Fallback eps: {optimal_eps:.3f}')
    else:
        # For focused data or standard approach, use the regular knee detection
        try:
            kneedle = KneeLocator(
                objects, k_distances, S=1.0, curve="convex", direction="increasing"
            )
            knee_index = kneedle.knee
            
            if knee_index is not None:
                optimal_eps = k_distances[knee_index]
                plt.axvline(x=knee_index, color='r', linestyle='--', linewidth=2,
                           label=f'Knee point: idx={knee_index}')
                plt.axhline(y=optimal_eps, color='g', linestyle='--', linewidth=2,
                           label=f'Optimal eps: {optimal_eps:.3f}')
            else:
                # If KneeLocator fails to find a knee, use simple heuristic
                print("KneeLocator couldn't determine a clear knee point. Using heuristic.")
                
                # Simple heuristic - find point of maximum curvature
                # Calculate first derivative (slope)
                dy = np.diff(k_distances)
                dx = np.diff(objects)
                slopes = dy/dx
                
                # Find the point of maximum change in slope
                slope_changes = np.diff(slopes)
                max_change_idx = np.argmax(np.abs(slope_changes))
                
                # Convert to original index (we lost 2 points due to two diff operations)
                optimal_idx = max_change_idx + 1  # +1 for the diff operation offset
                optimal_eps = k_distances[optimal_idx]
                
                plt.axvline(x=optimal_idx, color='r', linestyle='--', linewidth=2,
                           label=f'Heuristic knee: idx={optimal_idx}')
                plt.axhline(y=optimal_eps, color='g', linestyle='--', linewidth=2,
                           label=f'Heuristic eps: {optimal_eps:.3f}')
        except Exception as e:
            print(f"Error finding knee: {e}")
            
            # Simple fallback - use a percentile of the distances
            optimal_idx = int(len(k_distances) * 0.15)  # Use 15th percentile
            optimal_eps = k_distances[optimal_idx]
            
            plt.axvline(x=optimal_idx, color='r', linestyle='--', linewidth=2,
                       label=f'Percentile knee: idx={optimal_idx}')
            plt.axhline(y=optimal_eps, color='g', linestyle='--', linewidth=2,
                       label=f'Percentile eps: {optimal_eps:.3f}')
    
    # Enforce reasonable bounds on epsilon
    min_eps = min(k_distances) * 0.8  # Don't go below minimum distance
    max_eps = np.percentile(k_distances, 50)  # Don't exceed median distance
    
    if optimal_eps is None or optimal_eps > max_eps:
        print(f"Capping epsilon from {optimal_eps} to {max_eps:.4f} (median)")
        optimal_eps = max_eps
        plt.axhline(y=optimal_eps, color='y', linestyle=':', linewidth=2,
                   label=f'Capped eps: {optimal_eps:.3f}')
    
    if optimal_eps < min_eps:
        print(f"Raising epsilon from {optimal_eps} to {min_eps:.4f} (min threshold)")
        optimal_eps = min_eps
        plt.axhline(y=optimal_eps, color='y', linestyle=':', linewidth=2,
                   label=f'Min eps: {optimal_eps:.3f}')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return optimal_eps

# --- DBSCAN Execution Function ---
def perform_dbscan_clustering(X, min_samples=None, eps=None, dataset_type="standard"):
    """
    Performs DBSCAN clustering with automatic parameter determination if not provided.
    Returns cluster labels, estimated number of clusters, and parameters used.
    """
    if X.shape[0] < 2:
        print("Warning: Not enough points for clustering.")
        return None, 0, None, None
    
    # Set default min_samples if not provided
    if min_samples is None:
        # Common heuristic: min_samples = 2*dim
        min_samples = max(2 * X.shape[1], 4)  # At least 4 points
    
    # If eps not provided, determine optimal value
    if eps is None:
        # Calculate k-distances (k = min_samples - 1)
        k = min_samples - 1
        k_distances = calculate_k_distances(X, k)
        
        # Find optimal eps using the k-distance plot
        eps = determine_optimal_eps(k_distances, dataset_type)
        print(f"Automatically determined optimal eps for {dataset_type}: {eps:.4f}")
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Count number of clusters (-1 labels are noise points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    print(f"Parameters used: eps={eps:.4f}, min_samples={min_samples}")
    
    return labels, n_clusters, eps, min_samples

# --- Plotting Functions ---
def plot_kdistance_comparison(k_dist_focused, k_dist_overlap, k):
    """Plot the k-distance curves with optimal knee points and epsilon values."""
    plt.figure(figsize=(10, 8))
    
    N_focused = len(k_dist_focused)
    N_overlap = len(k_dist_overlap)
    objects_focused = np.arange(N_focused)
    objects_overlap = np.arange(N_overlap)
    
    # Plot k-distance curves
    if N_focused > 0:
        plt.plot(objects_focused, k_dist_focused, marker='x', linestyle='-', 
                color='tab:orange', label='Focused distribution', markersize=6)
    
    if N_overlap > 0:
        plt.plot(objects_overlap, k_dist_overlap, marker='s', linestyle='-', 
                color='tab:blue', label='Overlapping distribution', 
                markerfacecolor='none', markersize=6)
    
    # Find optimal knee points and epsilon values
    try:
        # For focused data
        kneedle_focused = KneeLocator(
            objects_focused, k_dist_focused, 
            S=1.0, curve="convex", direction="increasing"
        )
        knee_idx_focused = kneedle_focused.knee
        if knee_idx_focused is not None:
            eps_focused = k_dist_focused[knee_idx_focused]
            # Add vertical line for knee point
            plt.axvline(x=knee_idx_focused, color='tab:orange', linestyle='--', linewidth=1.5)
            # Add horizontal line for epsilon
            plt.axhline(y=eps_focused, color='tab:orange', linestyle=':', linewidth=1.5)
            # Add text labels - focused knee on axis
            plt.text(knee_idx_focused, -0.02, f'knee={knee_idx_focused}', 
                    color='tab:orange', fontsize=11, horizontalalignment='center', verticalalignment='top')
            plt.text(-1, eps_focused, f'eps={eps_focused:.3f}', 
                    color='tab:orange', fontsize=11, horizontalalignment='right', verticalalignment='center')
        
        # For overlapping data
        kneedle_overlap = KneeLocator(
            objects_overlap, k_dist_overlap, 
            S=1.0, curve="convex", direction="increasing"
        )
        knee_idx_overlap = kneedle_overlap.knee
        if knee_idx_overlap is not None:
            eps_overlap = k_dist_overlap[knee_idx_overlap]
            # Add vertical line for knee point
            plt.axvline(x=knee_idx_overlap, color='tab:blue', linestyle='--', linewidth=1.5)
            # Add horizontal line for epsilon
            plt.axhline(y=eps_overlap, color='tab:blue', linestyle=':', linewidth=1.5)
            # Add text labels - overlapping knee on graph
            plt.text(knee_idx_overlap, eps_overlap, f'knee={knee_idx_overlap}', 
                    color='tab:blue', fontsize=11, horizontalalignment='left', verticalalignment='bottom')
            plt.text(-1, eps_overlap, f'eps={eps_overlap:.3f}', 
                    color='tab:blue', fontsize=11, horizontalalignment='right', verticalalignment='center')
    except Exception as e:
        print(f"Error finding knee points: {e}")
        # Fallback to simple heuristic
        if N_focused > 0:
            # For focused data
            optimal_idx_focused = int(len(k_dist_focused) * 0.15)  # 15th percentile
            eps_focused = k_dist_focused[optimal_idx_focused]
            # Add vertical line for knee point
            plt.axvline(x=optimal_idx_focused, color='tab:orange', linestyle='--', linewidth=1.5)
            # Add horizontal line for epsilon
            plt.axhline(y=eps_focused, color='tab:orange', linestyle=':', linewidth=1.5)
            # Add text labels - focused knee on axis
            plt.text(optimal_idx_focused, -0.02, f'knee={optimal_idx_focused}', 
                    color='tab:orange', fontsize=11, horizontalalignment='center', verticalalignment='top')
            plt.text(-1, eps_focused, f'eps={eps_focused:.3f}', 
                    color='tab:orange', fontsize=11, horizontalalignment='right', verticalalignment='center')
        
        if N_overlap > 0:
            # For overlapping data
            optimal_idx_overlap = int(len(k_dist_overlap) * 0.1)  # 10th percentile
            eps_overlap = k_dist_overlap[optimal_idx_overlap]
            # Add vertical line for knee point
            plt.axvline(x=optimal_idx_overlap, color='tab:blue', linestyle='--', linewidth=1.5)
            # Add horizontal line for epsilon
            plt.axhline(y=eps_overlap, color='tab:blue', linestyle=':', linewidth=1.5)
            # Add text labels - overlapping knee on graph
            plt.text(optimal_idx_overlap, eps_overlap, f'knee={optimal_idx_overlap}', 
                    color='tab:blue', fontsize=11, horizontalalignment='left', verticalalignment='bottom')
            plt.text(-1, eps_overlap, f'eps={eps_overlap:.3f}', 
                    color='tab:blue', fontsize=11, horizontalalignment='right', verticalalignment='center')
    
    plt.xlabel('Object', fontweight='bold')
    plt.ylabel('K-distance', fontweight='bold')
    plt.legend(fontsize=12)
    plt.yticks(np.arange(0, 0.5, 0.1))
    plt.ylim(-0.02, 0.45)
    plt.xticks(np.arange(0, 31, 5))
    plt.xlim(-1, 30)
    plt.grid(True, alpha=0.3)
    plt.title(f'K-Distance Plot (k={k})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Return the epsilon values for use in DBSCAN
    return eps_focused if N_focused > 0 else None, eps_overlap if N_overlap > 0 else None

def plot_deployment_cluster(X, true_colors, labels, eps, min_samples, dataset_type, before_after):
    """Plot a single clustering view (either before or after DBSCAN)."""
    plt.figure(figsize=(10, 8))
    
    marker_size = 80
    font_size = 12
    title_size = 16
    
    if before_after == "before":
        # Plot original data with generation colors
        unique_gen_colors = np.unique(true_colors)
        for color in unique_gen_colors:
            idx = [i for i, c in enumerate(true_colors) if c == color]
            if idx: 
                plt.scatter(X[idx, 0], X[idx, 1], 
                           c=color, s=marker_size, alpha=0.8, 
                           label=f'Original Group: {color}',
                           edgecolors='k', linewidths=0.5)
        plt.title(f'{dataset_type} Deployment (Before DBSCAN)', fontsize=title_size)
        
    else:  # after clustering
        if labels is not None:
            # Use a colormap for clusters, with -1 (noise) as black
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels) - (1 if -1 in unique_labels else 0)))
            color_map = {}
            color_idx = 0
            
            # Assign colors to labels
            for label in unique_labels:
                if label == -1:
                    color_map[label] = 'k'  # Noise points are black
                else:
                    color_map[label] = colors[color_idx]
                    color_idx += 1
            
            # Plot each cluster
            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    # Noise points
                    plt.scatter(X[mask, 0], X[mask, 1], 
                              c='k', s=marker_size/2, alpha=0.5, 
                              label='Noise', edgecolors='k', linewidths=0.5)
                else:
                    # Regular clusters
                    plt.scatter(X[mask, 0], X[mask, 1], 
                               c=[color_map[label]], s=marker_size, alpha=0.8,
                               label=f'Cluster {label}', edgecolors='k', linewidths=0.5)
            
        plt.title(f'{dataset_type} Deployment (DBSCAN: eps={eps:.3f}, min_samples={min_samples})', 
                 fontsize=title_size)
    
    # Set axis limits based on dataset type
    if dataset_type == "Focused":
        plt.xlim(1, 6)
        plt.ylim(0, 6.2)
    else:  # Overlapping
        plt.xlim(0, 7.5)
        plt.ylim(-1.0, 8.0)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=font_size, loc='best')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("X", fontsize=font_size)
    plt.ylabel("Y", fontsize=font_size)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
def main():
    # Parameters for Deployment Plots
    N_users_per_cluster_deploy = [8, 7, 10] # Total 25 users for deployment visuals

    # Parameters for K-Distance Plot
    N_kdist = 30       # Number of points to generate for K-dist calculation
    k_for_dbscan = 4   # Value of k for k-distance plot (used for min_samples = k+1 in DBSCAN)
    D = 2              # Data dimensionality

    # --- Generate Data for Deployment Plots ---
    print("--- Generating Data for Deployment Plots ---")
    X_focused_deploy, colors_focused_deploy, markers_focused_deploy = generate_focused_deployment_v_target(N_users_per_cluster_deploy, D)
    X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy = generate_overlapping_deployment_v_target(N_users_per_cluster_deploy, D)

    # --- Calculate K-Distances for Deployment Data ---
    print(f"\n--- Calculating {k_for_dbscan}-Distances for Deployment Data ---")
    k_dist_focused_deploy = calculate_k_distances(X_focused_deploy, k_for_dbscan)
    k_dist_overlap_deploy = calculate_k_distances(X_overlap_deploy, k_for_dbscan)

    # --- Plot K-Distance Comparison (from deployment data) ---
    print("\n--- Plotting K-Distance Comparison for Deployment Data ---")
    plot_kdistance_comparison(k_dist_focused_deploy, k_dist_overlap_deploy, k_for_dbscan)

    # --- Perform DBSCAN on Focused Deployment ---
    print("\n--- Running DBSCAN on Focused Deployment Data ---")
    min_samples_focused = k_for_dbscan + 1  # Common setting: min_samples = k+1
    # Automatically determine eps for focused data
    labels_focused, n_clusters_focused, eps_focused, min_samples_focused = perform_dbscan_clustering(
        X_focused_deploy, min_samples=min_samples_focused, dataset_type="focused")

    # --- Perform DBSCAN on Overlapping Deployment ---
    print("\n--- Running DBSCAN on Overlapping Deployment Data ---")
    min_samples_overlap = k_for_dbscan + 1  # Common setting: min_samples = k+1
    # Automatically determine eps for overlapping data
    labels_overlap, n_clusters_overlap, eps_overlap, min_samples_overlap = perform_dbscan_clustering(
        X_overlap_deploy, min_samples=min_samples_overlap, dataset_type="overlapping")

    # --- Plot Before/After for Each Deployment ---
    print("\n--- Plotting Deployment Data (Before & After DBSCAN) ---")
    
    # Focused deployment - before clustering
    plot_deployment_cluster(X_focused_deploy, colors_focused_deploy, None, None, None, 
                          "Focused", "before")
    
    # Focused deployment - after clustering
    plot_deployment_cluster(X_focused_deploy, colors_focused_deploy, labels_focused, 
                          eps_focused, min_samples_focused, "Focused", "after")
    
    # Overlapping deployment - before clustering
    plot_deployment_cluster(X_overlap_deploy, colors_overlap_deploy, None, None, None, 
                          "Overlapping", "before")
    
    # Overlapping deployment - after clustering
    plot_deployment_cluster(X_overlap_deploy, colors_overlap_deploy, labels_overlap, 
                          eps_overlap, min_samples_overlap, "Overlapping", "after")
    
    # --- Optional: Additional Analysis for Paper-Like Visuals ---
    # Generate specialized data for k-distance visualization like in the paper
    print("\n--- Generating Data for Paper-Like K-Distance Plot ---")
    kdist_generator = THzNomaSystem()
    X_focused_paper = kdist_generator.generate_focused_data_v3(N_kdist, D)
    X_overlap_paper = kdist_generator.generate_overlapping_data_v3(N_kdist, D)
    
    # Calculate k-distances for paper-like plot
    k_dist_focused_paper = calculate_k_distances(X_focused_paper, k_for_dbscan)
    k_dist_overlap_paper = calculate_k_distances(X_overlap_paper, k_for_dbscan)
    
    # Plot paper-like k-distance comparison
    print("\n--- Plotting Paper-Like K-Distance Comparison ---")
    plot_kdistance_comparison(k_dist_focused_paper, k_dist_overlap_paper, k_for_dbscan)

if __name__ == "__main__":
    main()