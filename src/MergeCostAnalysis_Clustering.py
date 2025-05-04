import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
import matplotlib.cm as cm

# --- Data Generation Functions ---
def generate_focused_data_for_ahc(n_points, D=2, centers=None, std_dev=0.5, seed=101):
    """Generates data with distinct clusters for AHC analysis."""
    np.random.seed(seed)
    if centers is None:
        centers = [[0, 5], [5, 0], [-5, 0]] # Example centers
    
    n_clusters = len(centers)
    if n_points < n_clusters:
        n_points = n_clusters
        
    points_per_cluster = n_points // n_clusters
    remainder = n_points % n_clusters
    
    X = []
    for i in range(n_clusters):
        num = points_per_cluster + (1 if i < remainder else 0)
        cluster_points = np.random.randn(num, D) * std_dev + centers[i]
        X.append(cluster_points)
        
    X = np.vstack(X)
    np.random.shuffle(X) # Shuffle order doesn't matter for linkage
    return X

def generate_overlapping_data_for_ahc(n_points, D=2, scale=10, offset=-5, seed=202):
    """Generates more spread-out/uniform data for AHC analysis."""
    np.random.seed(seed)
    # Example: Uniform distribution or wider Gaussian clouds
    X = np.random.rand(n_points, D) * scale + offset
    return X

def generate_focused_deployment_v_target(N_per_cluster=[8, 7, 10], D=2):
    """Generates placeholder data aiming to visually match the 'Focused' deployment target."""
    means = {'blue': [1.8, 5.0], 'red': [4.8, 5.2], 'green':[3.3, 1.5]}
    stds = {'blue': 0.3, 'red': 0.25, 'green':0.4}
    # Original colors used for generation, not necessarily the final clustering
    gen_colors = ['blue', 'red', 'green'] 
    n_points = N_per_cluster
    all_points = []
    true_colors_for_plotting = [] # Store the generation color for 'before' plot
    np.random.seed(40) 
    for i, color in enumerate(gen_colors):
        points = np.random.randn(n_points[i], D) * stds[color] + means[color]
        all_points.append(points)
        true_colors_for_plotting.extend([color] * n_points[i]) # Use actual colors
    X = np.vstack(all_points)
    # Centroids from the visual target plot (represent ideal centers)
    target_markers = np.array([[2.0, 5.0], [5.0, 5.0], [3.5, 1.5]])
    return X, true_colors_for_plotting, target_markers

def generate_overlapping_deployment_v_target(N_per_cluster=[8, 7, 10], D=2):
    """Generates placeholder data aiming to visually match the 'Overlapping' deployment target."""
    means = {'green': [2.0, 5.5], 'blue': [5.5, 5.0], 'red': [3.5, 1.5]} 
    stds = {'green': 1.2, 'blue': 0.9, 'red': 1.5} 
    gen_colors = ['green', 'blue', 'red'] 
    n_points = N_per_cluster
    all_points = []
    true_colors_for_plotting = []
    np.random.seed(42) 
    for i, color in enumerate(gen_colors):
        points = np.random.randn(n_points[i], D) * stds[color] + means[color]
        all_points.append(points)
        true_colors_for_plotting.extend([color] * n_points[i])
    X = np.vstack(all_points)
    # Target markers (can be same as focused or adjusted based on visual)
    target_markers = np.array([[2.0, 5.0], [5.0, 5.0], [3.5, 1.5]])
    return X, true_colors_for_plotting, target_markers


# --- AHC Execution and Analysis ---
def determine_optimal_clusters(X_focused, X_overlap, max_clusters=10):
    """Determines optimal number of clusters using the dendrogram distance analysis for both deployments."""
    if X_focused.shape[0] < 2 or X_overlap.shape[0] < 2:
        print("Warning: Not enough points for clustering.")
        return 1, 1, None, None
    
    # Calculate the linkage matrices
    Z_focused = linkage(X_focused, method='ward')
    Z_overlap = linkage(X_overlap, method='ward')
    
    # Extract distances from the linkage matrices
    distances_focused = Z_focused[:, 2]
    distances_overlap = Z_overlap[:, 2]
    
    # Number of clusters at each merge step (from n to 1)
    n_samples_focused = X_focused.shape[0]
    n_samples_overlap = X_overlap.shape[0]
    
    # Create the x-axis values: number of clusters at each merge step
    n_clusters_range_focused = np.arange(n_samples_focused, 1, -1)[:len(distances_focused)]
    n_clusters_range_overlap = np.arange(n_samples_overlap, 1, -1)[:len(distances_overlap)]
    
    # Plot the distances to visualize the elbow/knee
    plt.figure(figsize=(12, 8))
    
    # Plot both curves
    plt.plot(n_clusters_range_focused, distances_focused, 'b-', marker='o', markersize=5, label='Focused Deployment')
    plt.plot(n_clusters_range_overlap, distances_overlap, 'r-', marker='s', markersize=5, label='Overlapping Deployment')
    
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Merge Distance', fontsize=14)
    plt.title('Hierarchical Clustering Merge Cost Comparison', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    
    # Try to find the knee/elbow points using KneeLocator
    try:
        from kneed import KneeLocator
        
        # Find the knee/elbow for focused deployment
        kneedle_focused = KneeLocator(
            n_clusters_range_focused, distances_focused, S=1.0, curve="convex", direction="decreasing"
        )
        optimal_k_focused = int(kneedle_focused.knee) if kneedle_focused.knee else None
        
        # Find the knee/elbow for overlapping deployment
        kneedle_overlap = KneeLocator(
            n_clusters_range_overlap, distances_overlap, S=1.0, curve="convex", direction="decreasing"
        )
        optimal_k_overlap = int(kneedle_overlap.knee) if kneedle_overlap.knee else None
        
        if optimal_k_focused:
            plt.axvline(x=optimal_k_focused, color='b', linestyle='--', linewidth=2,
                       label=f'Focused optimal: {optimal_k_focused}')
        if optimal_k_overlap:
            plt.axvline(x=optimal_k_overlap, color='r', linestyle='--', linewidth=2,
                       label=f'Overlapping optimal: {optimal_k_overlap}')
        
        plt.legend(fontsize=12)
    except (ImportError, Exception) as e:
        print(f"Error finding knee: {e}")
        # Fallback: Simple heuristic based on the elbow in the curve
        # Use the point of maximum acceleration
        acceleration_focused = np.diff(distances_focused, 2) if len(distances_focused) >= 3 else np.diff(distances_focused, 1)
        acceleration_overlap = np.diff(distances_overlap, 2) if len(distances_overlap) >= 3 else np.diff(distances_overlap, 1)
        
        idx_focused = np.argmax(acceleration_focused) if len(acceleration_focused) > 0 else 0
        idx_overlap = np.argmax(acceleration_overlap) if len(acceleration_overlap) > 0 else 0
        
        optimal_k_focused = int(n_clusters_range_focused[min(idx_focused, len(n_clusters_range_focused)-1)])
        optimal_k_overlap = int(n_clusters_range_overlap[min(idx_overlap, len(n_clusters_range_overlap)-1)])
        
        plt.axvline(x=optimal_k_focused, color='b', linestyle='--', linewidth=2,
                   label=f'Focused heuristic: {optimal_k_focused}')
        plt.axvline(x=optimal_k_overlap, color='r', linestyle='--', linewidth=2,
                   label=f'Overlapping heuristic: {optimal_k_overlap}')
        plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Ensure we have reasonable numbers of clusters (at least 2, at most max_clusters)
    optimal_k_focused = max(2, min(optimal_k_focused, max_clusters))
    optimal_k_overlap = max(2, min(optimal_k_overlap, max_clusters))
    
    return optimal_k_focused, optimal_k_overlap, Z_focused, Z_overlap


def perform_ahc_and_get_clusters(X, n_clusters=None, max_clusters=10, dataset_type="Dataset"):
    """Performs AHC and returns cluster labels and calculated centroids.
    If n_clusters is None, determines optimal number automatically."""
    if X.shape[0] < 2:
        print("Warning: Not enough points for clustering.")
        return None, None, None
    
    # If no specific number of clusters is provided, determine it
    if n_clusters is None:
        optimal_k, Z = determine_optimal_clusters(X, max_clusters, dataset_type)
        n_clusters = optimal_k
        print(f"Determined optimal number of clusters: {n_clusters}")
    else:
        # Calculate linkage matrix
        Z = linkage(X, method='ward')
    
    # Cut the dendrogram to get the specified number of clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Calculate centroids for the obtained clusters
    centroids = np.zeros((n_clusters, X.shape[1]))
    unique_labels = np.unique(labels)
    
    if len(unique_labels) != n_clusters:
        print(f"Warning: Found {len(unique_labels)} clusters, expected {n_clusters}. Adjusting.")
        n_clusters = len(unique_labels)
        centroids = np.zeros((n_clusters, X.shape[1]))

    for i, label_id in enumerate(unique_labels):
        # Find points belonging to this cluster
        cluster_points = X[labels == label_id]
        if cluster_points.shape[0] > 0:
             # Calculate mean (centroid)
            centroids[i, :] = np.mean(cluster_points, axis=0)
        else:
            print(f"Warning: Cluster label {label_id} has no points assigned.")
            centroids[i, :] = np.nan
            
    return labels, centroids, n_clusters


# --- Plotting Function for Before vs. After ---
def plot_deployment_clusters(X, true_colors, labels, centroids, k, dataset_type, before_after):
    """Plot a single clustering view (either before or after)."""
    plt.figure(figsize=(10, 8))
    
    # Use better color maps and marker sizes
    plot_cmap = cm.tab10 
    marker_size = 80
    centroid_size = 200
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
        plt.title(f'{dataset_type} Deployment (Before Clustering)', fontsize=title_size)
        
    else:  # after clustering
        if labels is not None:
            # Map labels to colors using a consistent colormap
            unique_labels = np.unique(labels)
            
            # Plot each cluster with a different color
            for label in unique_labels:
                mask = labels == label
                plt.scatter(X[mask, 0], X[mask, 1], 
                           c=[plot_cmap(int(label-1) % 10)], s=marker_size, alpha=0.8,
                           label=f'Cluster {label}', edgecolors='k', linewidths=0.5)
            
            # Add centroids
            if centroids is not None and centroids.shape[0] > 0:
                plt.scatter(centroids[:, 0], centroids[:, 1], 
                            c='red', marker='X', s=centroid_size, 
                            edgecolors='k', linewidths=1.5,
                            label=f'AHC Centroids (k={k})')
        plt.title(f'{dataset_type} Deployment (After AHC, k={k})', fontsize=title_size)
    
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


# --- Visualize Dendrogram Function ---
def plot_dendrogram(Z, title="Hierarchical Clustering Dendrogram"):
    """Plot the dendrogram for the hierarchical clustering."""
    plt.figure(figsize=(14, 8))
    dendrogram(
        Z,
        truncate_mode='none',  # show all levels
        p=5,  # p for truncate_mode 'lastp'
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Sample index', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
def main():
    N_users_per_cluster_deploy = [8, 7, 10] # Total 25 users for deployment visuals
    D = 2
    max_clusters = 10  # Maximum number of clusters to consider

    # --- Generate Data for Deployment Plots ---
    print("--- Generating Data for Deployment Plots ---")
    X_focused_deploy, colors_focused_deploy, markers_focused_deploy = generate_focused_deployment_v_target(N_users_per_cluster_deploy, D)
    X_overlap_deploy, colors_overlap_deploy, markers_overlap_deploy = generate_overlapping_deployment_v_target(N_users_per_cluster_deploy, D)

    # --- Perform AHC on Deployment Data with Dynamic Clustering ---
    print("\n--- Performing AHC with Dynamic Clustering on Deployment Data ---")
    print("\nAnalyzing both deployments:")
    optimal_k_focused, optimal_k_overlap, Z_focused, Z_overlap = determine_optimal_clusters(X_focused_deploy, X_overlap_deploy, max_clusters)
    
    print(f"\nDetermined optimal clusters - Focused: {optimal_k_focused}, Overlapping: {optimal_k_overlap}")
    
    # Get clusters for focused deployment
    labels_focused, centroids_focused, k_focused = perform_ahc_and_get_clusters(X_focused_deploy, n_clusters=optimal_k_focused, max_clusters=max_clusters)
    
    # Get clusters for overlapping deployment
    labels_overlap, centroids_overlap, k_overlap = perform_ahc_and_get_clusters(X_overlap_deploy, n_clusters=optimal_k_overlap, max_clusters=max_clusters)

    if labels_focused is not None:
         print(f"Focused - Assigned {k_focused} clusters. Labels (sample): {labels_focused[:10]}...")
         print(f"Focused - Calculated Centroids:\n{centroids_focused}")
    if labels_overlap is not None:
         print(f"Overlap - Assigned {k_overlap} clusters. Labels (sample): {labels_overlap[:10]}...")
         print(f"Overlap - Calculated Centroids:\n{centroids_overlap}")

    # --- Plot each view separately ---
    print("\n--- Generating Separate Plots ---")
    # Focused deployment - before clustering
    plot_deployment_clusters(X_focused_deploy, colors_focused_deploy, None, None, None, 
                            "Focused", "before")
    
    # Focused deployment - after clustering
    plot_deployment_clusters(X_focused_deploy, colors_focused_deploy, labels_focused, 
                            centroids_focused, k_focused, "Focused", "after")
    
    # Overlapping deployment - before clustering
    plot_deployment_clusters(X_overlap_deploy, colors_overlap_deploy, None, None, None, 
                            "Overlapping", "before")
    
    # Overlapping deployment - after clustering
    plot_deployment_clusters(X_overlap_deploy, colors_overlap_deploy, labels_overlap, 
                            centroids_overlap, k_overlap, "Overlapping", "after")
    
    # Plot the dendrograms
    plot_dendrogram(Z_focused, "Focused Deployment Dendrogram")
    plot_dendrogram(Z_overlap, "Overlapping Deployment Dendrogram")

if __name__ == "__main__":
    main()