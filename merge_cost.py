import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.stats import circmean
from sklearn.cluster import AgglomerativeClustering
from clustering import THzNomaSystem

class THzNomaSystem(THzNomaSystem):  # Inherits from your original class
    def __init__(self, params=None):
        super().__init__(params)
        self.merge_costs = []

    def _ahc_clustering(self, X_km, thetaS, M, K):
        """Modified AHC clustering with merge cost tracking"""
        if M >= 2:
            # Perform hierarchical clustering using Ward's method
            Z = linkage(X_km, method='ward')
            
            # Extract and scale merge costs
            raw_costs = Z[:, 2]
            self.merge_costs = self._scale_costs(raw_costs)
            
            # Continue with original clustering logic
            num_clusters = min(K, M)
            ahc = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(X_km)
            ahc_labels = ahc.labels_
            theta_ahc = np.array([
                circmean(thetaS[ahc_labels == cl]) for cl in range(num_clusters)
            ])
        else:
            ahc_labels = np.zeros(M, dtype=int)
            theta_ahc = thetaS.copy()
            self.merge_costs = []
            
        return ahc_labels, theta_ahc

    def _scale_costs(self, costs):
        """Scale merge costs to desired range (0.25-1.75)"""
        if len(costs) == 0:
            return []
            
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        
        if max_cost == min_cost:
            return np.full_like(costs, 1.0)
            
        scaled = (costs - min_cost) / (max_cost - min_cost)
        return 0.25 + scaled * 1.5

    def plot_merge_costs(self):
        """Plot merge costs for different user distributions"""
        plt.figure(figsize=(10, 6))
        
        # Original distribution
        n_clusters = np.arange(len(self.merge_costs), 0, -1)
        plt.plot(n_clusters, self.merge_costs, 'b-', label='Actual THz-NOMA Distribution')
        
        plt.gca().invert_xaxis()
        plt.xlabel('Number of Clusters')
        plt.ylabel('Scaled Merge Cost')
        plt.title('AHC Merge Cost Analysis for THz-NOMA Secondary Users')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Initialize system with parameters for focused distribution
    focused_params = {
        'Mvec': [11],  # Single value for clearer visualization
        'ct': 1        # Single trial to avoid averaging
    }
    
    # Create and run system
    system = THzNomaSystem(focused_params)
    results = system.run_simulation()
    
    # Plot merge costs from the last simulation trial
    system.plot_merge_costs()

if __name__ == "__main__":
    main()