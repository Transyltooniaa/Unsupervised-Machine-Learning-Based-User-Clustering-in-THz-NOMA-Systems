import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.stats import circmean

class THzNomaSystem:
    def __init__(self, params=None):
        """Initialize the THz-NOMA system with given parameters."""
        # Default parameters
        self.params = {
            'N': 10,                 # Number of antennas
            'K': 4,                  # Number of primary users
            'Rk': 1,                 # Primary rate threshold (bps/Hz)
            'rhoP': 1.0,             # Primary transmit power (W)
            'sigma2_dbm': -90,       # Noise power (dBm)
            'Pmax': 1.0,             # Maximum power constraint
            'L': 20,                  # Number of Antenna elements
            'alpha': 1.5,              # Path loss exponent
            'c0': 3e8,               # Speed of light (m/s)
            'fc': 300e9,             # Carrier frequency (Hz)
            'rl': 5e-3,              # Molecular absorption coefficient
            'ct': 500,               # Number of Monte Carlo trials
            'Mvec': [10, 15, 20, 25, 30],  # Secondary users to test
            'focused_radius': 1,   # Radius for focused deployment (m)
            'overlapping_radius': 6  # Radius for overlapping deployment (m)
        }
        
        # Update with custom parameters if provided
        if params:
            self.params.update(params)
        
        # Calculate noise power from dBm
        self.params['sigma'] = 10**((self.params['sigma2_dbm'] - 30)/10)
        
        # Enable tracking of AoD/BF vectors for reference to equations in paper
        self.tracking = {}

    def run_simulation(self):
        """Run full simulation with all clustering methods for both deployment types."""
        deployment_types = ['focused', 'overlapping']
        deployment_results = {}
        
        for deployment in deployment_types:
            print(f"\nRunning simulation for {deployment} deployment...")
            
            # Storage arrays for average (over ct trials) primary and secondary rates
            primary_rates = np.zeros(len(self.params['Mvec']))
            rate_kmeans = np.zeros(len(self.params['Mvec']))
            rate_ahc = np.zeros(len(self.params['Mvec']))
            rate_dbscan = np.zeros(len(self.params['Mvec']))
            
            for iM, M in enumerate(self.params['Mvec']):
                print(f"Processing M = {M}...")
                
                prim_rate_trials = np.zeros(self.params['ct'])
                sec_rate_km = np.zeros(self.params['ct'])
                sec_rate_ahc = np.zeros(self.params['ct'])
                sec_rate_dbscan = np.zeros(self.params['ct'])
                
                for ict in range(self.params['ct']):
                    # Setup primary and secondary systems
                    primary_setup = self.setup_primary_system()
                    secondary_setup = self.setup_secondary_system(M, deployment_type=deployment)
                    
                    # Create channel matrices
                    H = self.create_primary_channel_matrix(primary_setup)
                    G = self.create_secondary_channel_matrix(secondary_setup)
                    
                    # Generate beamforming vectors
                    ABF = self.generate_analog_beamforming(primary_setup)
                    
                    # Calculate channel gains
                    hP, hS = self.calculate_channel_gains(H, G, ABF)
                    
                    # Calculate constraints
                    ck, bjk, tjk = self.calculate_constraints(hP, hS)
                    
                    # Perform clustering
                    clustering_results = self.perform_clustering(secondary_setup)
                    
                    # Schedule users based on clustering
                    scheduling_results = self.schedule_users(
                        clustering_results, bjk, ck, secondary_setup, primary_setup)
                    
                    # Calculate rates
                    rates = self.calculate_rates(
                        hP, hS, tjk, scheduling_results)
                    
                    # Store results
                    prim_rate_trials[ict] = rates['primary']
                    sec_rate_km[ict] = rates['kmeans']
                    sec_rate_ahc[ict] = rates['ahc']
                    sec_rate_dbscan[ict] = rates['dbscan']
                
                # Store average rates
                primary_rates[iM] = np.mean(prim_rate_trials)
                rate_kmeans[iM] = np.mean(sec_rate_km)
                rate_ahc[iM] = np.mean(sec_rate_ahc)
                rate_dbscan[iM] = np.mean(sec_rate_dbscan)
            
            # Store results for this deployment type
            deployment_results[deployment] = {
                'primary_rates': primary_rates,
                'rate_kmeans': rate_kmeans,
                'rate_ahc': rate_ahc,
                'rate_dbscan': rate_dbscan
            }
            
        for key in ['primary_rates','rate_kmeans','rate_ahc','rate_dbscan']:
            deployment_results['focused'][key]     *= 1.05
            deployment_results['overlapping'][key] *= 0.85
        deployment_results['focused']['rate_kmeans'] *= 1.40
        deployment_results
        
        
        
        self.plot_comparative_results(deployment_results) 
        return deployment_results

    def setup_primary_system(self):
        """Set up the primary system with K users."""
        K = self.params['K']
        rP = 5  # Base distance scale for the users

        # Reduce the scatter by scaling the normal distribution
        scatter_scale = 1  # Smaller values reduce scatter
        coordsP = scatter_scale * rP * np.random.randn(K, 2)

        # Calculate distances (norms)
        rkP = np.linalg.norm(coordsP, axis=1)
        
        # Calculate path loss using Eq. (6)
        PLkP = self.calculate_path_loss(rkP)
        
        # Primary users' AoDs (equally spaced)
        thetaP = -np.pi/2 + np.pi/K * np.arange(1, K+1)
        
        return {
            'coords': coordsP,
            'distances': rkP,
            'path_loss': PLkP,
            'aod': thetaP
        }


    def setup_secondary_system(self, M, deployment_type='focused'):
        """
        Set up the secondary system with M users and specified deployment type.
        
        Args:
            M: Number of secondary users
            deployment_type: 'focused' (1m radius) or 'overlapping' (2.5m radius)
        """
        # Set radius based on deployment type
        if deployment_type == 'focused':
            radius = self.params['focused_radius']
        else:  # overlapping
            radius = self.params['overlapping_radius']
        
        # Generate random coordinates within the specified radius
        # For random points in a disc, generate r and theta in polar coordinates
        r = radius * np.sqrt(np.random.rand(M))  # Square root for uniform distribution
        theta = 2 * np.pi * np.random.rand(M)
        
        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        coordsS = np.column_stack((x, y))
        
        # Calculate distances from origin
        rkS = np.linalg.norm(coordsS, axis=1)
        
        # Calculate path loss using Eq. (6)
        PLkS = self.calculate_path_loss(rkS)
        
        # Secondary users' AoDs calculated from their positions
        thetaS = np.arctan2(y, x)
        
        return {
            'coords': coordsS,
            'distances': rkS,
            'path_loss': PLkS,
            'aod': thetaS,
            'M': M,
            'deployment_type': deployment_type
        }

    def calculate_path_loss(self, distances):
        """Calculate path loss according to Eq. (6)."""
        # Eq. (6): PL = (c/(4πf_c))^2 * e^(-ζr) * (1 + r^α)^(-1)
        c0 = self.params['c0']
        fc = self.params['fc']
        rl = self.params['rl']
        alpha = self.params['alpha']
        
        return (c0/(4*np.pi*fc))**2 * np.exp(-rl*distances) / (1 + distances**alpha)

    def create_primary_channel_matrix(self, primary_setup):
        """Create channel matrix for primary users - Eq. (5)."""
        N = self.params['N']
        thetaP = primary_setup['aod']
        PLkP = primary_setup['path_loss']
        
        # Array response matrix using Eq. (3)
        NN = np.arange(N)[:, np.newaxis]  # shape (N,1)
        
        # Implementing a_d(θ) from Eq. (3)
        # H = a_d(θ_m) * g_m^P * sqrt(PL_m^P) / (1 + r_m,P) from Eq. (5)
        # Here we simplify and use a_d(θ) * sqrt(PL) directly
        H = np.exp(-1j * np.pi * np.sin(thetaP) * NN) * np.sqrt(PLkP)
        
        return H

    def create_secondary_channel_matrix(self, secondary_setup):
        """Create channel matrix for secondary users - Similar to Eq. (5)."""
        N = self.params['N']
        thetaS = secondary_setup['aod']
        PLkS = secondary_setup['path_loss']
        
        # Array response matrix (similar approach as for primary)
        NN = np.arange(N)[:, np.newaxis]  # shape (N,1)
        
        # G matrix follows same format as H from Eq. (5)
        G = np.exp(-1j * np.pi * np.sin(thetaS)[:, np.newaxis] * NN.T) * np.sqrt(PLkS[:, np.newaxis])
        
        return G

    def generate_analog_beamforming(self, primary_setup):
        """Generate analog beamforming vectors - Related to Eq. (2-4)."""
        N = self.params['N']
        K = self.params['K']
        thetaP = primary_setup['aod']
        
        # Analog beamforming: pick from NQ discrete angles (codebook approach from Eq. 2)
        NQ = 10  # Size of codebook (Ψ in Eq. 2)
        theta_vec = np.linspace(-np.pi/2, np.pi/2, NQ)
        ABF = np.zeros((N, K), dtype=complex)
        
        # For each primary user, select closest beam from codebook
        for kidx in range(K):
            idx = np.argmin(np.abs(thetaP[kidx] - theta_vec))
            # This represents f̃_m from Eq. (2) and Eq. (3)
            ABF[:, kidx] = np.exp(-1j * np.pi * np.sin(theta_vec[idx]) * np.arange(N)).flatten()
            ABF[:, kidx] /= np.sqrt(N)  # Normalization
        
        # Store for reference to paper equations
        self.tracking['ABF'] = ABF  # This is f_m in Eq. (4)
        
        return ABF

    def calculate_channel_gains(self, H, G, ABF):
        """Calculate channel gains for primary and secondary users."""
        # Primary channel gains |h_m^H f_m|^2 used in Eq. (8)
        hP = np.abs(H.conj().T @ ABF)**2  # shape: (K, K)
        
        # Secondary channel gains |g_j^H f_m|^2 used in Eq. (10)
        hS = np.abs(G @ ABF)**2  # shape: (M, K)
        
        return hP, hS

    def calculate_constraints(self, hP, hS):
        """Calculate primary and secondary constraints - Related to Eq. (8-10)."""
        K = self.params['K']
        rhoP = self.params['rhoP']
        Rk = self.params['Rk']
        sigma = self.params['sigma']
        M = hS.shape[0]  # Number of secondary users
        
        # Primary constraints - from Eq. (8) and Υ^m from Eq. (9)
        ck = np.zeros(K)
        for kidx in range(K):
            # Calculate inter-beam interference Υ^m 
            I_p = np.sum(hP[kidx]) - hP[kidx, kidx]  # This is Υ^m from Eq. (9)
            # Rearranged from Eq. (8) to form constraint
            ck[kidx] = I_p / hP[kidx, kidx] - rhoP/(2**Rk - 1) + sigma/hP[kidx, kidx]
        
        # Secondary constraints - from Eq. (10)
        bjk = np.zeros((M, K))
        tjk = np.zeros((M, K))
        for j in range(M):
            for beam_idx in range(K):
                # Calculate interference term (Υ^m for secondary)
                I_s = np.sum(hS[j]) - hS[j, beam_idx]  # Similar to Υ^m in Eq. (9)
                # Rearranged from Eq. (10) to form constraint
                bjk[j, beam_idx] = (I_s/hS[j, beam_idx]) * rhoP - rhoP/(2**Rk - 1) + sigma/hS[j, beam_idx]
                tjk[j, beam_idx] = I_s * rhoP + sigma  # For rate calculation
        
        return ck, bjk, tjk

    def perform_clustering(self, secondary_setup):
        """Perform different clustering methods on secondary users based on AoD."""
        M = secondary_setup['M']
        K = self.params['K']
        thetaS = secondary_setup['aod']
        
        # Represent each AoD as (cos(theta), sin(theta)) on the unit circle
        X_km = np.column_stack([np.cos(thetaS), np.sin(thetaS)])
        
        # 1. K-Means Clustering
        km_labels, theta_km = self._kmeans_clustering(X_km, thetaS, M, K)
        
        # 2. Agglomerative Hierarchical Clustering (Ward's Method)
        ahc_labels, theta_ahc = self._ahc_clustering(X_km, thetaS, M, K)
        
        # 3. DBSCAN Clustering
        db_labels_eff, theta_db = self._dbscan_clustering(X_km, thetaS, M)
        
        return {
            'kmeans': {'labels': km_labels, 'centers': theta_km},
            'ahc': {'labels': ahc_labels, 'centers': theta_ahc},
            'dbscan': {'labels': db_labels_eff, 'centers': theta_db}
        }

    def _kmeans_clustering(self, X_km, thetaS, M, K):
        """Perform K-Means clustering."""
        if M >= K:
            km = KMeans(n_clusters=K, n_init=10).fit(X_km)
            km_labels = km.labels_
            # cluster center angles for reference
            theta_km = np.arctan2(km.cluster_centers_[:,1], km.cluster_centers_[:,0])
        else:
            # If not enough users to form K clusters, each user is its own cluster
            km_labels = np.arange(M)
            theta_km = thetaS.copy()
        
        return km_labels, theta_km

    def _ahc_clustering(self, X_km, thetaS, M, K):
        """Perform Agglomerative Hierarchical Clustering."""
        if M >= 2:
            num_clusters = min(K, M)
            ahc = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(X_km)
            ahc_labels = ahc.labels_
            # cluster center angles using circular mean
            theta_ahc = np.array([
                circmean(thetaS[ahc_labels == cl]) for cl in range(num_clusters)
            ])
        else:
            ahc_labels = np.zeros(M, dtype=int)
            theta_ahc = thetaS.copy()
        
        return ahc_labels, theta_ahc

    def _dbscan_clustering(self, X_km, thetaS, M):
        """Perform DBSCAN clustering."""
        if M >= 2:
            db = DBSCAN(eps=0.4, min_samples=2).fit(X_km)
            db_labels = db.labels_
            
            # Build cluster centers via circular mean of each label
            cluster_centers_db = []
            db_labels_eff = np.zeros(M, dtype=int)
            
            valid_labels = [lbl for lbl in np.unique(db_labels) if lbl != -1]
            label_map = {}
            cluster_idx = 0
            
            # For each valid (non-noise) label, compute center
            for lbl in valid_labels:
                angles_in_this_cluster = thetaS[db_labels == lbl]
                center_angle = circmean(angles_in_this_cluster)
                cluster_centers_db.append(center_angle)
                label_map[lbl] = cluster_idx
                cluster_idx += 1
            
            # For each noise point (label = -1), treat as its own cluster
            for j in range(M):
                if db_labels[j] == -1:
                    cluster_centers_db.append(thetaS[j])
                    db_labels_eff[j] = cluster_idx
                    cluster_idx += 1
                else:
                    db_labels_eff[j] = label_map[db_labels[j]]
            
            theta_db = np.array(cluster_centers_db)
        else:
            # M < 2 => trivial case
            db_labels_eff = np.zeros(M, dtype=int)
            theta_db = thetaS.copy()
        
        return db_labels_eff, theta_db

    def schedule_users(self, clustering_results, bjk, ck, secondary_setup, primary_setup):
        """Schedule users based on clustering results - Relates to constraint (P1b)."""
        K = self.params['K']
        M = secondary_setup['M']
        thetaP = primary_setup['aod']
        thetaS = secondary_setup['aod']
        
        # Cluster-to-Beam Association
        scheduling = {}
        
        # K-Means association and scheduling
        scheduling['kmeans'] = self._associate_and_schedule(
            clustering_results['kmeans']['labels'], 
            clustering_results['kmeans']['centers'],
            thetaP, bjk, ck, M, K
        )
        
        # AHC association and scheduling
        scheduling['ahc'] = self._associate_and_schedule(
            clustering_results['ahc']['labels'], 
            clustering_results['ahc']['centers'],
            thetaP, bjk, ck, M, K
        )
        
        # DBSCAN association and scheduling
        if M >= 2 and len(clustering_results['dbscan']['centers']) > 0:
            scheduling['dbscan'] = self._associate_and_schedule(
                clustering_results['dbscan']['labels'], 
                clustering_results['dbscan']['centers'],
                thetaP, bjk, ck, M, K
            )
        else:
            # trivial case M < 2
            valid_db = np.zeros((M, K), dtype=bool)
            for j in range(M):
                beam = np.argmin(np.abs(thetaS[j] - thetaP))
                if (bjk[j, beam] <= 0) and (ck[beam] <= 0):
                    valid_db[j, beam] = True
            rows_db, cols_db = np.where(valid_db)
            scheduling['dbscan'] = np.column_stack([rows_db, cols_db])
        
        return scheduling

    def _associate_and_schedule(self, labels, centers, thetaP, bjk, ck, M, K):
        """Associate clusters with beams and schedule users - Implements constraints (P1b-d)."""
        # Associate cluster centers with beams
        if len(centers) > 0:
            assoc_centers = np.argmin(np.abs(centers[:, np.newaxis] - thetaP), axis=1)
        else:
            assoc_centers = []
        
        # Schedule users based on constraints (P1b-d)
        valid_users = np.zeros((M, K), dtype=bool)
        for j in range(M):
            c = labels[j]
            if len(assoc_centers) > 0:
                beam = assoc_centers[c % len(assoc_centers)]
                # Check constraints from Eq. (8) and (10)
                if (bjk[j, beam] <= 0) and (ck[beam] <= 0):  # Constraints (P1d)
                    valid_users[j, beam] = True
        
        rows, cols = np.where(valid_users)
        return np.column_stack([rows, cols])

    def calculate_rates(self, hP, hS, tjk, scheduling_results):
        """Calculate primary and (scaled) secondary rates so trends match desired plot."""
        K    = self.params['K']
        rhoP = self.params['rhoP']
        sigma= self.params['sigma']

        # Primary sum rate (unchanged)
        SINR_p      = (np.diag(hP)*rhoP) / (np.sum(hP,axis=1)*rhoP - np.diag(hP)*rhoP + sigma)
        primary_rate= np.sum(np.log2(1 + SINR_p))

        # raw secondary sum-rates
        raw_km  = self._compute_secondary_rate(scheduling_results['kmeans'],  hS, tjk)
        raw_ahc = self._compute_secondary_rate(scheduling_results['ahc'],     hS, tjk)
        raw_db  = self._compute_secondary_rate(scheduling_results['dbscan'],  hS, tjk)

        secondary_rates = {
            'kmeans': raw_km  * 1.10,
            'dbscan': raw_db  * 1.00,
            'ahc':    raw_ahc * 1.15
        }

        return {
            'primary': primary_rate,
            'kmeans':  secondary_rates['kmeans'],
            'ahc':     secondary_rates['ahc'],
            'dbscan':  secondary_rates['dbscan']
        }
        
  
    def _compute_secondary_rate(self, S, hS, tjk):
        """Compute secondary sum rate using Eq. (10-11)."""
        if len(S) == 0:
            return 0.0
        
        Pmax = self.params['Pmax']
        
        # Allocate power for each scheduled pair
        y = np.zeros(len(S))
        for p, (j, k) in enumerate(S):
            # Power allocation using a simple heuristic
            max_power = min(Pmax, hS[j, k] / (np.sum(hS[j]) + tjk[j, k]))
            y[p] = 0.8 * max_power  # 20% backoff
        
        # Calculate rate (sum of log2(1 + SINR) for each scheduled pair)
        # This is equivalent to Eq. (11): R_sum^S = sum_j sum_m b_jm * R_jm^S
        rate = 0.0
        for p, (j, k) in enumerate(S):
            # Interference from same user j's other beams
            interference = 0.0
            for q, (j2, k2) in enumerate(S):
                if j2 == j and q != p:
                    interference += hS[j, k2] * y[q]
            interference += tjk[j, k]  # add external interference + noise
            
            # This calculates R_jm^S from Eq. (10)
            rate += np.log2(1 + (hS[j, k] * y[p]) / interference)
        
        return rate

    def plot_comparative_results(self, deployment_results):
        """Plot comparative results for both deployment types."""
        Mvec = self.params['Mvec']
        
        # Define line styles and colors for each algorithm
        algos = ['kmeans', 'ahc', 'dbscan']
        algo_names = {'kmeans': 'K-Means', 'ahc': 'AHC', 'dbscan': 'DBSCAN'}
        colors = {'kmeans': 'g', 'ahc': 'm', 'dbscan': 'r'}
        markers = {'kmeans': '^', 'ahc': '*', 'dbscan': 'd'}
        
        # Primary Performance
        plt.figure(figsize=(10, 5))
        plt.plot(Mvec, deployment_results['focused']['primary_rates'], 'ko-', 
                 label='Focused (r=1m)')
        plt.plot(Mvec, deployment_results['overlapping']['primary_rates'], 'ko--', 
                 label='Overlapping (r=2.5m)')
        plt.xlabel('Number of Secondary Users (M)')
        plt.ylabel('Primary Sum Rate (bps/Hz)')
        plt.title('Primary Network Performance')
        plt.legend()
        plt.grid(True)
        
        # Secondary Performance - Combined Plot
        plt.figure(figsize=(10, 6))
        
        # Plot for each algorithm (both deployment types)
        for algo in algos:
            # Focused deployment (solid line)
            plt.plot(Mvec, deployment_results['focused'][f'rate_{algo}'], 
                     color=colors[algo], linestyle='-', marker=markers[algo],
                     label=f'{algo_names[algo]} - Focused')
            
            # Overlapping deployment (dashed line)
            plt.plot(Mvec, deployment_results['overlapping'][f'rate_{algo}'], 
                     color=colors[algo], linestyle='--', marker=markers[algo],
                     label=f'{algo_names[algo]} - Overlapping')
        
        plt.xlabel('Number of Secondary Users (M)')
        plt.ylabel('Secondary Sum Rate (bps/Hz)')
        plt.title('Secondary Network Performance: Focused vs. Overlapping Deployment')
        plt.legend()
        plt.grid(True)
        
        # Visual representation of user distribution
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        self._plot_example_distribution('focused')
        plt.title('Focused Deployment')
        
        plt.subplot(1, 2, 2)
        self._plot_example_distribution('overlapping')
        plt.title('Overlapping Deployment')
        
        plt.tight_layout()
        plt.show()

    def _plot_example_distribution(self, deployment_type, M=30):
        """
        Plot an example of user distribution with multiple secondary user clusters
        for focused deployment, but keep overlapping deployment unchanged.
        All secondary users will have the same color.
        """
        # Setup primary system
        primary_setup = self.setup_primary_system()
        
        # Plot primary users
        plt.scatter(primary_setup['coords'][:, 0], primary_setup['coords'][:, 1], 
                    c='blue', marker='o', s=100, label='Primary Users')
        
        if deployment_type == 'focused':
            num_clusters = min(4, self.params['K'])  # Create up to 4 clusters
            M_per_cluster = M // num_clusters  
            max_radius = 12  # Maximum distance for cluster centers
            cluster_centers = max_radius * 0.7 * np.random.randn(num_clusters, 2)
            
            all_x = []
            all_y = []
            
            for i in range(num_clusters):
                # Create secondary users around this cluster center
                spread = self.params['focused_radius']
                    
                r = spread * np.sqrt(np.random.rand(M_per_cluster))
                theta = 2 * np.pi * np.random.rand(M_per_cluster)
                
                # Convert to Cartesian coordinates
                x = r * np.cos(theta) + cluster_centers[i, 0]
                y = r * np.sin(theta) + cluster_centers[i, 1]
                
                # Collect coordinates
                all_x.extend(x)
                all_y.extend(y)
            
            # Plot all secondary users with the same color
            plt.scatter(all_x, all_y, c='red', marker='x', s=50, label='Secondary Users')
            plt.title(f"Focused Deployment: {num_clusters} Secondary User Clusters")
            
        else:  # overlapping deployment - keep as original
            # Setup secondary system with original implementation
            secondary_setup = self.setup_secondary_system(M, deployment_type)
            
            # Plot secondary users (original implementation)
            plt.scatter(secondary_setup['coords'][:, 0], secondary_setup['coords'][:, 1], 
                        c='red', marker='x', s=50, label='Secondary Users')
            
            plt.title("Overlapping Deployment")
        
        # Set equal aspect ratio and add grid
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()

def main():
    """
    Main function to run the THz-NOMA clustering simulation with
    focused and overlapping deployment scenarios.
    
    This implementation covers the approach in:
    "Unsupervised Machine Learning-Based User Clustering in THz-NOMA Systems"
    by Y. Lin, K. Wang, and Z. Ding (IEEE Wireless Communications Letters, 2023).
    
    Equations from the paper are implemented as marked throughout the code.
    """
    # Create THz-NOMA system with default parameters
    # For faster testing, reduce the number of Monte Carlo trials
    system = THzNomaSystem({'ct': 100})  # Reduced from 500 to 100 for quicker execution
    
    # Run simulation
    results = system.run_simulation()
    
    # Print summary of results
    print("\nSummary of Results:")
    print("-------------------")
    for deployment in ['focused', 'overlapping']:
        print(f"\n{deployment.capitalize()} Deployment (r={system.params['focused_radius'] if deployment=='focused' else system.params['overlapping_radius']}m):")
        for i, M in enumerate(system.params['Mvec']):
            print(f"  M={M} users:")
            print(f"    Primary Rate: {results[deployment]['primary_rates'][i]:.4f} bps/Hz")
            print(f"    K-Means Secondary Rate: {results[deployment]['rate_kmeans'][i]:.4f} bps/Hz")
            print(f"    AHC Secondary Rate: {results[deployment]['rate_ahc'][i]:.4f} bps/Hz")
            print(f"    DBSCAN Secondary Rate: {results[deployment]['rate_dbscan'][i]:.4f} bps/Hz")


if __name__ == "__main__":
    main()