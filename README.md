

# Unsupervised Machine Learning-Based User Clustering in THz-NOMA Systems

This repository implements **three unsupervised machine learning algorithms**—K-Means, Agglomerative Hierarchical Clustering (AHC), and DBSCAN—for **user clustering** in Terahertz (THz) Non-Orthogonal Multiple Access (NOMA) systems. The aim is to optimize user clustering, improve system capacity, and enhance energy efficiency in next-generation wireless communication networks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Code Structure](#code-structure)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Clustering Methods](#clustering-methods)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

THz-NOMA systems, as a part of next-generation wireless networks, face challenges in user clustering due to high-frequency signal constraints. This repository provides a Monte Carlo-based simulation framework for user clustering using **unsupervised machine learning** techniques. It evaluates the performance of these clustering techniques in THz-NOMA systems under various conditions.

---

## Features

- **Three Clustering Algorithms**:
  - **K-Means**: Requires a predefined number of clusters.
  - **Agglomerative Hierarchical Clustering (AHC)**: Clusters users based on Ward's linkage.
  - **DBSCAN**: Density-based clustering with automatic cluster count detection.
- **Monte Carlo Simulations**: Randomizes user topologies and calculates average results over multiple trials.
- **Performance Analysis**: Evaluates primary and secondary network performance in terms of sum rates.

---

## Code Structure

1. **Imports and Parameters**:
   - Libraries: `numpy`, `matplotlib`, `sklearn`, `scipy.stats`.
   - System Parameters: Defines constants like the number of antennas (`N`), primary users (`K`), transmit power (`rhoP`), and more.

2. **Main Function (`main()`)**:
   - Iterates over different numbers of secondary users (`M`).
   - Performs Monte Carlo simulations to compute data rates for primary and secondary users.
   - Outputs performance plots.

3. **Key Functions**:
   - `compute_secondary_rate(S, hS, tjk)`: Calculates secondary user rates based on channel conditions and interference constraints.

Refer to the [README file](https://github.com/Transyltooniaa/Unsupervised-Machine-Learning-Based-User-Clustering-in-THz-NOMA-Systems/blob/61d3249b850f969a54b1f38378be945c375aac73/README.md) for a detailed walkthrough.

---

## Technology Stack

| Language          | Purpose                                   |
|-------------------|------------------------------------------|
| **Python**         | Core algorithms and clustering methods. |
| **Jupyter Notebook** | Simulation and performance analysis.    |
| **MATLAB**         | Optional for extended modeling.          |

---

## Setup

### Prerequisites

1. **Python Environment**: Install Python 3.8+.
2. **Dependencies**: Install required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. **MATLAB** (Optional): For running MATLAB scripts.

### Clone the Repository

```bash
git clone https://github.com/Transyltooniaa/Unsupervised-Machine-Learning-Based-User-Clustering-in-THz-NOMA-Systems.git
cd Unsupervised-Machine-Learning-Based-User-Clustering-in-THz-NOMA-Systems
```

---

## Usage

### Running Simulations

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the simulation notebooks in the `notebooks/` directory and execute the cells.

### Running Python Scripts

1. Execute the main script:
   ```bash
   python <filename>.py
   ```

---

## Clustering Methods

1. **K-Means**:
   - Groups secondary users into `K` clusters based on their AoDs (Angles of Departure).
   - Uses the **circular mean** to compute cluster centers.

2. **Agglomerative Hierarchical Clustering (AHC)**:
   - Groups users in a bottom-up fashion.
   - Computes cluster centers using Ward's linkage and circular angle calculations.

3. **DBSCAN**:
   - Automatically detects the number of clusters based on density.
   - Handles noise points by treating them as single-user clusters.

---

## Results

### Simulation Outputs

1. **Primary Network Performance**:
   - Sum rate vs. the number of secondary users (`M`).

2. **Secondary Network Performance**:
   - Sum rate for each clustering method (K-Means, AHC, DBSCAN) vs. `M`.

### Key Insights

- **Angle-Based Clustering**: Groups secondary users with similar AoDs to share the same beam.
- **Monte Carlo Approach**: Randomizes user topologies to produce generalized results.

---

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**.
2. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. **Commit your changes**:
   ```bash
   git commit -m "Add feature-name"
   ```
4. **Push to your branch**:
   ```bash
   git push origin feature-name
   ```
5. **Open a pull request**.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details. 

---

Let me know if you'd like me to refine this further or add additional details!


Below is a **step-by-step** explanation of how the code works. The code implements **three unsupervised clustering algorithms**—**K-Means**, **Agglomerative Hierarchical Clustering (AHC)**, and **DBSCAN**—to group secondary users by their angles (AoDs) and then schedules those users on beams while respecting interference constraints for both primary and secondary users.

---

## 1. Overall Code Structure

1. **Imports and Parameters**:  
   - Imports: `numpy`, `matplotlib`, `sklearn` clustering modules, and `scipy.stats.circmean`.  
   - System Parameters: Constants such as the number of antennas (`N`), number of primary users (`K`), transmit power (`rhoP`), path loss exponent (`alpha`), frequency (`fc`), molecular absorption coefficient (`rl`), etc.  
   - `Mvec` is a list of different numbers of **secondary** users to test (e.g., `[1, 2, 4, 6, 8]`).

2. **Main Function** (`main()`):  
   - Loops over each `M` in `Mvec`.  
   - For each `M`, runs a Monte Carlo simulation (`ct` trials).  
   - In each trial, sets up **primary** and **secondary** systems, **clusters** the secondary users with three algorithms, **schedules** feasible user–beam pairs, and **computes** data rates.  
   - Averages results across all trials, then **plots** final curves for:
     - Primary Network Performance vs. `M`.
     - Secondary Network Performance vs. `M` for each clustering method.

3. **`compute_secondary_rate(S, hS, tjk)`**:  
   - Given a set of scheduled pairs \( (j,k) \) and the channel/interference arrays, allocates power and computes the sum rate of the secondary users.

---

## 2. Detailed Walkthrough

### 2.1. Monte Carlo Setup

Inside `main()`:

- **`primary_rates`, `rate_kmeans`, `rate_ahc`, `rate_dbscan`** are arrays to store the *average rates* (across `ct` trials) for each scenario:
  - `primary_rates[iM]`: average primary sum rate for a given `M`.
  - `rate_kmeans[iM]`, `rate_ahc[iM]`, `rate_dbscan[iM]`: average secondary sum rate under K-Means, AHC, and DBSCAN clustering, respectively.

```python
primary_rates = np.zeros(len(Mvec))
rate_kmeans  = np.zeros(len(Mvec))
rate_ahc     = np.zeros(len(Mvec))
rate_dbscan  = np.zeros(len(Mvec))
```

For each `M` in `Mvec`:
```python
for iM, M in enumerate(Mvec):
    ...
    prim_rate_trials = np.zeros(ct)
    sec_rate_km      = np.zeros(ct)
    sec_rate_ahc     = np.zeros(ct)
    sec_rate_dbscan  = np.zeros(ct)
    
    for ict in range(ct):
        ...
```
- We run `ct` independent trials. Each trial sets up a **new** random topology (user coordinates, path losses, angles).

---

### 2.2. Primary System Setup

1. **Primary User Coordinates**:  
   ```python
   coordsP = rP * np.random.randn(K, 2)
   rkP = np.linalg.norm(coordsP, axis=1)
   ```
   - Generates `K` primary users around the BS with some random distribution (Gaussian, scaled by `rP=10`).

2. **Path Loss Calculation** (`PLkP`):  
   ```python
   PLkP = (c0/(4*np.pi*fc))**2 * np.exp(-rl*rkP) / (1 + rkP**alpha)
   ```
   - Incorporates free-space path loss, molecular absorption, and distance-based attenuation.

3. **Primary User AoDs** (`thetaP`):  
   ```python
   thetaP = -np.pi/2 + np.pi/K * np.arange(1, K+1)
   ```
   - For demonstration, these are spaced equally between \(-\pi/2\) and \(\pi/2\). (You could also randomize them.)

4. **Array Response Matrix** (`H`):  
   ```python
   NN = np.arange(N)[:, np.newaxis]   # shape (N, 1)
   H = np.exp(-1j * np.pi * np.sin(thetaP) * NN) * np.sqrt(PLkP)
   ```
   - Models how each user’s channel vector depends on AoD.

5. **Analog Beamforming**:  
   ```python
   NQ = 10
   theta_vec = np.linspace(-np.pi/2, np.pi/2, NQ)
   ABF = np.zeros((N, K), dtype=complex)
   for kidx in range(K):
       idx = np.argmin(np.abs(thetaP[kidx] - theta_vec))
       ABF[:, kidx] = np.exp(-1j * np.pi * np.sin(theta_vec[idx]) * NN).flatten()
       ABF[:, kidx] /= np.sqrt(N)
   ```
   - We pick from a discrete set of `NQ=10` possible steering angles (`theta_vec`) to approximate the best match for each primary user’s AoD.  
   - Each beam is normalized by \(\sqrt{N}\).

---

### 2.3. Secondary System Setup

1. **Secondary User Coordinates**:
   ```python
   coordsS = rS * np.random.randn(M, 2)
   rkS = np.linalg.norm(coordsS, axis=1)
   ```
   - Similar to primary users, but scaled by `rS=15`.

2. **Secondary Path Loss** (`PLkS`):
   ```python
   PLkS = (c0/(4*np.pi*fc))**2 * np.exp(-rl*rkS) / (1 + rkS**alpha)
   ```

3. **Secondary User AoDs** (`thetaS`):
   ```python
   thetaS = np.pi * np.random.rand(M) - np.pi/2
   ```
   - Each user has an AoD in \([- \pi/2, \pi/2]\).

4. **Secondary Channel Matrix** (`G`):
   ```python
   G = np.exp(-1j * np.pi * np.sin(thetaS)[:, np.newaxis] * NN.T) \
       * np.sqrt(PLkS[:, np.newaxis])
   ```
   - Similar to `H`, but for the secondary user angles and path losses.

---

### 2.4. Clustering of Secondary Users

We **cluster** the secondary users based on **angle**. To do that, we embed each user’s AoD \(\theta_j\) on the **unit circle** \(\bigl(\cos(\theta_j), \sin(\theta_j)\bigr)\):

```python
X_km = np.column_stack([np.cos(thetaS), np.sin(thetaS)])
```

#### 2.4.1. K-Means

```python
if M >= K:
    km = KMeans(n_clusters=K, n_init=10).fit(X_km)
    km_labels = km.labels_
    theta_km = np.arctan2(km.cluster_centers_[:,1], km.cluster_centers_[:,0])
else:
    # If not enough users for K clusters, each user is its own cluster
    km_labels = np.arange(M)
    theta_km = thetaS.copy()
```

- If there are at least as many users as clusters (`M >= K`), run standard K-Means with `K` clusters.  
- Otherwise, we skip K-Means logic and just assign each user to its own cluster.

#### 2.4.2. Agglomerative Hierarchical Clustering (AHC)

```python
if M >= 2:
    num_clusters = min(K, M)
    ahc = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(X_km)
    ahc_labels = ahc.labels_
    # cluster center angles
    theta_ahc = np.array([
        circmean(thetaS[ahc_labels == cl]) for cl in range(num_clusters)
    ])
else:
    ahc_labels = np.zeros(M, dtype=int)
    theta_ahc  = thetaS.copy()
```
- Uses **Ward’s** linkage.  
- If `M < 2`, trivially everything is in one cluster.  
- Otherwise, form `min(K, M)` clusters, then compute each cluster’s “center angle” using a circular mean of the AoDs in that cluster.

#### 2.4.3. DBSCAN

```python
if M >= 2:
    db = DBSCAN(eps=0.4, min_samples=2).fit(X_km)
    db_labels = db.labels_
    
    # Build cluster centers
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
    db_labels_eff = np.zeros(M, dtype=int)
    theta_db = thetaS.copy()
```

- DBSCAN automatically decides how many clusters to form, based on density.  
- `eps=0.4` and `min_samples=2` is a basic example.  
- Users labeled `-1` are considered **noise** and become their own single-user cluster.  
- We compute each cluster’s center angle using a circular mean of the AoDs in that cluster.

---

### 2.5. Beam Association

Once the clusters are formed, each cluster is assigned to a **beam** (among the existing `K` beams for the primary users). We pick whichever beam in `thetaP` is **closest** in angle to the cluster center:

- **K-Means**: `km_assoc = np.argmin(np.abs(theta_km[:, np.newaxis] - thetaP), axis=1)`
  - For each cluster center angle in `theta_km`, find the best matching beam index.  
- **AHC**: `ahc_assoc = np.argmin(np.abs(theta_ahc[:, np.newaxis] - thetaP), axis=1)`
- **DBSCAN**: `db_assoc_centers = np.argmin(np.abs(theta_db[:, np.newaxis] - thetaP), axis=1)`

---

### 2.6. Channel Gains and Constraints

#### 2.6.1. Channel Gains

```python
hP = np.abs(H.conj().T @ ABF)**2  # shape: (K, K)
hS = np.abs(G @ ABF)**2           # shape: (M, K)
```
- `hP[k, i]` is the gain from beam `i` to primary user `k`.  
- `hS[j, i]` is the gain from beam `i` to secondary user `j`.

#### 2.6.2. Primary Constraints

```python
ck = np.zeros(K)
for kidx in range(K):
    I_p = np.sum(hP[kidx]) - hP[kidx, kidx]
    ck[kidx] = I_p/hP[kidx, kidx] - rhoP/(2**Rk - 1) + sigma/hP[kidx, kidx]
```

- For each primary user `kidx`, compute how much interference it can tolerate before violating the QoS threshold \(R_k\).  
- This `ck[kidx]` will be used to check if secondary user transmissions on that beam are feasible.

#### 2.6.3. Secondary Constraints

```python
bjk = np.zeros((M, K))
tjk = np.zeros((M, K))
for j in range(M):
    for beam_idx in range(K):
        I_s = np.sum(hS[j]) - hS[j, beam_idx]
        bjk[j, beam_idx] = ...
        tjk[j, beam_idx] = ...
```

- For each secondary user `j` and beam `beam_idx`, compute:
  - `bjk[j, beam_idx]`: checks if scheduling that user–beam pair violates the primary constraints.  
  - `tjk[j, beam_idx]`: partial interference plus noise for that user–beam pair.

---

### 2.7. Scheduling

After we have cluster labels and beam associations, we see which pairs \((j,k)\) are valid (i.e., do not violate constraints).

#### K-Means Scheduling

```python
valid_km = np.zeros((M, K), dtype=bool)
for j in range(M):
    c = km_labels[j]
    if len(km_assoc) > 0:
        beam = km_assoc[c % len(km_assoc)]
        if (bjk[j, beam] <= 0) and (ck[beam] <= 0):
            valid_km[j, beam] = True

rows_km, cols_km = np.where(valid_km)
S_km = np.column_stack([rows_km, cols_km])
```

- Each user `j` belongs to cluster `c`.  
- We map that cluster `c` to the beam indicated by `km_assoc[c]`.  
- Check if that user–beam pair is feasible (both `bjk` and `ck` are within constraints).  
- If yes, mark it as valid. The resulting array `S_km` contains all scheduled \((j, beam)\) pairs.

#### AHC Scheduling

Similar logic with `ahc_labels` and `ahc_assoc`.

#### DBSCAN Scheduling

We have to handle the “noise” clusters in `db_labels_eff`, but the logic is the same—map each user to the beam of its cluster center and check feasibility.

---

### 2.8. Rate Calculations

#### 2.8.1. Primary Rate

```python
SINR_p = (np.diag(hP)*rhoP) / (np.sum(hP, axis=1)*rhoP - np.diag(hP)*rhoP + sigma)
prim_rate_trials[ict] = np.sum(np.log2(1 + SINR_p))
```
- Basic formula for the primary sum rate, using diagonal entries of `hP` as the desired beam gain.

#### 2.8.2. Secondary Rate

We compute the secondary sum rate for each clustering method with the function `compute_secondary_rate(...)`:

```python
def compute_secondary_rate(S, hS, tjk):
    if len(S) == 0:
        return 0.0
    
    # 1) Power allocation
    y = np.zeros(len(S))
    for p, (j, k) in enumerate(S):
        max_power = min(Pmax, hS[j, k] / (np.sum(hS[j]) + tjk[j, k]))
        y[p] = 0.8 * max_power  # 20% backoff
    
    # 2) Calculate sum of log2(1 + SINR)
    rate = 0.0
    for p, (j, k) in enumerate(S):
        # Interference from same user j’s other beams
        interference = 0.0
        for q, (j2, k2) in enumerate(S):
            if j2 == j and q != p:
                interference += hS[j, k2] * y[q]
        interference += tjk[j, k]  # add external interference + noise
        rate += np.log2(1 + (hS[j, k] * y[p]) / interference)
    
    return rate
```

**Key Steps**:
1. For each scheduled pair \((j,k)\), compute a simple “max feasible power” based on `hS[j,k] / (sum(hS[j]) + tjk[j,k])`.  
2. Apply a “practical backoff” of 0.8 to avoid saturating the maximum.  
3. The **interference** for a pair is the sum of powers from the *other* beams of the *same user* plus the external interference term `tjk[j,k]`.  
4. Sum the log2(1 + SINR) over all pairs in `S`.

---

### 2.9. Final Averages and Plots

After all `ct` trials:

```python
primary_rates[iM] = np.mean(prim_rate_trials)
rate_kmeans[iM]   = np.mean(sec_rate_km)
rate_ahc[iM]      = np.mean(sec_rate_ahc)
rate_dbscan[iM]   = np.mean(sec_rate_dbscan)
```

Finally, we plot two figures:

1. **Primary Network Performance** vs. `M`.
2. **Secondary Network Performance** (three lines: K-Means, AHC, DBSCAN) vs. `M`.

---

## 3. Key Takeaways

1. **Angle-Based Clustering**:  
   The core idea is that secondary users with **similar AoDs** can share the same beam. Each clustering algorithm attempts to group users by their angle.

2. **Multiple Clustering Methods**:  
   - **K-Means** requires a known number of clusters \(K\).  
   - **AHC** (Ward’s method) merges clusters in a bottom-up fashion. We fix `n_clusters = min(K,M)` in code.  
   - **DBSCAN** automatically determines cluster counts based on data density (`eps`, `min_samples`), but we give it a fixed `eps=0.4` for demonstration.

3. **Scheduling**:  
   Each cluster is mapped to **one** beam, so all users in that cluster share the same beam. We check if scheduling that user–beam pair meets interference constraints.

4. **Rate Computation**:  
   - The **primary** sum rate is computed by standard SINR for each primary user.  
   - The **secondary** sum rate is computed by enumerating all feasible user–beam pairs \((j,k)\) and summing log2(1 + SINR).

5. **Monte Carlo**:  
   Because each scenario is randomized (random user coordinates, angles, path losses), we repeat many times and **average** to get smoother results.

---

### In Conclusion

- **Primary** users have fixed beams (one beam per primary user).  
- **Secondary** users are grouped by **unsupervised** angle clustering (K-Means, AHC, DBSCAN).  
- Each cluster is associated to a primary user’s beam if constraints allow.  
- The code *then* calculates the total rate for both primary and secondary networks under those cluster-based allocations.  

This is precisely in line with the *“Unsupervised Machine Learning-Based User Clustering in THz-NOMA Systems”* framework, showing how different clustering algorithms can be integrated and compared in a THz-NOMA scenario.
