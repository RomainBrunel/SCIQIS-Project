# SCIQIS-Project
From SCIQUIS courses

# 4D Cluster state evolution simulation
    - Description of gaussian state and gaussian operation and generalisation to multipartite entangled state,
        - Creation of class multipartite state,
        - Visualisation of specific modes from multipartite mode and cov,
    - Computation of the multimode state for different parameters of the loop lines n,m,k,
        - Implementation in spatial domain,
        - Implementation for octorail,
        - Implementation for dual rail,
    - Implementation of gaussian measurement on selected modes and state identification on resulting mode,
    - Implementation of non gaussian measurement on selected modes,
    - Numerical optimisation to search for specific state after gaussian + non gaussian measurement.

# Install requirement
```cmd
py -m pip install -r requirement.txt
```

# Running a calculation
```python
import multipartite_state as mss
cs = mss.cluster_state(spatial_depth:int,   # Depth of the cluster state (number of macronode in the cluster)
                       n:int,               # Size of the first dimension
                       m:int,               # Size of the second dimension
                       k:int,               # Size of the third dimension
                       structure:str        # Structure of the cluster state (dual rail or octo rail)  if dual rail k=1 simulate a 3D cluster and m=1 k=1 a 2D cluster
                       )
cs.run_calculation(r:float,                  # Squeezing parameter
                  gif:bool,                 # If true save a gif of the covariance evolution of the cluster state
                  save:bool                 # If true save a hdf5 file with cov and mean matrix 
                  )

cs.state.plot_covariance_matrix()           # Plot the covariance matrix
cd.state.plot_mean_matrix()                 # Plot the dispalcement vector
```
# Open a calculation
```python
import multipartite_state as mss
cs = mss.cluster_state()
cs.open_calculation(filename:str            # Filename of the previously saved calculation
                    )

cs.state.plot_covariance_matrix()           # Plot the covariance matrix
cd.state.plot_mean_matrix()                 # Plot the dispalcement vector
```
# Measurement of modes
## Gaussian measurement
After running the simulation the user can proceed to the measurement of its desired modes. The function can also plot the wigner function if 1 mode is remaining.
```python 
MU, COV, M = cs.measurement_gaussian(modes:np.ndarray,   # List of modes to be measured 
                                     thetas:np.ndarray,  # List of angles to measured the modes
                                     plot:bool           # True if users want to plot the wigner function of the result state
                                     )
```
MU, COV and M are respectively the updated mean vector, covariance matrix and outcome of measurement.
## Show one of the mode
Plot the wigner function of a specific mode while tracing out all the others
```python
MU, COV, M = cs.plot_wigner(mode:int,                    # Mode to plot the wigner function
                            angle:np.ndarray             # Angles of the macronodes for the measurement (repeated identicaly for each macronodes)
                            )
```
MU, COV and M are respectively the updated mean vector, covariance matrix and outcome of measurement.