import numpy as np

class multipartite_state():
    def __init__(self, number_of_modes:int, hbar = 1) -> None:
        """ Generation of dispacement vector and covariance matrix of the system
            State is initialise as independant vaccum states.
        Args: 
            - number_of_modes: int dimension of the multipartite state"""
        self.mu = np.zeros(2*number_of_modes)
        self.cov = np.eye(2*number_of_modes)*hbar/2
    
    def plot_cov_matrix(self):
        pass

    def plot_mean_matrix(self):
        pass

class cluster_state():
    """ Define the interferometer that will be the source of the generation of the 4D cluster state and perform the computations.
    The cluster state is defined by X rails that interract with each other. Some of the rails are delayed in time, this crates the 3 other dimentionality of the cluster.
    Modes have to interact with other modes in the past and the future. This is handle by indexing and the creation of "past" modes as vaccum states.
    The first I = spatial_dim * N are used as past modes and stays as vaccum.
    The first squeezed modes inserted in the interferometer are i, i+1, i+2, i+3, i+4, i+5, i+6 and i+7.
    Below is detailled how modes i to i+7 are interracting with other modes. 

    i   interact with { i+7 ;  i+1 + 8nm ;  i+2 + 8(nm-1) ; i+4 + 8(nm-nmk) }
    i+1 interact with { i+2 ;   i - 8nm  ;      i+3       ;       i+5       }
    i+2 interact with { i+1 ;   i+3 + 8  ;   i + 8(1-nm)  ;   i+6 + 8(1-n)  }
    i+3 interact with { i+4 ;   i+2 - 8  ;      i+1       ;       i+7       }
    i+4 interact with { i+3 ; i+5 + 8nmk ; i+6 + 8(nmk-n) ;  i + 8(nmk-nm)  }
    i+5 interact with { i+6 ; i+4 - 8nmk ;      i+7       ;       i+1       }
    i+6 interact with { i+5 ;  i+7 + 8n  ; i+4 + 8(n-nmk) ;   i+2 + 8(n-1)  }
    i+7 interact with {  i  ;  i+6 - 8n  ;      i+5       ;       i+3       }

    To facilitate the implementation I reduce the previous interraction graph to image only the interaction of the ith mode with the future modes

    i   interact with {    i+7    ;  i+1 + 8nm ;  i+2 + 8(nm-1) ;    past mode    }
    i+1 interact with {    i+2    ; past mode  ;      i+3       ;       i+5       }
    i+2 interact with { past mode ;   i+3 + 8  ;   past mode    ;    past mode    }
    i+3 interact with {    i+4    ; past mode  ;   past mode    ;       i+7       }
    i+4 interact with { past mode ; i+5 + 8nmk ; i+6 + 8(nmk-n) ;  i + 8(nmk-nm)  }
    i+5 interact with {    i+6    ; past mode  ;      i+7       ;    past mode    }
    i+6 interact with { past mode ;  i+7 + 8n  ;   past mode    ;   i+2 + 8(n-1)  }
    i+7 interact with { past mode ; past mode  ;   past mode    ;    past mode    }

    By encoding only the BS only with the earliest modes the modes i+7 is already defined by the previous 7 others. Which makes sens.
    
    Args:   
        - spatial_depth: depth of the spatial interferometer in the spatial dimension
        - n, m, k: int number of modes in the different loops. number_of_modes = n * m * k
        - structure: str {octo, dual} structure of the interferometer """
    def __init__(self, spatial_depth:int = 2, n:int = 2, m:int=2, k:int=2, structure: str = "octo") -> None:
        if structure == "octo":
            self.macronode_size = 8
        elif structure == "dual":
            self.macronode_size = 2

        self.spatial_depth = spatial_depth
        self.N = n*m*k 
        self.n = n
        self.m = m
        self.k = k
        self.state = multipartite_state(number_of_modes = spatial_depth*self.macronode_size*self.N)
        self.generate_BS_indice_array()

    def generate_BS_indice_array(self):
        """ Function that generate a list referencing all the interaction between all the modes for all column of beamsplitters in the set-up.
        
        Update:
            - self.BS_indices: list of 2D numpy arrays containing the all modes 'A' in the first dimension and the respective modes 'B' in the second dimension."""
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        if ms == 8:
            interaction_matrix = np.array([ [    0+7    ,  0+1 + 8*n*m ,  0+2 + 8*(n*m-1) ,       np.nan      ],
                                            [    0+2    ,    np.nan    ,      0+3       ,       0+5       ],
                                            [    np.nan   ,   0+3 + 8  ,      np.nan      ,       np.nan      ],
                                            [    0+4    ,    np.nan    ,      np.nan      ,       0+7       ],
                                            [    np.nan   , 0+5 + 8*n*m*k , 0+6 + 8*(n*m*k-n) ,  0 + 8*(n*m*k-n*m)  ],
                                            [    0+6    ,    np.nan    ,      0+7       ,       np.nan      ],
                                            [    np.nan   ,  0+7 + 8*n  ,      np.nan      ,   0+2 + 8*(n-1)  ],
                                            [    np.nan   ,    np.nan    ,      np.nan      ,       np.nan      ]])
            
        i_indices = np.arange(8)
        # Mask to identify non-nan values in the interaction matrix
        mask = ~np.isnan(interaction_matrix)
        # Broadcasting the i_indices to match the shape of the interaction_m
        i_to_i7 = np.broadcast_to(i_indices[:, np.newaxis], interaction_matrix.shape)
        # Apply the mask to get the valid indices and interaction values
        valid_i_to_i7 = [i_to_i7[mask[:, col], col] for col in range(interaction_matrix.shape[1])]
        valid_interactions = [interaction_matrix[mask[:, col], col] for col in range(interaction_matrix.shape[1])]

        self.BS_indices = [self.generate_arrays_from_pairs(valid_i_to_i7[col],valid_interactions[col],ms,N*ms*depth) for col in range(len(valid_i_to_i7))]

    def generate_arrays_from_pairs(self,A, B, step, max_value):
        """ Myscalenious function used to convert a list of starting point into array until max_value with step step.
        
        Args:
            - A: list of starting index of the A port of the beamsplitter
            - B: list of starting index of the B port of the beamsplitter
            - step: step between modes in the array (length of the macronode)
            - max_value: last modes on which to apply a beam splitter operation
        
        Return:
            results: 2D numpy array 2D numpy arrays containing the all modes 'A' in the first dimension and the respective modes 'B' in the second dimension."""
        # Convert the input lists to NumPy arrays
        A_array = np.array(A)
        B_array = np.array(B)
        
        # Ensure that A and B have the same length
        if len(A_array) != len(B_array):
            raise ValueError("Arrays A and B must have the same length.")
        
        resultA = []
        resultB = []

        # Process each pair independently
        for a, b in zip(A_array, B_array):
            # Calculate the number of steps for each pair (a, b) until max_value
            num_steps_A = (max_value - a) // step + 1
            num_steps_B = (max_value - b) // step + 1
            
            # Determine the size for the current pair
            min_num_steps = min(num_steps_A, num_steps_B)
            
            # Create the array for each pair
            result_A = a + np.arange(min_num_steps) * step
            result_B = b + np.arange(min_num_steps) * step
            
            # Append the result tuple to the results list
            resultA.append(np.array(result_A))
            resultB.append(np.array(result_B))
        results = np.array([np.concatenate(resultA),np.concatenate(resultB)],dtype=np.int64)
        
        return results


    def apply_symplectic(self, F:np.ndarray, indices:np.ndarray):
        """Function that apply a symplectic transformation onto specific modes of the multipartite state
        
        Args:
            - F: symplectic matrix
            - indices: indices of the mu vector onto wich apply the symplectic 
        
        Update:
            Update the covariance matrix and µ vector of the multipartite state """
        mu = self.state.mu
        cov = self.state.cov
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth

        mu[indices] = F @ mu[indices]

        ar = np.arange(2*N*ms*depth)
        mask = ~np.isin(ar, indices)


        slices = np.ix_(indices,indices)
        slices_B = np.ix_(indices,ar[mask])
        slices_C = np.ix_(ar[mask],indices)
        cov[slices] = F @ cov[slices] @ F.conj().T
        cov[slices_B] = F @ cov[slices_B] 
        cov[slices_C] = cov[slices_C] @ F.conj().T
    
    def BS(self, N:int, modesA:np.ndarray, modesB:np.ndarray):
        """ Create symplectic that apply a beam splitter onto specific modes
        
        Args:
            - N: Number of modes
            - modesA: list[int] modes A on which to apply the BS operation 
            - modesB: list[int] modes B on which to apply the BS operation
        
        Return:
            - F: symplectic matrix 2N x 2N"""
        F = np.eye(2*N)
        print(F.shape)
        A = np.concatenate([modesA,modesB,modesB,modesA+N,modesB+N,modesB+N])
        B = np.concatenate([modesA,modesB, modesA,modesA+N,modesB+N,modesA+N])
        F[A, B] = 1/np.sqrt(2)
        A = np.concatenate([modesA, modesA+N])
        B = np.concatenate([modesB, modesB+N])
        F[A, B] = -1/np.sqrt(2)
        return F

    def P(self, N:int, modes:np.ndarray, theta:float):
        """ Create symplectic that apply a phase shift operation onto specific modes
        
        Args:
            - N: Number of modes
            - modes: list[int] modes on which to apply the BS operation 
            - theta: angle of the rotation matrix, same for all modes
        
        Return:
            - F: symplectic matrix 2N x 2N"""
        F = np.eye(2*N)
        F[modes,modes] = np.cos(theta)
        F[modes,modes+N] = -np.sin(theta)
        F[modes+N,modes] = np.sin(theta)
        F[modes+N,modes+N] = np.cos(theta)
        return F

    def S(self, N:int, modes:np.ndarray, r:float):
        """ Create symplectic that apply a squeezing operation onto specific modes
        
        Args:
            - N: Number of modes
            - modes: list[int] modes on which to apply the BS operation 
            - r: squeezing parameter, same for all modes
        
        Return:
            - F: symplectic matrix 2N x 2N"""
        F = np.eye(2*N)
        F[modes,modes] = np.exp(-r)
        F[modes+N,modes+N] = np.exp(r)
        return F

    def squeeze_initial_state(self, r:float):
        """Introduce squeezing on the initial modes. The first i modes are let vaccum.
         
        Args:
            - r: squeezing parameter, same for all modes

        Update:
            Update the covariance matrix and µ vector of the multipartite state """
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        indice_sqz = np.arange(N*ms,N*ms*depth)
        S = self.S(N = N*ms*(depth-1),
                    modes = indice_sqz - N*ms,
                    r = r)
        indice_XP = np.concatenate([indice_sqz,indice_sqz+N*ms*depth])
        self.apply_symplectic(S, indice_XP)
    
    def rotate_half_state(self, theta:float):
        """Introduce theta phase shift on half of the initial modes. The first i modes are let vaccum.
         
        Args:
            - theta: angle of the rotation matrix, same for all modes
        
        Update:
            Update the covariance matrix and µ vector of the multipartite state"""
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth

        indice_rot = np.arange(N*ms+1,N*ms*depth,2)
        P = self.P(N = N*ms*(depth-1)//2,
                   modes = indice_rot//2 - N*ms,
                   theta = theta)
        
        indice_XP = np.concatenate([indice_rot,indice_rot+N*ms*depth])
        self.apply_symplectic(P, indice_XP)

    def apply_beamsplitter(self, col:int):
        """Introduce BS operation on all modes listed in BS_indices[col].
         
        Args:
            - col: int, colomn of beamsplitter to apply (in the structure of the cluster)
        
        Update:
            Update the covariance matrix and µ vector of the multipartite state"""
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        indices = self.BS_indices[col]
        BS = self.BS(N = 2*len(indices[0]),
                     modesA = np.arange(len(indices[0])),
                     modesB = np.arange(len(indices[0]),2*len(indices[0])))

        print(BS.shape)
        self.apply_symplectic(BS,np.concatenate([np.concatenate(indices),np.concatenate(indices+N*ms*depth)]))


if __name__ == "__main__":
    cs = cluster_state()
    cs.apply_beamsplitter(0) 