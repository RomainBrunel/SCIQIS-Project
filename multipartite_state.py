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

    i   interact with {    i+7    ;  i+1 + 8nm ;  i+2 + (8nm-1) ;    past mode    }
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

    def apply_symplectic(self, F:np.ndarray, indices:np.ndarray, slices:np.ndarray):
        """Function that apply a symplectic transformation onto specific modes of the multipartite state
        
        Args:
            - F: symplectic matrix
            - indices: indices of the mu vector onto wich apply the symplectic 
            - slices: slices of the covariance matrix onto which apply the symplectic
        
        Update:
            Update the covariance matrix and µ vector of the multipartite state """
        mu = self.state.mu
        cov = self.state.cov

        mu[indices] = mu[indices] @ mu
        cov[slices] = F[slices] @ cov @ F[slices].conj().T
    
    def BS(self, N:int, modesA:np.ndarray, modesB:np.ndarray):
        """ Create symplectic that apply a beam splitter onto specific modes
        
        Args:
            - N: Number of modes
            - modesA: list[int] modes A on which to apply the BS operation 
            - modesB: list[int] modes B on which to apply the BS operation
        
        Return:
            - F: symplectic matrix 2N x 2N"""
        F = np.eye(2*N)
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
        F[modes,modes] = np.exp(-2*r)
        F[modes+N,modes+N] = np.exp(2*r)
        return F

    def squeeze_initial_state(self, r:float):
        """Introduce squeezing on the initial modes. The first i modes are let vaccum.
         
        Args:
            - r: squeezing parameter, same for all modes

        Update:
            Update the covariance matrix and µ vector of the multipartite state """
        n, m, k , N, sd, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        mu , cov = self.state.mu, self.state.cov
        indice_sqz = np.arange(N*sd,N*sd*depth)
        S = self.S(N = N*sd*(depth-1),
                    modes = indice_sqz - N*sd,
                    r = r)
        indice_XP = np.concatenate([indice_sqz,indice_sqz+N*sd*depth])
        slices = np.ix_(indice_XP,indice_XP)
        self.apply_symplectic(S, indice_XP, slices)
    
    def rotate_half_state(self):
        """Introduce pi/2 phase shift on half of the initial modes. The first i modes are let vaccum.
         
        Args:
            - theta: angle of the rotation matrix, same for all modes
        
        Update:
            Update the covariance matrix and µ vector of the multipartite state"""
        n, m, k , N, sd, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        mu , cov = self.state.mu, self.state.cov
        theta = np.pi/2

        indice_rot = np.arange(N*sd+1,N*sd*depth,2)
        P = self.P(N = N*sd*(depth-1)//2,
                   modes = indice_rot//2 - N*sd,
                   theta = theta)
        
        indice_XP = np.concatenate([indice_rot,indice_rot+N*sd*depth])
        slices = np.ix_(indice_XP,indice_XP)
        self.apply_symplectic(P, indice_XP, slices)