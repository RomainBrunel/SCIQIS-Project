import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import h5py as h

class multipartite_state():
    def __init__(self, number_of_modes:int, hbar = 1) -> None:
        """ Generation of dispacement vector and covariance matrix of the system
            State is initialise as independant vaccum states.
        Args: 
            - number_of_modes: int dimension of the multipartite state"""
        self.mu = np.zeros(2*number_of_modes)
        self.cov = np.eye(2*number_of_modes)*hbar/2
    
    def plot_cov_matrix(self, ax = None,norm = mpl.colors.Normalize(vmin=-1, vmax=1), save:bool = False):        
        if save: 
            im = ax.imshow(self.cov,cmap="seismic", norm=norm)
            return im
        else:
            fig,ax = plt.subplots()
            fig.set_figheight(15)
            fig.set_figwidth(15)
            ax.imshow(self.cov,cmap= "seismic", norm=norm)
            num_modes = self.cov.shape[0]//2

            # Set custom x-axis labels with numbering
            first_X_index = 0
            first_P_index = num_modes
            middle_X_index = num_modes // 2
            middle_P_index = num_modes + middle_X_index
            last_P_index = 2 * num_modes - 1

            # Set custom x-axis labels at specified positions
            xticks = [first_X_index, first_P_index, middle_X_index, middle_P_index, last_P_index]
            xticklabels = [
            f"modes X{first_X_index + 1}",
            f"modes P{first_P_index - num_modes + 1}",
            f"modes X{middle_X_index + 1}",
            f"modes P{middle_P_index - num_modes + 1}",
            f"modes P{last_P_index - num_modes + 1}"
            ]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
            ax.set_yticks(xticks)
            ax.set_yticklabels(xticklabels, rotation=45, ha="right")
            fig.colorbar(mpl.cm.ScalarMappable(norm= norm, cmap=mpl.colormaps.get_cmap("seismic")),ax = ax)
            
    def plot_mean_matrix(self):
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        fig, ax = plt.subplots()
        ax.imshow(self.mu.reshape(-1,1),aspect=0.01,cmap="seismic",norm = norm)
        num_modes = self.cov.shape[0]//2

        # Set custom x-axis labels with numbering
        first_X_index = 0
        first_P_index = num_modes
        middle_X_index = num_modes // 2
        middle_P_index = num_modes + middle_X_index
        last_P_index = 2 * num_modes - 1

        # Set custom x-axis labels at specified positions
        xticks = [first_X_index, first_P_index, middle_X_index, middle_P_index, last_P_index]
        xticklabels = [
        f"modes X{first_X_index + 1}",
        f"modes P{first_P_index - num_modes + 1}",
        f"modes X{middle_X_index + 1}",
        f"modes P{middle_P_index - num_modes + 1}",
        f"modes P{last_P_index - num_modes + 1}"
        ]
        ax.set_yticks(xticks)
        ax.set_yticklabels(xticklabels, rotation=45, ha="right")
        fig.colorbar(mpl.cm.ScalarMappable(norm= norm, cmap=mpl.colormaps.get_cmap("seismic")),ax = ax)
    

class cluster_state():
    """ Define the interferometer that will be the source of the generation of the 4D cluster state and perform the computations.
    The cluster state is defined by X rails that interract with each other. Some of the rails are delayed in time, this crates the 3 other dimentionality of the cluster.
    Modes have to interact with other modes in the past and the future. This is handle by indexing and the creation of "past" modes as vaccum states.
    The first I = spatial_dim * N are used as past modes and stays as vaccum.

    Args:   
        - spatial_depth: depth of the spatial interferometer in the spatial dimension
        - n, m, k: int number of modes in the different loops
        - structure: str {octo, dual} structure of the interferometer
    
    To simulate 2D dual put m=k=1
    To simulate 3D dual put k=1

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

    The interaction for the dual rail is:
    i   interact with {    i+1    ;  past mode ;    past mode   ;    past mode    ;      past mode     }
    i+1 interact with { past mode ;   i + 2*1  ;   i + 2*(1+n)  ;  i + 2*(1+n+nm) ; i + 2*(1+n+nm+nmk) }
    
     """
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
        self.generate_macronodes()

    def open_calculation(self, filename):
        """ Open a previous calculation from an hdf5 file
        
        Args:
            - filename: name of the file to open"""
        with h.File(filename,'r') as f:
            self.spatial_depth = f.attrs["Depth"]
            self.n = f.attrs["n"]
            self.m = f.attrs["m"]
            self.k = f.attrs["k"]
            self.N = self.n*self.m*self.k 
            self.macronode_size = f.attrs["Macronode_size"]
            self.state.cov = np.array(f["Covariance matrix"][:])
            self.state.mu = np.array(f["Mean vector"][:])
        self.generate_BS_indice_array()
        self.generate_macronodes()

    ###########################################################################
    ############################    SIMULATION    #############################
    ###########################################################################

    def run_calculation(self, r, gif = False, save:bool = False, filename:str = "Cluster_state_simulation.hdf5"):
        """Run the calculation for the structure of the cluster state and save a gif for the evolution of the covariance matrix of the state.
        
        Args: 
            - r: squeezing parameter of the input states
            - gif: True create a gif of the covariance matrix evolution during the creation of the cluster state
            - save: True save the cov and mu as a h5 file with name filename
            - filename: name of the file to save"""
        if gif:
            fig, ax= plt.subplots()
            fig.set_figheight(15)
            fig.set_figwidth(15)
            ims = []
            norm = mpl.colors.Normalize(vmin=-1, vmax=1)

            title = ax.text(0.5, 1.05, "Squeezing operation", 
                        transform=ax.transAxes, ha="center", fontsize=16)
            
        self.apply_squeezing(r)

        if gif:
            ims.append([self.state.plot_cov_matrix(ax = ax,save = True, norm = norm),title])

            title = ax.text(0.5, 1.05, "Rotation operation", 
                        transform=ax.transAxes, ha="center", fontsize=16)
            
        self.apply_rotation_halfstate(theta=-np.pi/2)
        if gif : 
            ims.append([self.state.plot_cov_matrix(ax = ax,save = True, norm = norm),title])
        
        for i in range(len(self.BS_indices)):
            if gif:
                title = ax.text(0.5, 1.05, f"Beam splitter {i} operation", 
                        transform=ax.transAxes, ha="center", fontsize=16)
            self.apply_beamsplitter(col=i)
            if gif:
                ims.append([self.state.plot_cov_matrix(ax = ax,save = True, norm = norm),title])
            if i ==0:
                if gif:
                    title = ax.text(0.5, 1.05, "Rotation operation", 
                        transform=ax.transAxes, ha="center", fontsize=16)
                self.apply_rotation_halfstate(theta=np.pi/2)
                if gif:
                    ims.append([self.state.plot_cov_matrix(ax = ax,save = True, norm = norm),title])

        if gif:
            num_modes = self.N*self.macronode_size*self.spatial_depth

            # Set custom x-axis labels with numbering
            first_X_index = 0
            first_P_index = num_modes
            middle_X_index = num_modes // 2
            middle_P_index = num_modes + middle_X_index
            last_P_index =2 * num_modes - 1

            # Set custom x-axis labels at specified positions
            xticks = [first_X_index, first_P_index, middle_X_index, middle_P_index, last_P_index]
            xticklabels = [
            f"modes X{first_X_index + 1}",
            f"modes P{first_P_index - num_modes + 1}",
            f"modes X{middle_X_index + 1}",
            f"modes P{middle_P_index - num_modes + 1}",
            f"modes P{last_P_index - num_modes + 1}"
            ]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
            ax.set_yticks(xticks)
            ax.set_yticklabels(xticklabels, rotation=45, ha="right")
            fig.colorbar(mpl.cm.ScalarMappable(norm= norm, cmap=mpl.colormaps.get_cmap("seismic")),ax = ax)
            ani = animation.ArtistAnimation(fig,ims,interval=2000,blit =True)
            ani.save("Cluster covariance matrix animation.gif")
        
        if save:
            self.save_calculation(filename, r)
        
    def reset_calculation(self):
        """Function that reset the multipartite state to vaccum"""
        self.state = multipartite_state(number_of_modes = self.spatial_depth*self.macronode_size*self.N)
    
    def save_calculation(self, filename:str, r:float):
        """Function that save all metadata and covariance matrix and mean vector of the multipartite state
        
        Args:
            - filename: name of the file
            - r: float squeezing parameter"""
        with h.File(filename,'w') as f:
            f.create_dataset('Covariance matrix',data=self.state.cov)
            f.create_dataset('Mean vector',data=self.state.mu)
            f.attrs["n"] = self.n
            f.attrs["m"] = self.m
            f.attrs["k"] = self.k
            f.attrs["Depth"] = self.spatial_depth
            f.attrs["Macronode_size"] = self.macronode_size
            f.attrs["Squeezing parameter"] = r
         
    ###########################################################################
    ############################    MEASUREMENT    ############################
    ###########################################################################

    def measurement_based_squeezing(self, macronode_number:int):
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        mu, cov = self.state.mu, self.state.cov

        modes = self.macronode + macronode_number*ms

        Xip = np.pi/2 + np.arctan(2)
        Xim = np.pi/2 - np.arctan(2)
        thetas = np.array([Xip, Xim, Xip, Xim, Xip, Xim, Xip, Xim])
        m, c, u = self.measurement_gaussian(modes, thetas)

        indices = self.macronode_output + macronode_number*ms
        slices_C = np.ix_(np.concatenate([indices,indices+N*ms*depth]),np.concatenate([indices,indices+N*ms*depth]))
        slices_A = np.ix_(np.concatenate([modes,modes+N*ms*depth]),np.concatenate([modes,indices+N*ms*depth]))
        A = cov[slices_A]    # cov matrix of the macronode measured

        slices_0 = np.ix_([0,8],[0,8]) 
        slices_1 = np.ix_([1,9],[1,9]) 
        slices_2 = np.ix_([2,10],[2,10]) 
        slices_3 = np.ix_([3,11],[3,11]) 
        slices_4 = np.ix_([4,12],[4,12]) 
        slices_5 = np.ix_([5,13],[5,13]) 
        slices_6 = np.ix_([6,14],[6,14]) 
        slices_7 = np.ix_([7,15],[7,15]) 

        A0 = A[slices_0] # cov matrix of the measured modes 0 of the macronode
        A2 = A[slices_2] # cov matrix of the measured modes 2 of the macronode
        A4 = A[slices_4] # cov matrix of the measured modes 4 of the macronode
        A6 = A[slices_6] # cov matrix of the measured modes 6 of the macronode

        S = self.S(1,np.array([0]),1)
        print(S)

        A0_out = S @ A0 @ S.conj().T # Expected cov matrix for the output state 
        A2_out = S @ A2 @ S.conj().T # Expected cov matrix for the output state 
        A4_out = S @ A4 @ S.conj().T # Expected cov matrix for the output state 
        A6_out = S @ A6 @ S.conj().T # Expected cov matrix for the output state  
        
        C = c[slices_C]
        C1 = C[slices_1] # cov matrix of the measured modes 1 of the macronode
        C3 = C[slices_3] # cov matrix of the measured modes 3 of the macronode
        C5 = C[slices_5] # cov matrix of the measured modes 5 of the macronode
        C7 = C[slices_7] # cov matrix of the measured modes 7 of the macronode     

        print(r"$S_{0\rightarrow1}$:"+f"{np.allclose(A0_out,C1)}", np.abs(A0_out - C1))    
        print(r"$S_{2\rightarrow3}$:"+f"{np.allclose(A2_out,C3)}", np.abs(A2_out - C3)) 
        print(r"$S_{4\rightarrow5}$:"+f"{np.allclose(A4_out,C5)}", np.abs(A4_out - C5)) 
        print(r"$S_{6\rightarrow7}$:"+f"{np.allclose(A6_out,C7)}", np.abs(A6_out - C7)) 
        print(A)
        print(C)    

    def plot_wigner(self, mode:int, angle:np.ndarray):
        """Plot the wigner function of the desired mode after the measurement of all the other modes
        
        Args:
            - mode: int desired mode to plot the wigner
            - angle: list of the x angles for the x modes that will be repeated for all macronodes"""
        if self.macronode_size == 8:
            if len(angle) !=8:
                print("angle must be of size 8")
                return
        else:
            if len(angle)!= 2 :
                print("angle must be of size 2")
                return
        indice = np.arange(self.N*self.macronode_size*self.spatial_depth)    
        angle = np.tile(angle,self.N*self.spatial_depth)    
        indice = np.delete(indice,mode)
        angle = np.delete(angle,mode)

        mu, cov, u =self.measurement_gaussian(indice,angle, plot=True)
        return mu, cov, u

    def measurement_gaussian(self, modes:np.ndarray, thetas:np.ndarray, plot=False):
        """ Measure the X(theta) quadrature of the desired modes and plot the resulted wigner function
        
        Args:
            - modes: list of modes to with measure the x quadrature
            - theta: list of angles to wich the measurement will be apply
            """
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        self.apply_rotation(modes,thetas)
        mu, cov, u = self.measurement_X(modes)
        if plot :
            n = len(mu)//2
            def wigner(r):
                norm_factor = 1 / ((2 * np.pi) ** n * np.sqrt(np.linalg.det(cov)))
                exponent = -0.5 * (r - mu).T @ np.linalg.inv(cov) @ (r - mu)
                return norm_factor * np.exp(exponent)

            x_values = np.linspace(-3, 3, 100)
            p_values = np.linspace(-3, 3, 100)
            X, P = np.meshgrid(x_values, p_values)

            W = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    r = np.array([X[i, j], P[i, j]])
                    W[i, j] = wigner(r)

            plt.figure(figsize=(8, 6))
            plt.contourf(X, P, W, levels=100, cmap='seismic', norm = norm)
            plt.colorbar(label='Wigner Function Value')
            plt.xlabel('x')
            plt.ylabel('p')
            plt.title('Wigner Function in Phase Space')
            plt.show()
        return mu, cov, u
    
    def measurement_X(self, modes: np.ndarray):
        """ Measure the X quadrature of the desired modes
        
        Args:
            - modes: list of modes to with measure the x quadrature
            
        Return:
            - mu: displacement vector resulted from the measurement
            - cov: covariance matrix resulted from the measurement"""
        
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        mu, cov = self.state.mu, self.state.cov

        indices = np.concatenate([modes,modes + N*ms*depth])
        ar = np.arange(2*N*ms*depth)
        mask = ~np.isin(ar, indices)

        slices_B = np.ix_(indices,indices)
        slices_Ct = np.ix_(indices,ar[mask])
        slices_C = np.ix_(ar[mask],indices)
        slices_A = np.ix_(ar[mask],ar[mask])

        A = cov[slices_A].copy()
        B = cov[slices_B].copy()
        C = cov[slices_C].copy()

        PI = np.eye(2 * len(modes))
        PI[len(modes):, len(modes):] = 0
        cov = A - C @ np.linalg.pinv(PI @ B @ PI) @ C.T

        a = mu[ar[mask]]
        b = mu[indices]

        mean = PI @ b
        var = np.diagonal(B)
        distribution = np.vectorize(np.random.normal)
        u = distribution(mean,var)

        mu = a - C @ np.linalg.pinv(PI @ B @ PI) @ (b - u)

        return mu, cov, u
        
    ###########################################################################
    #########################    STATE EVOLUTION    ###########################
    ###########################################################################

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


        slices_B = np.ix_(indices,indices)
        slices_Ct = np.ix_(indices,ar[mask])
        slices_C = np.ix_(ar[mask],indices)
        cov[slices_B] = F @ cov[slices_B] @ F.conj().T 
        cov[slices_C] = cov[slices_C] @ F.conj().T
        cov[slices_Ct] = cov[slices_C].T
    
    def apply_squeezing(self, r:float):
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

    def apply_rotation(self, modes:np.ndarray, thetas:np.ndarray):
        """Introduce theta phase shift on desired modes.
         
        Args:
            - modes: to which apply the rotation
            - theta: angles of the rotation matrix
        
        Update:
            Update the covariance matrix and µ vector of the multipartite state"""
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth

        P = self.P(N = len(modes),
                   modes = np.arange(len(modes)),
                   theta = thetas)
        
        indice_XP = np.concatenate([modes+N*ms*depth,modes]) # I don"t understand why I have to invert the two indices of Xs an Ps to make the rotation in the good direction. Doing so same for squeezing makes everything collaps.
        self.apply_symplectic(P, indice_XP)
    
    def apply_rotation_halfstate(self, theta:float):
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
        
        indice_XP = np.concatenate([indice_rot+N*ms*depth,indice_rot]) # I don"t understand why I have to invert the two indices of Xs an Ps to make the rotation in the good direction. Doing so same for squeezing makes everything collaps.
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

        self.apply_symplectic(BS,np.concatenate([np.concatenate(indices),np.concatenate(indices+N*ms*depth)]))

    ###########################################################################
    ############################    SIMPLECTIC    #############################
    ###########################################################################
    
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
        F[modes,modes] = np.exp(-r)
        F[modes+N,modes+N] = np.exp(r)
        return F
    
    ###########################################################################
    ###########################    MYSCALENIOUS    ############################
    ###########################################################################

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
        elif ms == 2:
            interaction_matrix = np.array([ [    0+1    ,   np.nan          ,        np.nan      ,              np.nan    ,         np.nan   ],
                                            [    np.nan   ,  0 + 2*1   ,   0 + 2*(1+n)    ,      0 + 2*(1+n+n*m)        ,     0 + 2*(1+n+n*m+n*m*k)   ]])
            if k == 1:
                
                if m == 1:
                    interaction_matrix = interaction_matrix[:,:-2]
                else:
                    interaction_matrix = interaction_matrix[:,:-1]
            
                            
        i_indices = np.arange(ms)
        # Mask to identify non-nan values in the interaction matrix
        mask = ~np.isnan(interaction_matrix)
        # Broadcasting the i_indices to match the shape of the interaction_m
        i_to_i7 = np.broadcast_to(i_indices[:, np.newaxis], interaction_matrix.shape)
        # Apply the mask to get the valid indices and interaction values
        valid_i_to_i7 = [i_to_i7[mask[:, col], col] for col in range(interaction_matrix.shape[1])]
        valid_interactions = [interaction_matrix[mask[:, col], col] for col in range(interaction_matrix.shape[1])]

        self.BS_indices = [self.generate_arrays_from_pairs(valid_i_to_i7[col],valid_interactions[col],ms,N*ms*depth-1) for col in range(len(valid_i_to_i7))]

    def generate_macronodes(self):
        """Function that generate the indices of modes that represent a unique macronodes.
        one macronode : { i - 8nm ; i+1 ; i+2 - 8 ; i+3; i+4 - 8nmk ; i+5 ; i+6 - 8n ; i+7}
        Representation with future only : { i + 8(nmk - nm) ; i+1 + 8nmk ; i+2 + 8(nmk + 1) ; i+3 + 8nmk ; i+4 ; i+5 + 8nmk ; i+6 + 8(nmk-n) ; i+7 + 8nmk}"""
        n, m, k , N, ms, depth = self.n, self.m, self.k, self.N, self.macronode_size, self.spatial_depth
        if ms == 8:
            self.macronode = np.array([0 + 8*(n*m*k - n*m) , 0+1 + 8*n*m*k , 0+2 + 8*(n*m*k + 1) , 0+3 + 8*n*m*k , 0+4 , 0+5 + 8*n*m*k , 0+6 + 8*(n*m*k-n) , 0+7 + 8*n*m*k])
            entangled_to_macronode = np.array([0 + 8*(n*m*k - n*m) + 7 , 0+1 + 8*n*m*k + 1, 0+2 + 8*(n*m*k + 1) -1, 0+3 + 8*n*m*k + 1, 0+4 - 1, 0+5 + 8*n*m*k + 1, 0+6 + 8*(n*m*k-n) - 1, 0+7 + 8*n*m*k - 7])
            self.macronode_output = entangled_to_macronode - np.array([-2,-5,-3,-6,0,-7,2,-7])  # Indices of the output after tracing out measurement onto the macronode

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


if __name__ == "__main__":
    cs = cluster_state()
    cs.apply_beamsplitter(0) 