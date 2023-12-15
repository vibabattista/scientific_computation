"""Scientific Computation Project 3
01847210
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#use scipy as needed
from matplotlib.animation import FuncAnimation
import scipy as scipy
from scipy.signal import welch
#===== Code for Part 1=====#
def pca(A,p=None):
    means = np.mean(A, axis=1)
    A=(A.T-means).T #centralise to ignore means
    U, S, VT = np.linalg.svd(A) #svd for singular values and pcs
    pcs = U.T
    scores = pcs@A
    explained_variance_ratio = S**2/np.sum(S**2)
    A_reduced = pcs[:p].T@scores[:p] #reconstruction
    A_reduced = (A_reduced.T + means).T #add back means
    pcp = np.outer(pcs[p,:].T, scores[p,:]) #project only pth pc onto data
    return pcs, scores, explained_variance_ratio, A_reduced, pcp

def plot_spatialPCs(p, reshaped_u,lat,lon):
    pcs = pca(reshaped_u, p)[0]
    nrows = (p + 1) // 2 #format plots nicely
    if p > 1:
        ncols = 2 
    else:
        ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows * 4)) 

    for i in range(p):
        # Determine the correct subplot to use
        if nrows > 1:
            ax = axs[i // ncols, i % ncols]
        elif ncols > 1:
            ax = axs[i % ncols]
        else:
            ax = axs

        contour = ax.contourf(lon, lat, pcs[i].reshape(16,144), levels=50) #plot pc
        ax.set_title(f'PC{i+1}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.colorbar(contour, ax=ax)

    # Hide any unused subplots when p is an odd number
    if p % 2 != 0 and p > 1:
        axs[-1, -1].axis('off')

    # Adjust layout with additional space
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.4)
    plt.show()

def plot_temporalPCs(p, reshaped_u,lat,lon):
    temporal_pcs = pca(reshaped_u.T, p)[0]
    nrows = (p + 1) // 2
    if p > 1:
        ncols = 2 
    else:
        ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows * 4)) 

    for i in range(p):
        if nrows > 1:
            ax = axs[i // ncols, i % ncols]
        elif ncols > 1:
            ax = axs[i % ncols]
        else:
            ax = axs

        ax.set_title(f'PC{i+1}')
        ax.plot(temporal_pcs[i])

    if p % 2 != 0 and p > 1:
        axs[-1, -1].axis('off')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.4)
    plt.show()

def plot_temporalPCspectra(p,reshaped_u):
    temporal_pcs = pca(reshaped_u.T, p)[0]
    temporal_pcs = (temporal_pcs - temporal_pcs.mean(axis=1, keepdims=True))
    freqs = np.abs(np.fft.fftfreq(365))
    
    nrows = (p + 1) // 2
    if p > 1:
        ncols = 2 
    else:
        ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows * 4)) 

    for i in range(p):
        pc_fft = np.fft.fft(temporal_pcs[i])
        pc_fft_magnitude = np.abs(pc_fft)
        if nrows > 1:
            ax = axs[i // ncols, i % ncols]
        elif ncols > 1:
            ax = axs[i % ncols]
        else:
            ax = axs

        ax.set_title(f'PC{i+1}')
        ax.plot(freqs,pc_fft_magnitude)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')

    if p % 2 != 0 and p > 1:
        axs[-1, -1].axis('off')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.4)
    plt.show()



def plot_field(lat,lon,u,time,levels=20):
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    plt.figure()
    plt.contourf(lon,lat,u[time,:,:],levels)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    
    return None


def animate_field(lat, lon, u, levels=20):
    """
    Generates a contour plot of u for all times as an animation.
    """
    fig, ax = plt.subplots()
    contour = ax.contourf(lon, lat, u[0, :, :], levels)
    ax.set_title("Zonal Wind Speed, t = 0")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    def update(time):
        ax.clear()
        contour = ax.contourf(lon, lat, u[time, :, :], levels)
        ax.set_title(f"Zonal Wind Speed, t = {time}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        

    ani = FuncAnimation(fig, update, frames=range(365), interval=50)
    plt.show()

def plot_scores_spatial(n_components, reshaped_u):
    pcs, scores, _, u_approx, pcp = pca(reshaped_u, n_components)

    fig, axs = plt.subplots(n_components, 1, figsize=(10, 12))
    time_points = np.arange(scores.shape[1])
    for i in range(n_components):
        axs[i].plot(time_points, scores[i])
        axs[i].set_title(f'Scores of PC{i+1} Over Time')
        axs[i].set_xlabel('Time Point')
        axs[i].set_ylabel(f'Scores of PC{i+1}')
    plt.tight_layout()
    plt.show()

def plot_scores_temporal(n_components, reshaped_u,lon,lat):
    pcs, scores, _, u_approx, pcp = pca(reshaped_u.T, n_components)
    
    fig, axs = plt.subplots(n_components, 1, figsize=(10, 12))
    for i in range(n_components):
        axs[i].contourf(lon,lat, scores[i].reshape(16,144), levels=20)
        axs[i].set_title(f'Scores of PC{i+1} Over Time')
        axs[i].set_xlabel('Time Point')
        axs[i].set_ylabel(f'Scores of PC{i+1}')
    plt.tight_layout()
    plt.show()

def part1(no_components=None, explained_variance_spatial=False, animation=False, spatialPCs=False, temporalPCs=False, averagespeeds=False, plotspectra=False, explained_variance_temporal=False, spatial_scores=False, temporal_scores=False):#add input if needed
    """
    Code for part 1
    """ 

    #--- load data ---#
    d = np.load('data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    #-------------------------------------#

    #Add code here 
    reshaped_u = u.reshape(365,16*144).T
    pcs, scores, variance_ratio, u_approx, _ = pca(reshaped_u,no_components) 

    if explained_variance_spatial:
        cumulative_variance = np.cumsum(variance_ratio)
        plt.plot(cumulative_variance, '.')
        plt.xlabel('Component Number')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')
        plt.show()
    if explained_variance_temporal:
        variances = pca(reshaped_u.T,no_components)[2]
        cumulative_variance = np.cumsum(variances)
        plt.plot(cumulative_variance, '.')
        plt.xlabel('Component Number')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')
        plt.show()
    if animation:
        u_approx = u_approx.T.reshape(365,16,144)
        animate_field(lat,lon,u_approx,levels = 50)
    if spatialPCs:
        plot_spatialPCs(no_components,reshaped_u,lat,lon)
    if temporalPCs:
        plot_temporalPCs(no_components, reshaped_u,lat,lon)
    if averagespeeds:
        yearly_averages = np.mean(u, axis=0)
        # Create a contour plot
        plt.figure(figsize=(10, 6))
        plt.contourf(lon, lat, yearly_averages, 100, cmap='viridis')  
        plt.colorbar(label='Average Zonal Speed')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Yearly Average Zonal Speed')
        plt.show()
    if plotspectra:
        plot_temporalPCspectra(no_components,reshaped_u)
    if spatial_scores:
        plot_scores_spatial(no_components, reshaped_u)
    if temporal_scores:
        plot_scores_temporal(no_components, reshaped_u, lon, lat)

    return None #modify if needed

d = np.load('data1.npz')
lat = d['lat'];lon = d['lon'];u=d['u']
#-------------------------------------#

#===== Code for Part 2=====#
def part2(f,method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """

    m,n = f.shape
    fI = np.zeros((m-1,n)) #use/modify as needed

    if method==1:
        fI = 0.5*(f[:-1,:]+f[1:,:])
    else:
        #Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1
        
        #coefficients for near-boundary points
        a_bc,b_bc,c_bc,d_bc = (5/16,15/16,-5/16,1/16)

        #write the system as a banded matrix equation
        A_banded = [[0]*2+(m-3)*[alpha], list(np.ones(m-1)), (m-3)*[alpha] +2*[0]]
        c = np.empty_like(f[1:])
        #write the RHS as a vector c
        c[0] = a_bc*f[0,:] + b_bc*f[1,:] + c_bc*f[2,:] + d_bc*f[3,:]
        c[-1] = a_bc*f[-1,:] + b_bc*f[-2,:] + c_bc*f[-3,:] + d_bc*f[-4,:]
        c[1:-1] = (b/2) * (f[:-3, :] + f[3:, :]) + (a/2) * (f[2:-1, :]+f[1:-2,:])
        fI = scipy.linalg.solve_banded((1,1),A_banded, c)

    return fI #modify as needed

def errors(k, display=False):
    n,m = 50,42
    x = np.linspace(0,np.pi,n)
    y = np.linspace(0,np.pi,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    xgI, ygI = np.meshgrid(x,yI)

    f = np.cos(k*xg)*np.sin(k*yg) #test function of wavenumber k
    factual = np.cos(k*xgI)*np.sin(k*ygI) #actual data to compare with interpolated

    fI1 = part2(f,method=1) #interpolated
    fI2 = part2(f,method=2)

    error1 = np.abs(fI1-factual) #absolute difference
    error2 = np.abs(fI2-factual)

    if display: #plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Plot contour subplot 1
        axs[0].contourf(xgI, ygI, error1, cmap='magma', levels=50)
        axs[0].set_title('Method 1')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')

        # Plot contour subplot 2
        axs[1].contourf(xgI, ygI, error2, cmap='magma',levels=50)
        axs[1].set_title('method 2')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')

        cbar1 = fig.colorbar(axs[0].contourf(xgI, ygI, error1, cmap='magma', levels=50), ax=axs[0])
        cbar2 = fig.colorbar(axs[1].contourf(xgI, ygI, error2, cmap='magma', levels=50), ax=axs[1])

        plt.tight_layout()
        plt.show()
        
    ferror1, ferror2 = np.linalg.norm(error1, ord='fro'), np.linalg.norm(error2, ord='fro')
    print('Frobenius norm of method 1 interpolation error:', ferror1)
    print('Frobenius norm of method 2 interpolation error:', ferror2)
    # Display the plots
    
    return ferror1, ferror2 #returns frobenius norm errors

def data(k):
    n,m = 50,42
    x = np.linspace(0,np.pi,n)
    y = np.linspace(0,np.pi,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    xgI, ygI = np.meshgrid(x,yI)

    factual = np.cos(k*xgI)*np.sin(k*ygI)
    plt.contourf(xgI, ygI, factual, cmap='magma',levels=50)
    plt.title('Actual data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def wavenumberanalysis():
    v = np.linspace(0,np.pi,100)
    alpha = 0.3
    a = 1.5
    b = 0.1

    #coefficients for near-boundary points
    a_bc,b_bc,c_bc,d_bc = (5/16,15/16,-5/16,1/16)

    plt.plot(v,v,label='Exact')
    plt.plot(v,v*np.cos(v/2), label='method 1 modified wavenumber')
    plt.plot(v, v*(b*np.cos(v*3/2)+a*np.cos(v*1/2)-2*alpha*np.cos(v)),label='method 2 modified wavenumber')
    plt.legend()
    plt.xlabel('kh')
    plt.ylabel("modified wave number kh'")
    plt.show()

def part2_analyze(k=5, error=False, k_errors=False, wavenumbers =False):
    """
    Add input/output as needed
    """

    #----- Code for generating grid, use/modify/discard as needed ----#
    n,m = 50,40 #arbitrary grid sizes
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    #--------------------------------------------#

    #add code here
    if error:
        data(k)
        errors(k, display=True)
    if k_errors:
        kvals = np.arange(42)
        error1list = []
        error2list = []
        for k in kvals:
            e1, e2 = errors(k)
            error1list.append(e1)
            error2list.append(e2)
        plt.plot(error1list, label = 'method 1')
        plt.plot(error2list, label='method 2')
        plt.legend()
        plt.xlabel('wavenumber')
        plt.ylabel('Error')
        plt.show()
    if wavenumbers:
        wavenumberanalysis()
    return None #modify as needed



#===== Code for Part 3=====#
def part3q1(y0,alpha,beta,b,c,tf=200,Nt=800,err=1e-6,method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays

    n = y0.size//2
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,2*n))
    yarray[0,:] = y0


    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        u = y[:n];v=y[n:]
        r2 = u**2+v**2
        nu = r2*u
        nv = r2*v
        cu = np.roll(u,1)+np.roll(u,-1)
        cv = np.roll(v,1)+np.roll(v,-1)

        dydt = alpha*y
        dydt[:n] += beta*(cu-b*cv)-nu+c*nv+b*(1-alpha)*v
        dydt[n:] += beta*(cv+b*cu)-nv-c*nu-b*(1-alpha)*u

        return dydt


    sol = solve_ivp(RHS, (tarray[0],tarray[-1]), y0, t_eval=tarray, method=method,atol=err,rtol=err)
    yarray = sol.y.T 
    return tarray,yarray

def correlation_sum(u):
    u = u[::4] #slice such that successive iterations are uncorrelated
    print("correlation between consecutive vectors in u (time step 0.25):", np.corrcoef(u[0],u[1])[0,1])
    print('correlation between consecutive vectors in u[::4] (time step 1):', np.corrcoef(u[0],u[4])[0,1])
    C = np.zeros(100) #initialise C vector
    D = scipy.spatial.distance.pdist(u) #Calculate distances
    eps = np.linspace(min(D)+0.01,max(D),100)[::-1] #decreasing order
    for i in range(len(eps)):
        D = D[D<eps[i]] #keeps distances smaller than eps, eps[i+1]<eps[i] 
        C[i] = D.size #C is a count of distances smaller than eps
    eps = eps[::-1] #increasing order
    C = C[::-1] #increasing order
    plt.plot(np.log(eps),np.log(C)) #loglog plot
    a,b = np.polyfit(np.log(eps[30:45]),np.log(C[30:45]), deg=1) #fits to a linear section of the graph
    yfit = np.poly1d([a,b])
    plt.plot(np.log(eps[20:65]), yfit(np.log(eps[20:65]))) #plot gradient line
    plt.xlabel('log(eps)')
    plt.ylabel('log(C(eps))')
    plt.show()
    print('Estimated fractal dimension:',a) #gradient
    return None

def spectra(u,t): #plots as a contour for all u at once
    fxx,Pxx = welch(u,fs=1/t[1], axis=0)
    # plt.figure()
    # plt.semilogy(fxx,Pxx)
    #plt.xlabel(r'$f$')
    # plt.ylabel(r'$P_{xx}$')
    # plt.grid()
    #print(Pxx.shape)
    plt.contourf(np.arange(Pxx.shape[1]),fxx,Pxx,levels=20)
    plt.ylabel(r'$f$')
    plt.ylim(0,0.25)
    plt.show()

def part3_analyze(c=1.3, display = False, spectrum=False, phaseplot=False, correlation_dimension=False):#add/remove input variables if needed
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    #c = 0.5
    t,y = part3q1(y0,alpha,beta,b,c,tf=200,Nt=2,method='RK45') #for transient, tf=200 is sufficient
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    u,v = y[:,:n],y[:,n:]

    if display:
        plt.figure()
        plt.contourf(np.arange(n),t,u,20)
        plt.show()
    #-------------------------------------------#

    #Add code here
    if phaseplot:
        plt.plot(u[700], v[700])
        plt.show()

    if spectrum:
        spectra(u,t)
        
    if correlation_dimension:
        correlation_sum(u)

    return None #modify if needed
#part3_analyze(display=True, c=1.3, spectrum=True, correlation_dimension=True, phaseplot=True)

def part3q2(x,c=1.0):
    """
    Code for part 3, question 2
    """
    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    y0 = np.zeros(2*n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)

    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #Compute solution
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    A = y[:,:n]

    #Analyze code here
    l1,v1 = np.linalg.eigh(A.T.dot(A)) #evals and evecs for covariance matrix of A
    v2 = A.dot(v1) #projection
    A2 = (v2[:,:x]).dot((v1[:,:x]).T) #reconstruct estimate
    e = np.sum((A2.real-A)**2) #cumulative error

    return A2.real,e


if __name__=='__main__':
    x=None #Included so file can be imported
    #Add code here to call functions above if needed
