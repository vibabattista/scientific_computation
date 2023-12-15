"""
Code for Scientific Computation Project 2
Please add college id here
CID: 01847210
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#use scipy in part 2 as needed

#===== Codes for Part 1=====#
def searchGPT(graph, source, target):
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('inf') for node in graph}
    distances[source] = 0

    # Initialize a priority queue to keep track of nodes to explore
    priority_queue = [(0, source)]  # (distance, node)

    # Initialize a dictionary to store the parent node of each node in the shortest path
    parents = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current node is the target, reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node in parents:
                path.insert(0, current_node)
                current_node = parents[current_node]
            path.insert(0, source)
            return current_distance,path

        # If the current distance is greater than the known distance, skip this node
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = max(distances[current_node], weight['weight'])
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')  # No path exists


def searchPKR(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s])
    Mdict[s]=Mlist[0]
    found = False

    while len(Mlist)>0:
        dmin,nmin = heapq.heappop(Mlist)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn)
                if dcomp<Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp,en]
                    heapq.heappush(Mlist,lnew)
                    Mdict[en]=lnew
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en]
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    return dmin


def searchPKR2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to searchGPT given same input
    """

    from collections import deque #use deque instead of list for left appending of order 1

    Fdict = {} #initialise dictionaries
    Mdict = {}
    Mlist = []
    parents = {} #create a parents dictionary which tracks each nodes parent in a path, for constructing the path
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s]) #place source node in priority queue
    Mdict[s]=Mlist[0]
    found = False

    while len(Mlist)>0: 
        dmin,nmin = heapq.heappop(Mlist) #O(logN)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m,en,wn in G.edges(nmin,data='weight'): #O(2E) over the whole loop each edge looked at twice
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn) 
                if dcomp<Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n #does not change outcome since nodes are numbered 0 to n-1
                    lnew = [dcomp,en] 
                    heapq.heappush(Mlist,lnew) #O(logN)
                    Mdict[en]=lnew
                    parents[en]=nmin
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en]
                heapq.heappush(Mlist, lnew) #O(logN)
                Mdict[en] = lnew
                parents[en]=nmin
    if found == True: #reconstruct path to target node
        path = deque() #allows for order 1 left appending
        current_node = x
        while current_node in parents: #O(N) #appends each parent node successively to the path until hitting the source node
            path.appendleft(current_node)
            current_node = parents[current_node]
        path.appendleft(s) #since source node has no parent, append to queue separately
        return dmin, list(path)
    else:
        return float('inf') #if found != True there is no path, return inf as in searchGPT

    #return dmin and list

#===== Code for Part 2=====#
def part2q1(y0,tf=1,Nt=5000):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    import numpy as np
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = np.zeros_like(y)
        for i in range(1,n-1):
            dydt[i] = alpha*y[i]-y[i]**3 + beta*(y[i+1]+y[i-1])
        
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt 


    #Compute numerical solutions
    dt = tarray[1]
    for i in range(Nt):
        yarray[i+1,:] = yarray[i,:]+dt*RHS(0,yarray[i,:])

    return tarray,yarray

def part2q1new(y0,tf=40,Nt=800):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.empty((Nt+1,n)) #changed from zeros to save time
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta

    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = np.empty_like(y) #changed from zeros_like()
        dydt[1:-1] = alpha*y[1:-1] + beta*(y[2:] + y[:-2]) -y[1:-1]**3 #changed by using ndarrays

        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])
        return dydt 

    #Compute numerical solutions dydt[i] = alpha*y[i]-y[i]**3 + beta*(y[i+1]+y[i-1])

    solution = solve_ivp(RHS, [0,tf], y0, t_eval=tarray, method ='BDF', atol = 10**(-6), rtol=10**(-6)) #BDF method is the most efficient after experimentation
    #note the default absolute tolerance for this function is 10e-6, which is the required accuracy
    tarray, yarray = solution.t, solution.y.T #Transposing the solution array gives required dimensions
    return tarray,yarray



def part2q2(): #add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import optimize
    data = np.load('project2.npy') #modify/discard as needed
    y0A = data[0,:] #first initial condition
    y0B = data[1,:] #second initial condition
    tarrayA, yarrayA = part2q1new(y0A, tf = 40, Nt=5000)
    tarrayB, yarrayB = part2q1new(y0B, tf = 40, Nt=5000)
    
    def RHS(y): #defining for optimize.root
        """
        Compute RHS of model
        """        
        beta = 10000/np.pi**2
        alpha = 1-2*beta
        dydt = np.empty_like(y) #changed from zeros
        
        dydt[1:-1] = alpha*y[1:-1] + beta*(y[2:] + y[:-2]) -y[1:-1]**3 #changed by using ndarrays
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt 
    
    def jacobian(y):
        n = len(y)
        J = np.zeros((n, n))
        beta = 10000/np.pi**2
        alpha = 1-2*beta

    # Fill in the diagonal elements
        for i in range(n):
            J[i, i] = alpha - 3 * y[i] ** 2

    # Fill in the off-diagonal elements for the coupling terms
        for i in range(n - 1):
            J[i, i + 1] = beta  # For y[i+1]
            J[i + 1, i] = beta  # For y[i-1]

    # Handle the periodic boundary conditions
        J[0, n - 1] = beta  # For y[-1] when i = 0
        J[n - 1, 0] = beta  # For y[0] when i = n - 1
        return J

    equilibriumA = optimize.root(fun = RHS, x0 = y0A).x
    equilibriumB = optimize.root(fun = RHS, x0 = y0B).x

    # Create a figure and a 2 by 3 grid of subplots
    fig, axs = plt.subplots(3, 2)

    # Titles for each subplot
    titles = ["Solution for initial conditions from dataset A", "Solution for initial conditions from dataset B", "Dataset A", "Dataset B", "Equilibrium Solution close to A", "Equilibrium solution close to B"]

    # Iterate through the grid and set titles and labels
    for i in range(3):
        for j in range(2):
            axs[i, j].set_title(titles[i*2 + j])
    axs[0,0].plot(tarrayA,yarrayA)
    axs[0,1].plot(tarrayB,yarrayB)
    axs[1,0].scatter(range(1000),y0A,s=1)
    axs[1,1].scatter(range(1000),y0B,s=1)
    axs[2,0].scatter(range(1000),equilibriumA, s=0.1)
    axs[2,1].scatter(range(1000),equilibriumB, s=0.1)
    axs[0, 0].set_xlabel('t')
    axs[0, 0].set_ylabel('y')
    axs[0, 1].set_xlabel('t')
    axs[0, 1].set_ylabel('y')
    axs[1, 0].set_xlabel('i')
    axs[1, 0].set_ylabel(r'$y_{\mathrm{i}}$'+' at t=0')
    axs[1, 1].set_xlabel('i')
    axs[1, 1].set_ylabel(r'$y_{\mathrm{i}}$'+' at t=0')
    axs[2,0].set_xlabel('i')
    axs[2,0].set_ylabel(r'$y_{\mathrm{i}}$')
    axs[2,1].set_xlabel('i')
    axs[2,1].set_ylabel(r'$y_{\mathrm{i}}$')
    # Adjust layout for better display
    plt.tight_layout()

    # Show the plot
    plt.show()

    jacobian_at_epA = jacobian(equilibriumA)
    jacobian_at_epB = jacobian(equilibriumB)
    jacobian_at_1 = jacobian(np.ones(1000))
    eigA = np.linalg.eig(jacobian_at_epA)
    eigB = np.linalg.eig(jacobian_at_epB)
    eig1 = np.linalg.eig(jacobian_at_1)
    print("positive eigenvalues for Jacobian at the 1 Vector:",[i for i in eig1[0] if i>0])
    print("positive eigenvalues for Jacobian at equilibrium A:",[i for i in eigA[0] if i>0])
    print("positive eigenvalues for Jacobian at equilibrium B:",[i for i in eigB[0] if i>0])
    
    eigenvaluesA, eigenvectorsA = eigA #to create a subspace spanned by eigenvectors
    # to create subspace with only positive eigenvalue eigenvectors
    positive_eigenvaluesA = eigenvaluesA > 0
    basis_positive_eigenvectorsA = eigenvectorsA[:, positive_eigenvaluesA]

    def project_onto_subspace(basis, vector):
        """
        project a vector into a subspace spanned by basis.
        """
        # use qr to orthogonalise 
        Q, _ = np.linalg.qr(basis)
        
        # projection matrix constructed as so, when multiplied by x does a dot product with each eigenvector then multiples by that eigenvector
        projection_matrix = Q @ Q.T
        
        # gives the dimensions of vector in said subspace
        projection = projection_matrix @ vector
        
        return projection

    # we want to check if initial conditions A are wholly contained in the subspace of eigenvectors with negative eigenvalue, or equivalently if any of y0A is not in this space:
    print("Initial condition A is contained in subspace attracted to equilibrium point A:",np.allclose(project_onto_subspace(basis_positive_eigenvectorsA, y0A),np.zeros(1000)))
    
    return None 
    #return None #modify as needed


def part2q3(tf=10,Nt=1000,mu=0.2,seed=1):
    """
    Input:
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same random numbers are generated with each simulation

    Output:
    tarray: size Nt+1 array
    X size n x Nt+1 array containing solution
    """

    #Set initial condition
    y0 = np.array([0.3,0.4,0.5])
    np.random.seed(seed)
    n = y0.size #must be n=3
    Y = np.zeros((Nt+1,n)) #may require substantial memory if Nt, m, and n are all very large
    Y[0,:] = y0

    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    beta = 0.04/np.pi**2
    alpha = 1-2*beta

    def RHS(t,y):
        """
        Compute RHS of model
        """
        dydt = np.array([0.,0.,0.])
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[2])
        dydt[1] = alpha*y[1]-y[1]**3 + beta*(y[0]+y[2])
        dydt[2] = alpha*y[2]-y[2]**3 + beta*(y[0]+y[1])

        return dydt 

    dW= np.sqrt(Dt)*np.random.normal(size=(Nt,n))

    #Iterate over Nt time steps
    for j in range(Nt):
        y = Y[j,:]
        F = RHS(0,y)
        Y[j+1,0] = y[0]+Dt*F[0]+mu*dW[j,0]
        Y[j+1,1] = y[1]+Dt*F[1]+mu*dW[j,1]
        Y[j+1,2] = y[2]+Dt*F[2]+mu*dW[j,2]

    return tarray,Y


def part2q3Analyze(): #add input variables as needed
    """
    Code for part 2, question 3
    """
    import matplotlib.pyplot as plt
    t1, Y1 = part2q3(tf = 150, mu = 0.5)
    colors = ['darksalmon', 'lightsalmon', 'coral']
    colors2 = ['lightseagreen', 'aquamarine', 'turquoise']
    colors3 = ['cornflowerblue','navy','royalblue']
    labels = ['mu = 0.5', '_', '_']
    labels2 = ['mu=0.2','_','_']
    labels3 = ['mu=0.0','_','_']
    for i in range(3):
        plt.plot(t1,Y1[:,i], color = colors[i],label = labels[i]) #plots noisy solution

    t2, Y2 = part2q3(tf = 150, mu = 0.2)
    for i in range(3): #plots small noise solution
        plt.plot(t2,Y2[:,i], color = colors2[i],label = labels2[i])


    t3, Y3 = part2q3(tf = 150, mu = 0.0)
    for i in range(3): #plots noiseless solution
        plt.plot(t3,Y3[:,i], color =colors3[i],label=labels3[i])

    

    plt.title('A plot to show each dimension of a solution for varying mu')
    plt.legend()
    plt.show()
    #add code for generating figures and any other relevant calculations here

    return None #modify as needed
