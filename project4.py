"""
Project 4 code
CID: Add your CID here
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
#use scipy as needed
from hottbox.core import Tensor
from hottbox.metrics.decomposition import residual_rel_error
import time as time


def load_image(normalize=True,display=False):
    """"
    Load and return test image as numpy array
    """
    import scipy
    from scipy.datasets import face
    A = face()
    if normalize:
        A = A.astype(float)/255
    if display:
        plt.figure()
        plt.imshow(A)
    return A


#---------------------------
# Code for Part 1
#---------------------------
def unfold(tensor, mode):
    """
    Input:
    tensor: ndarray
    mode: arbitrary index of the tensor

    Output:
    unfolded tensor: order 2 ndarray
    """
    #add code here
    unfolded = np.reshape(np.moveaxis(tensor,mode,0), (tensor.shape[mode],-1))
    #rotates the axes of tensor such that 'mode'th axis is at index 0.
    #it then reshapes to (size of mode, product of other mode sizes)
    return unfolded

def truncated_svd(mat, delta):
    """
    Inputs:
    mat: order 2 numpy array
    delta: float
    Output:
    U[:, :rank]: numpy array
    S[:rank]: numpy array
    Vt[:rank, :]: numpy array
    """
    #add code here
    U, S, Vt = scipy.linalg.svd(mat, full_matrices=False)
    #computes SVD of matrix
    cumsum = np.cumsum(S**2)
    rank = np.searchsorted(cumsum, cumsum[-1] - delta**2, side = 'right')
    #The Frobenius norm of the matrix is equal to sqrt(sum(S**2))
    #The truncated SVD must be able to produce a matrix with frobenius norm within delta
    #searchsorted binary searches through S**2 to find the index of S such that the Frobenius norm is within the error
    rank = min(max(rank,1), len(S))
    #we set the rank to be at least 1 and at most full rank.
    return U[:, :rank], S[:rank], Vt[:rank, :] # truncate

def decompose1(A,eps):
    """
    Implementation of Algorithm 3 from KCSM
    Input:
    A: tensor stored as numpy array 
    eps: accuracy parameter
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """
    Glist = []
    N = A.ndim
    ranks = [1] #initialise the calculated ranks
    delta = np.linalg.norm(A.reshape(768, 3072), 'fro')*eps/np.sqrt(N-1) #compute truncation parameter
    Z = unfold(A,0) #initialise Z as unfolded A along first mode
    for n in range(N-1):
        #delta truncated svd
        U, S, Vt = truncated_svd(Z, delta)
        ranks.append(U.shape[1]) #stores calculated rank from svd
        G = U.reshape((ranks[-2], A.shape[n], ranks[-1])) #reshape to form core
        Glist.append(G) #record core
        Z = np.diag(S)@Vt #next iteration is for S*Vt
        Z = Z.reshape((ranks[-1]*A.shape[n+1], -1)) #unfolds and refolds SVt into the next mode unfolding
    Glist.append(Z.reshape(ranks[-1], A.shape[-1], 1)) #final core is the leftover S*Vt folded into a tensor with final dimension 1
    
    return Glist

def reconstruct(Glist):
    """
    Reconstruction of tensor from TT decomposition core matrices
    Input:
    Glist: list containing core matrices [G1,G2,...]
    Output:
    Anew: reconstructed tensor stored as numpy array
    """
    N = len(Glist)
    tensor = Glist[0] #initialise tensor
    for n in range(1, N):
        tensor = np.tensordot(tensor, Glist[n], axes=[-1, 0])
        #conduct (-1,0)-contraction on tensor and the next core.
        #-1 is the last index of tensor which matches with the first index of the core
    return np.squeeze(tensor) #remove first and last indicies as they are 1


def decompose2(A,Rlist):
    """
    Implementation of modified Algorithm 3 from KCSM with rank provided as input
    Input:
    A: tensor stored as numpy array 
    Rlist: list of values for rank, [R1,R2,...]
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """
    #same algorithm using input ranks
    Glist = []
    N = A.ndim
    Rlist.insert(0,1) #first rank is 1 by convention
    Z = unfold(A,0)
    for n in range(N-1):
        #delta truncated svd
        U, S, Vt = scipy.linalg.svd(Z, full_matrices=False)
        Rlist[n+1] = min(Rlist[n+1],len(S))
        #immediately truncate with known ranks
        U, S, Vt = U[:, :Rlist[n+1]], S[:Rlist[n+1]], Vt[:Rlist[n+1], :]
        G = U.reshape((Rlist[n], A.shape[n], Rlist[n+1]))
        Glist.append(G)
        Z = np.diag(S)@Vt
        Z = Z.reshape((Rlist[n+1]*A.shape[n+1], -1))
    Glist.append(Z.reshape(Rlist[-1], A.shape[-1], 1))
    return Glist


def compression_ratio(A, Glist):
    A_size = A.size
    A_compressed_size = sum(core.size for core in Glist) #number of floating point numbers needed to store
    return A_compressed_size / A_size #ratio

def part1(fig1=False):
    """
    Add code here for part 1, question 2 if needed
    """
    A = load_image()
    #Add code here
    if fig1:
        Glist2 = decompose2(A,[20,2]) #decompose into cores
        A_tt2 = reconstruct(Glist2) #reconstruct
        ratio_tt2 = compression_ratio(A,Glist2)
        err = residual_rel_error(Tensor(A),Tensor(A_tt2)) #calculate error
        Glist1 = decompose1(A,err)
        ratio_tt1 = compression_ratio(A,Glist1)
        A_tt1 = reconstruct(Glist1)
        coreranks1 = [Glist1[i].shape[-1] for i in range(A.ndim-1)] #prepare list of ranks
        coreranks2 = [Glist2[i].shape[-1] for i in range(A.ndim-1)]
        print(f'compression ratio for decompose1, decompose2:{ratio_tt1, ratio_tt2}')
        print(f'multilinear ranks for each decomposition:{coreranks1,coreranks2}')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Use imshow on each subplot
        ax1.imshow(A_tt1)
        ax1.set_title(f'Decompose 1. Error:{residual_rel_error(Tensor(A),Tensor(A_tt1)): .4g}')

        ax2.imshow(A_tt2)
        ax2.set_title(f'Decompose 2. Error:{err: .4g}')

        plt.show()
    return None #modify as needed


#-------------------------
# Code for Part 2
#-------------------------
def unfold(nptensor, mode):
    #simple unfolding function using default numpy reshaping ordering
    unfolded = np.reshape(np.moveaxis(nptensor,mode,0), (nptensor.shape[mode],-1))
    return unfolded

def fold(matrix, mode, shape):
    #fold using same reshaping ordering as unfold
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    tensor = np.moveaxis(np.reshape(matrix, full_shape), 0, mode)
    return tensor

def mode_n_product(tensor, matrix, mode):
    orig_shape = list(tensor.shape)
    new_shape = orig_shape
    new_shape[mode] = matrix.shape[0] #calculate shape of resulting tensor
    result = fold(np.dot(matrix, unfold(tensor, mode)), mode, tuple(new_shape)) #unfold along correct mode, matrix multiply does axis 0 to axis 1 contraction, reshape.
    return result

def decomposeHOSVD(X, Rlist):
    N = X.ndim
    fmat=[] #initialise
    G = X.copy() #does not change input
    for n in range(N):
        tensor = unfold(X,n)
        U, S, Vt = scipy.linalg.svd(tensor, full_matrices=False)
        U, S, Vt = U[:, :Rlist[n]], S[:Rlist[n]], Vt[:Rlist[n], :]
        fmat.append(U)
        G = mode_n_product(G, U.T, n) #construct core by successively projecting using the svd eigenvectors
    return fmat, G

def reconstructHOSVD(fmat, G):
    for i,factor in enumerate(fmat):
        G = mode_n_product(G,factor,i)
        #project back successively
    return G

def rrrim(A, r):
    N = A.ndim
    shape = list(A.shape)
    X = A.copy()
    components = [] #stores reduced components used to reconstruct original tensor
    evecs = [] #for successive projection and rank reduction as done in PCA
    for i in range(shape[2]):
        U,S,Vt = scipy.linalg.svd(X[:,:,i], full_matrices=False)
        U, S, Vt = U[:, :r], S[:r], Vt[:r, :] #truncate
        components.append(U.T@X[:,:,i]) #components are projected reduced slices
        evecs.append(U) #evecs for reconstruction
    return evecs, components

def reconstructrr(evecs, components): #projects each slice back to original tensor
    slices = []
    for i in range(len(components)):
        slice = evecs[i]@components[i]
        slices.append(slice)
    return np.stack(slices,2)

def calculate_psnr(A, A_compressed): #An error metric
    mse = np.mean((A - A_compressed)**2)
    if mse == 0:
        return float('inf')
    maxval = 1.0  # Assuming the data is normalized between 0 and 1
    psnr = 20 * np.log10(maxval / np.sqrt(mse)) #by definition of psnr
    return psnr

def test_method_tt(A,ranks):
    t1 = time.time()
    Glist = decompose2(A,ranks) #records decompositio time
    t2 = time.time()
    time_taken = t2-t1
    A_compressed = reconstruct(Glist) #returns reconstructed image
    psnr = calculate_psnr(A,A_compressed) #psnr
    ratio = compression_ratio(A,Glist) #compression ratio
    error = np.mean((A-A_compressed)**2) #error
    return psnr, ratio, error, time_taken, A_compressed

def test_method_hosvd(A,ranks): #as above
    t1 = time.time()
    fmat, G = decomposeHOSVD(A,ranks)
    t2 = time.time()
    time_taken = t2-t1
    A_compressed = reconstructHOSVD(fmat, G)
    psnr = calculate_psnr(A,A_compressed)
    ratio = compression_ratio(A,fmat+[G])
    error = np.mean((A-A_compressed)**2)
    return psnr, ratio, error, time_taken, A_compressed

def test_method_rr(A,ranks): #as above
    t1 = time.time()
    evecs, components = rrrim(A,ranks)
    t2 = time.time()
    time_taken = t2-t1
    A_compressed = reconstructrr(evecs, components)
    psnr = calculate_psnr(A,A_compressed)
    ratio = compression_ratio(A, evecs+components)
    error = np.mean((A-A_compressed)**2)
    return psnr, ratio, error, time_taken, A_compressed

def part2(on_image=False, on_video=False, temporal = False):
    """
    Add input/output as needed
    """
    if on_image: #returns analysis plots for three methods on image
        A = load_image()
        ranks = list(i**2 for i in range(1,15))
        psnr_tts = []
        ratio_tts = []
        error_tts = []
        times_tt = []
        psnr_hosvds = []
        ratio_hosvds = []
        error_hosvds = []
        times_hosvd = []
        psnr_rrs = []
        ratio_rrs = []
        error_rrs = []
        times_rr = []
        for rank in ranks:
            psnr_tt, ratio_tt, error_tt, time_taken_tt, A_tt = test_method_tt(A,[rank]*2)
            psnr_hosvd, ratio_hosvd, error_hosvd, time_taken_hosvd, A_hosvd = test_method_hosvd(A,[rank]*3)
            psnr_rr,ratio_rr,error_rr,time_taken_rr,A_rr = test_method_rr(A,rank)
            psnr_tts.append(psnr_tt)
            ratio_tts.append(ratio_tt)
            error_tts.append(error_tt)
            times_tt.append(time_taken_tt)
            psnr_hosvds.append(psnr_hosvd)
            ratio_hosvds.append(ratio_hosvd)
            error_hosvds.append(error_hosvd)
            times_hosvd.append(time_taken_hosvd)
            psnr_rrs.append(psnr_rr)
            ratio_rrs.append(ratio_rr)
            error_rrs.append(error_rr)
            times_rr.append(time_taken_rr)

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(12,6))

        ax1.plot(ratio_tts,error_tts, marker='o',label='tt')
        ax1.plot(ratio_hosvds,error_hosvds,marker='o',label='hosvd')
        ax1.plot(ratio_rrs,error_rrs,marker='o',label='rr')
        ax1.set_xlabel('compression ratio')
        ax1.set_ylabel('MSE')

        ax2.plot(ratio_tts, times_tt, marker='o',label='tt')
        ax2.plot(ratio_hosvds, times_hosvd, marker='o',label='hosvd')
        ax2.plot(ratio_rrs, times_rr, marker='o',label='rr')
        ax2.set_xlabel('Compression ratio')
        ax2.set_ylabel('Compress time')

        ax3.plot(psnr_tts, ratio_tts, marker='o',label='tt')
        ax3.plot(psnr_hosvds, ratio_hosvds, marker='o',label='hosvd')
        ax3.plot(psnr_rrs, ratio_rrs, marker='o',label='rr')
        ax3.set_xlabel('PSNR')
        ax3.set_ylabel('Compression ratio')

        ax4.plot(psnr_tts, times_tt, marker='o',label='tt')
        ax4.plot(psnr_hosvds, times_hosvd, marker='o',label='hosvd')
        ax4.plot(psnr_rrs, times_rr, marker='o',label='rr')
        ax4.set_xlabel('PSNR')
        ax4.set_ylabel('Compression time')

        ax1.legend()
        plt.show()

    if on_video or temporal: 
        B = video2numpy()
        ranktt = 130
        rankhosvd = 150
        X = B.copy()
        print('Compressing video, this may take a few minutes.')
        psnr_tt, ratio_tt, error_tt, time_taken_tt, B_tt = test_method_tt(X,[ranktt]*3)
        psnr_hosvd, ratio_hosvd, error_hosvd, time_taken_hosvd, B_hosvd = test_method_hosvd(X,[rankhosvd]*4)
        if on_video: #returns data for two methods on image
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.ravel()

            axs[0].bar(['TTSVD', 'HOSVD'], [psnr_tt, psnr_hosvd], color=['blue', 'purple'])
            axs[0].set_title('PSNR (dB)')
            axs[0].set_ylabel('PSNR (dB)')

            axs[1].bar(['TTSVD', 'HOSVD'], [ratio_tt, ratio_hosvd], color=['blue', 'purple'])
            axs[1].set_title('Compression Ratio')
            axs[1].set_ylabel('Compression Ratio')

            axs[2].bar(['TTSVD', 'HOSVD'], [error_tt, error_hosvd], color=['blue', 'purple'])
            axs[2].set_title('MSE')
            axs[2].set_ylabel('MSE')

            axs[3].bar(['TTSVD', 'HOSVD'], [time_taken_tt, time_taken_hosvd], color=['blue', 'purple'])
            axs[3].set_title('Time Taken (s)')
            axs[3].set_ylabel('Time (seconds)')

            plt.tight_layout()
            plt.show()
        if temporal: #plots MSE for each frame
            MSEs_tt = []
            MSEs_hosvd = []
            for i in range(X.shape[0]):
                MSE_tt = np.mean((B_tt[i]-X[i])**2)
                MSE_hosvd = np.mean((B_hosvd[i]-X[i])**2)
                MSEs_tt.append(MSE_tt)
                MSEs_hosvd.append(MSE_hosvd)
            plt.plot(range(191),MSEs_tt,label='tt')
            plt.plot(range(191),MSEs_hosvd,label='hosvd')
            plt.xlabel('Frame')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

    return None #modify as needed


def video2numpy(fname='project4.mp4'):
    """
    Convert mp4 video with filename fname into numpy array
    """
    import cv2
    cap = cv2.VideoCapture(fname)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    A = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, A[fc] = cap.read()
        fc += 1

    cap.release()
    
    return A[:,::2,::2,:].astype(float)/255 #Scales A to contain values between 0 and 1

def numpy2video(output_fname, A, fps=30):
    """
    Convert numpy array A into mp4 video and save as output_fname
    fps: frames per second.
    """
    import cv2
    video_array = A*255 #assumes A contains values between 0 and 1
    video_array  = video_array.astype('uint8')
    height, width, _ = video_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_fname, fourcc, fps, (width, height))

    for frame in video_array:
        out.write(frame)

    out.release()

    return None

#----------------------
if __name__=='__main__':
    pass
    out = part2(on_image=True,on_video=True,temporal=True) #Uncomment and modify as needed after completing part2
