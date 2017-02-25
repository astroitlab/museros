'''
/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This sample implements a separable convolution filter
 * of a 2D signal with a gaussian kernel.
 */

 Ported to pycuda by Andrew Wagner <awagner@illinois.edu>, June 2009.
'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import string

import scipy.cluster.vq as vq
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
npa = np.array

import numpy as np
import pylab as pl
import matplotlib
from matplotlib.ticker import NullFormatter
from matplotlib.widgets import Slider

import sys; sys.path.append('.')
#import pdb

import pylab
#from normal import Normal


# Pull out a bunch of stuff that was hard coded as pre-processor directives used
# by both the kernel and calling code.
KERNEL_RADIUS = 16  #Original: 8
UNROLL_INNER_LOOP = True
KERNEL_W = 2 * KERNEL_RADIUS + 1
ROW_TILE_W = 128
KERNEL_RADIUS_ALIGNED = 32   #Origin: 16
COLUMN_TILE_W = 16    #Original: 16
COLUMN_TILE_H = 48    #Original: 48
template = '''
//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)
////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS $KERNEL_RADIUS
#define KERNEL_W $KERNEL_W
__device__ __constant__ float d_Kernel_rows[KERNEL_W];
__device__ __constant__ float d_Kernel_columns[KERNEL_W];

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W  $ROW_TILE_W
#define KERNEL_RADIUS_ALIGNED  $KERNEL_RADIUS_ALIGNED

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W $COLUMN_TILE_W
#define COLUMN_TILE_H $COLUMN_TILE_H


////////////////////////////////////////////////////////////////////////////////
// Moving Average KERNEL
//     M - Rows, N: Columns
////////////////////////////////////////////////////////////////////////////////

__global__ void moving_average_1d_GPU(int *in, int *out, int M, int N, int RADIUS) {
    //#define GRID(x,y,W) ((x)+((y)*W))

    // Identify place on grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id  = idy+idx*N;
    int i = 0;
    if (idx < N - RADIUS) {
        out[id] = 0;
        for (i=0;i<RADIUS;i++)
            out[id] += in[id];
        out[id] /= RADIUS;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    //Row start index in d_Data[]
    const int          rowStart = IMUL(blockIdx.y, dataW);

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples
    //of half-warp size, rowStart + apronStartAligned is also a
    //multiple of half-warp size, thus having proper alignment
    //for coalesced d_Data[] read.
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] =
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tileStart + threadIdx.x;
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + k] * d_Kernel_rows[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template(
    'sum += data[smemPos + $k] * d_Kernel_rows[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

#print unrolledLoop
template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
template += '''
        d_Result[rowStart + writePos] = sum/(2*KERNEL_RADIUS);
        //d_Result[rowStart + writePos] = 128;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    //Data cache
    __shared__ float data[COLUMN_TILE_W *
    (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] =
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ?
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + IMUL(k, COLUMN_TILE_W)] *
            d_Kernel_columns[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template('sum += data[smemPos + IMUL($k, COLUMN_TILE_W)] * d_Kernel_columns[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

#print unrolledLoop
template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
template += '''
        d_Result[gmemPos] = sum/(2*KERNEL_RADIUS);
        //d_Result[gmemPos] = 128;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}
'''
template = string.Template(template)
code = template.substitute(KERNEL_RADIUS = KERNEL_RADIUS,
                           KERNEL_W = KERNEL_W,
                           COLUMN_TILE_H=COLUMN_TILE_H,
                           COLUMN_TILE_W=COLUMN_TILE_W,
                           ROW_TILE_W=ROW_TILE_W,
                           KERNEL_RADIUS_ALIGNED=KERNEL_RADIUS_ALIGNED)

module = SourceModule(code)
convolutionRowGPU = module.get_function('convolutionRowGPU')
convolutionColumnGPU = module.get_function('convolutionColumnGPU')
meanaverageGPU = module.get_function('moving_average_1d_GPU')

d_Kernel_rows = module.get_global('d_Kernel_rows')[0]
d_Kernel_columns = module.get_global('d_Kernel_columns')[0]

# Helper functions for computing alignment...
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

def iDivDown(a, b):
    # Round a / b to nearest lower integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a / b;

def iAlignUp(a, b):
    # Align a to nearest higher multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a - a % b + b) if (a % b != 0) else a

def iAlignDown(a, b):
    # Align a to nearest lower multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a - a % b

def gaussian_kernel(width = KERNEL_W, sigma = 64.0):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.ones(width) #numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)   #x = numpy.float32(x)
    sigma = numpy.float32(sigma)
    filterx = x #*x / (2 * sigma * sigma)

    #filterx = numpy.exp(-1 * filterx)
    #assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    # filterx /= filterx.sum()
    # x = numpy.linspace(-radius,  radius,  width)
    # x = numpy.float32(x)   #x = numpy.float32(x)
    # sigma = numpy.float32(sigma)
    # filterx = x*x / (2 * sigma * sigma)
    # filterx = numpy.exp(-1 * filterx)
    # assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    # filterx /= filterx.sum()
    return filterx

def derivative_of_gaussian_kernel(width = KERNEL_W, sigma = 4):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    # The derivative of a gaussian is really just a gaussian times x, up to scale.
    filterx = gaussian_kernel(width,  sigma)
    filterx *= x
    # Rescale so that filter returns derivative of 1 when applied to x:
    scale = (x * filterx).sum()
    filterx /= scale
    # Careful with sign; this will be uses as a ~convolution kernel, so should start positive, then go negative.
    filterx *= -1.0
    return filterx

def test_derivative_of_gaussian_kernel():
    width = 20
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filter * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'
    width = 19
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filterx * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'

def convolution_cuda(sourceImage,  filterx,  filtery):
    # Perform separable convolution on sourceImage using CUDA.
    # Operates on floating point images with row-major storage.
    destImage = sourceImage.copy()
    assert sourceImage.dtype == 'float32',  'source image must be float32'
    (imageHeight,  imageWidth) = sourceImage.shape
    assert filterx.shape == filtery.shape == (KERNEL_W, ) ,  'Kernel is compiled for a different kernel size! Try changing KERNEL_W'
    filterx = numpy.float32(filterx)
    filtery = numpy.float32(filtery)
    DATA_W = iAlignUp(imageWidth, 16);
    DATA_H = imageHeight;
    BYTES_PER_WORD = 4;  # 4 for float32
    DATA_SIZE = DATA_W * DATA_H * BYTES_PER_WORD;
    KERNEL_SIZE = KERNEL_W * BYTES_PER_WORD;
    # Prepare device arrays
    destImage_gpu = cuda.mem_alloc_like(destImage)
    sourceImage_gpu = cuda.mem_alloc_like(sourceImage)
    intermediateImage_gpu = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu, sourceImage)
    cuda.memcpy_htod(d_Kernel_rows,  filterx) # The kernel goes into constant memory via a symbol defined in the kernel
    cuda.memcpy_htod(d_Kernel_columns,  filtery)
    # Call the kernels for convolution in each direction.
    blockGridRows = (iDivUp(DATA_W, ROW_TILE_W), DATA_H)
    blockGridColumns = (iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H))
    threadBlockRows = (KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS, 1, 1)
    threadBlockColumns = (COLUMN_TILE_W, 8, 1)
    DATA_H = numpy.int32(DATA_H)
    DATA_W = numpy.int32(DATA_W)
    grid_rows = tuple([int(e) for e in blockGridRows])
    block_rows = tuple([int(e) for e in threadBlockRows])
    grid_cols = tuple([int(e) for e in blockGridColumns])
    block_cols = tuple([int(e) for e in threadBlockColumns])
    convolutionRowGPU(intermediateImage_gpu,  sourceImage_gpu,  DATA_W,  DATA_H,  grid=grid_rows,  block=block_rows)
    convolutionColumnGPU(destImage_gpu,  intermediateImage_gpu,  DATA_W,  DATA_H,  numpy.int32(COLUMN_TILE_W * threadBlockColumns[1]),  numpy.int32(DATA_W * threadBlockColumns[1]),  grid=grid_cols,  block=block_cols)

    # Pull the data back from the GPU.
    cuda.memcpy_dtoh(destImage,  destImage_gpu)
    return destImage

def mean_average_1d(in_image, out_image, height, width, radius):
    blocksize2D = (8, 16, 1)
    gridsize2D = (iDivUp(width, blocksize2D[0]), iDivUp(height, blocksize2D[1]))

    meanaverageGPU(in_image, out_image, numpy.int32(height), numpy.int32(width), numpy.int32(radius), grid=gridsize2D,  block=blocksize2D)
    return out_image

def test_convolution_cuda():
    # Test the convolution kernel.
    # Generate or load a test image
    original = numpy.random.rand(1, 2048) * 255
    original = numpy.float32(original)
    # You probably want to display the image using the tool of your choice here.
    filterx = gaussian_kernel()
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filterx)
    # You probably wand to display the result image using the tool of your choice here.
    print 'Done running the convolution kernel!'


class GMM(object):

    def __init__(self, dim = None, ncomps = None, data = None,  method = None, filename = None, params = None):

        if not filename is None:  # load from file
            self.load_model(filename)

        elif not params is None: # initialize with parameters directly
            self.comps = params['comps']
            self.ncomps = params['ncomps']
            self.dim = params['dim']
            self.priors = params['priors']

        elif not data is None: # initialize from data

            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps
            self.comps = []

            if method is "uniform":
                # uniformly assign data points to components then estimate the parameters
                npr.shuffle(data)
                n = len(data)
                s = n / ncomps
                for i in range(ncomps):
                    self.comps.append(Normal(dim, data = data[i * s: (i+1) * s]))

                self.priors = np.ones(ncomps, dtype = "double") / ncomps

            elif method is "random":
                # choose ncomp points from data randomly then estimate the parameters
                mus = pr.sample(data,ncomps)
                clusters = [[] for i in range(ncomps)]
                for d in data:
                    i = np.argmin([la.norm(d - m) for m in mus])
                    clusters[i].append(d)

                for i in range(ncomps):
                    print mus[i], clusters[i]
                    self.comps.append(Normal(dim, mu = mus[i], sigma = np.cov(clusters[i], rowvar=0)))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            elif method is "kmeans":
                # use kmeans to initialize the parameters
                (centroids, labels) = vq.kmeans2(data, ncomps, minit="points", iter=100)
                clusters = [[] for i in range(ncomps)]
                for (l,d) in zip(labels,data):
                    clusters[l].append(d)

                # will end up recomputing the cluster centers
                for cluster in clusters:
                    self.comps.append(Normal(dim, data = cluster))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            else:
                raise ValueError, "Unknown method type!"

        else:

            # these need to be defined
            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps

            self.comps = []

            for i in range(ncomps):
                self.comps.append(Normal(dim))

            self.priors = np.ones(ncomps,dtype='double') / ncomps

    def __str__(self):
        res = "%d" % self.dim
        res += "\n%s" % str(self.priors)
        for comp in self.comps:
            res += "\n%s" % str(comp)
        return res

    def save_model(self):
        pass

    def load_model(self):
        pass

    def mean(self):
        return np.sum([self.priors[i] * self.comps[i].mean() for i in range(self.ncomps)], axis=0)

    def covariance(self): # computed using Dan's method
        m = self.mean()
        s = -np.outer(m,m)

        for i in range(self.ncomps):
            cm = self.comps[i].mean()
            cvar = self.comps[i].covariance()
            s += self.priors[i] * (np.outer(cm,cm) + cvar)

        return s

    def pdf(self, x):
        responses = [comp.pdf(x) for comp in self.comps]
        return np.dot(self.priors, responses)

    def condition(self, indices, x):
        """
        Create a new GMM conditioned on data x at indices.
        """
        condition_comps = []
        marginal_comps = []

        for comp in self.comps:
            condition_comps.append(comp.condition(indices, x))
            marginal_comps.append(comp.marginalize(indices))

        new_priors = []
        for (i,prior) in enumerate(self.priors):
            new_priors.append(prior * marginal_comps[i].pdf(x))
        new_priors = npa(new_priors) / np.sum(new_priors)

        params = {'ncomps' : self.ncomps, 'comps' : condition_comps,
                  'priors' : new_priors, 'dim' : marginal_comps[0].dim}

        return GMM(params = params)

    def em(self, data, nsteps = 100):

        k = self.ncomps
        d = self.dim
        n = len(data)

        for l in range(nsteps):

            # E step

            responses = np.zeros((k,n))

            for j in range(n):
                for i in range(k):
                    responses[i,j] = self.priors[i] * self.comps[i].pdf(data[j])

            responses = responses / np.sum(responses,axis=0) # normalize the weights

            # M step

            N = np.sum(responses,axis=1)

            for i in range(k):
                mu = np.dot(responses[i,:],data) / N[i]
                sigma = np.zeros((d,d))

                for j in range(n):
                   sigma += responses[i,j] * np.outer(data[j,:] - mu, data[j,:] - mu)

                sigma = sigma / N[i]

                self.comps[i].update(mu,sigma) # update the normal with new parameters
                self.priors[i] = N[i] / np.sum(N) # normalize the new priors


npa = np.array
ix  = np.ix_ # urgh - sometimes numpy is ugly!

class Normal(object):
    """
    A class for storing the parameters of a multivariate normal
    distribution. Supports evaluation, sampling, conditioning and
    marginalization.
    """

    def __init__(self, dim, mu = None, sigma = None, data = None,
                 parent = None, cond = None, margin = None):
        """
        Initialize a normal distribution.

        Parameters
        ----------
        dim : int
            Number of dimensions (e.g. number of components in the mu parameter).
        mu : array, optional
            The mean of the normal distribution.
        sigma : array, optional
            The covariance matrix of the normal distribution.
        data : array, optional
            If provided, the parameters of the distribution will be estimated from the data. Rows are observations, columns are components.
        parent : Normal, optional
            A reference to a parent distribution that was marginalized or conditioned.
        cond : dict, optional
            A dict of parameters describing how the parent distribution was conditioned.
        margin : dict, optional
            A dict of parameters describing how the parent distribution was marginalized.

        Examples
        --------
        >>> x = Normal(2,mu = np.array([0.1,0.7]), sigma = np.array([[ 0.6,  0.4], [ 0.4,  0.6]]))
        >>> print x
        [ 0.1  0.7]
        [[ 0.6  0.4]
        [ 0.4  0.6]]

        To condition on a value (and index):

        >>> condx = x.condition([0],0.1)
        >>> print condx
        [ 0.7]
        [[ 0.33333333]]

        """

        self.dim = dim # full data dimension

        if not mu is None  and not sigma is None:
            pass
        elif not data is None:
            # estimate the parameters from data - rows are samples, cols are variables
            mu, sigma = self.estimate(data)
        else:
            # generate random means
            mu = npr.randn(dim)
            sigma = np.eye(dim)

        self.cond = cond
        self.margin = margin
        self.parent = parent

        self.update(npa(mu),npa(sigma))


    def update(self, mu, sigma):
        """
        Update the distribution with new parameters.

        Parameters
        ----------
        mu : array
            The new mean parameters.
        sigma : array
            The new covariance matrix.

        Example
        -------

        >>> x = Normal(2,mu = np.array([0.1,0.7]), sigma = np.array([[ 0.6,  0.4], [ 0.4,  0.6]]))
        >>> print x
        [ 0.1  0.7]
        [[ 0.6  0.4]
        [ 0.4  0.6]]

        >>> x.update(np.array([0.0,0.0]), x.E)
        >>> print x
        [ 0.0  0.0]
        [[ 0.6  0.4]
        [ 0.4  0.6]]
        """

        self.mu = mu
        self.E = sigma

        det = None
        if self.dim == 1:
            self.A = 1.0 / self.E
            det = np.fabs(self.E[0])
        else:
            self.A = la.inv(self.E) # precision matrix
            det = np.fabs(la.det(self.E))

        self.factor = (2.0 * np.pi)**(self.dim / 2.0) * (det)**(0.5)

    def __str__(self):
        return "%s\n%s" % (str(self.mu), str(self.E))

    def mean(self):
        return self.mu

    def covariance(self):
        return self.E

    def pdf(self, x):
        dx = x - self.mu
        A = self.A
        fE = self.factor

        return np.exp(-0.5 * np.dot(np.dot(dx,A),dx)) / fE

    def pdf_mesh(self, x, y):
        # for 2d meshgrids
        # use matplotlib.mlab.bivariate_normal -- faster (vectorized)

        z = np.zeros((len(y),len(x)))

        for (i,v) in enumerate(x):
            for (j,w) in enumerate(y):
                z[j,i] = self.pdf([v,w])

        return z

    def simulate(self, ndata = 100):
        """
        Draw pts from the distribution.
        """
        return npr.multivariate_normal(self.mu, self.E, ndata)

    def estimate(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.cov(data, rowvar=0)
        return mu, sigma

    def marginalize(self, indices):
        """
        Creates a new marginal normal distribution for ''indices''.
        """
        indices = npa(indices)
        return Normal(len(indices), mu = self.mu[indices], sigma = self.E[ix(indices,indices)], margin = {'indices' : indices}, parent = self)

    def condition(self, indices, x):
        """
        Creates a new normal distribution conditioned on the data x at indices.
        """

        idim = indices
        odim = npa([i for i in range(self.dim) if not i in indices])

        Aaa = self.A[ix(odim,odim)]
        Aab = self.A[ix(odim,idim)]
        iAaa = None
        det = None

        if len(odim) == 1: # linalg does not handle d1 arrays
            iAaa = 1.0 / Aaa
            det = np.fabs(iAaa[0])
        else:
            iAaa = la.inv(Aaa)
            det = np.fabs(la.det(iAaa))

        # compute the new mu
        premu = np.dot(iAaa, Aab)

        mub = self.mu[idim]
        mua = self.mu[odim]
        new_mu = mua - np.dot(premu, (x - mub))

        new_E = iAaa
        return Normal(len(odim), mu = new_mu, sigma = new_E,
                      cond = {'data' : x, 'indices' : indices},
                      parent = self)


def shownormal(data,gmm):

    xnorm = data[:,0]
    ynorm = data[:,1]

    # Plot the normalized faithful data points.
    fig = pylab.figure(num = 1, figsize=(4,4))
    axes = fig.add_subplot(111)
    axes.plot(xnorm,ynorm, '+')

    # Plot the ellipses representing the principle components of the normals.
    for comp in gmm.comps:
        comp.patch(axes)

    pylab.draw()
    pylab.show()

def draw2dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 2d normal pdf.
    """
    # create a meshgrid centered at mu that takes into account the variance in x and y
    delta = 0.025

    lower_xlim = norm.mu[0] - (2.0 * norm.E[0,0])
    upper_xlim = norm.mu[0] + (2.0 * norm.E[0,0])
    lower_ylim = norm.mu[1] - (2.0 * norm.E[1,1])
    upper_ylim = norm.mu[1] + (2.0 * norm.E[1,1])

    x = np.arange(lower_xlim, upper_xlim, (upper_xlim - lower_xlim)/2000.)
    y = np.arange(lower_ylim, upper_ylim, (upper_ylim - lower_ylim)/2000.)

    X,Y = np.meshgrid(x,y)

    # remember sqrts!
    Z = matplotlib.mlab.bivariate_normal(X, Y, sigmax=np.sqrt(norm.E[0,0]), sigmay=np.sqrt(norm.E[1,1]), mux=norm.mu[0], muy=norm.mu[1], sigmaxy=norm.E[0,1])

    minlim = min(lower_xlim, lower_ylim)
    maxlim = max(upper_xlim, upper_ylim)

    # Plot the normalized faithful data points.
    if not axes:
        fig = pl.figure(num = 1, figsize=(4,4))
        pl.contour(X,Y,Z)
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)
    else:
        axes.contour(X,Y,Z)
        #axes.set_xlim(minlim,maxlim)
        #axes.set_ylim(minlim,maxlim)

    if show:
        pl.show()

def evalpdf(norm):
    delta = 1
    mu = norm.mu[0]
    sigma = norm.E[1,1]
    lower_xlim = mu - (2.0 * sigma)
    upper_xlim = mu + (2.0 * sigma)
    print "MU, SIGMA =" , mu, sigma, norm.mu[1] #np.sqrt(sigma)
    x = np.arange(0,2000,1) #lower_xlim,upper_xlim, delta)
    y = [norm.mu[1]*1./(np.sqrt(2*np.pi*np.sqrt(sigma)))*np.exp(-(i-mu)**2/(2.*sigma)) for i in x] #matplotlib.mlab.normpdf(x, mu, np.sqrt(sigma))


    return x,y

def draw1dnormal(norm, show = False, axes = None):
    """
    Just draw a simple 1d normal pdf. Used for plotting the conditionals in simple test cases.
    """
    x,y = evalpdf(norm)
    if axes is None:
        pl.plot(x,y)
    else:
        return axes.plot(y,x)

    if show:
        pl.show()

def draw2d1dnormal(norm, cnorm, show = False):

    pl.figure(1, figsize=(8,8))

    nullfmt = NullFormatter()

    rect_2d = [0.1, 0.1, 0.65, 0.65]
    rect_1d = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]
    ax2d = pl.axes(rect_2d)
    ax1d = pl.axes(rect_1d)
    ax1d.xaxis.set_major_formatter(nullfmt)
    ax1d.yaxis.set_major_formatter(nullfmt)
    draw2dnormal(norm, axes = ax2d)
    draw1dnormal(cnorm, axes = ax1d)
    y = ax2d.get_ylim()
    x = [cnorm.cond['data'], cnorm.cond['data']]
    ax2d.plot(x,y)


def draw_slider_demo(norm):

    fig = pl.figure(1, figsize=(8,8))

    nullfmt = NullFormatter()

    cnorm = norm.condition([0],2.0)

    rect_slide = [0.1, 0.85, 0.65 + 0.1, 0.05]
    rect_2d = [0.1, 0.1, 0.65, 0.65]
    rect_1d = [0.1 + 0.65 + 0.02, 0.1, 0.2, 0.65]
    ax2d = pl.axes(rect_2d)
    ax1d = pl.axes(rect_1d)
    ax1d.xaxis.set_major_formatter(nullfmt)
    ax1d.yaxis.set_major_formatter(nullfmt)
    axslide = pl.axes(rect_slide)
    slider = Slider(axslide, 'Cond', -4.0,4.0,valinit=2.0)

    draw2dnormal(norm, axes = ax2d)
    l2, = draw1dnormal(cnorm, axes = ax1d)

    y = ax2d.get_ylim()
    x = [cnorm.cond['data'], cnorm.cond['data']]
    l1, = ax2d.plot(x,y)

    def update(val):
        cnorm = norm.condition([0],val)
        x = [cnorm.cond['data'], cnorm.cond['data']]
        l1.set_xdata(x)
        x,y = evalpdf(cnorm)
        #print y
        l2.set_xdata(y)
        l2.set_ydata(x)
        pl.draw()


    slider.on_changed(update)

    return slider

if __name__ == '__main__':

    """
    Tests for gmm module.
    """


    # x = npr.randn(20, 2)

    # print "No data"
    # gmm = GMM(2,1,2) # possibly also broken
    # print gmm

    # print "Uniform"
    # gmm = GMM(2,1,2,data = x, method = "uniform")
    # print gmm

    # print "Random"
    # gmm = GMM(2,1,2,data = x, method = "random") # broken
    # print gmm

    # print "Kmeans"
    # gmm = GMM(2,1,2,data = x, method = "kmeans") # possibly broken
    # print gmm


    x = np.arange(-10,30)
    #y = x ** 2 + npr.randn(20)
    y = x + npr.randn(40) # simple linear function
    #y = np.sin(x) + npr.randn(20)
    data = np.vstack([x,y]).T


    gmm = GMM(dim = 2, ncomps = 4,data = data, method = "random")
    print gmm
    shownormal(data,gmm)

    gmm.em(data,nsteps=1000)
    shownormal(data,gmm)
    print gmm
    ngmm = gmm.condition([0],[-3])
    print ngmm.mean()
    print ngmm.covariance()
