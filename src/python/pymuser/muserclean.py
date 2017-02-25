import numpy as np
import time, pdb, sys, pyfits
import math
import os
import sys

import matplotlib
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import skcuda.fft as fft
import skcuda.misc as misc
import gaussfitter as gf
# import muserfilter as mfilter
import logging
import gausspeak as peakutils

# Initialize the CUDA device
# Elementwise stuff
from pycuda.elementwise import ElementwiseKernel
from pycuda import cumath
from scipy.ndimage.filters import gaussian_filter
from slope import get_peaks
from  datetime import *
from musersun import *
from muserdraw import *
from muserfits import *
import musercuda as mc
from muserfilter import *
from muserant import *
from muserfits import *
from musersundisk import *

# from sklearn.mixture import GMM

from argparse import *

logger = logging.getLogger('muser')

GRID = lambda x, y, W: ((x) + ((y) * W))

IGRIDX = lambda tid, W: tid % W
IGRIDY = lambda tid, W: int(tid) / int(W)

MATRIX_SIZE = 4
TILE_SIZE = 2
BLOCK_SIZE = TILE_SIZE

SUN_DISK = 16.1164

class MuserClean:
    ######################
    # CUDA kernels


    def __init__(self):

        self.muser_draw = MuserDraw()
        self.muser_fits = MuserFits()
        self.muser_ant = MuserAntenna()
        self.muser_fits = MuserFits()

        # -------------------
        # Gridding kernels
        # -------------------
        self.sub_beam_kernel = self.cuda_compile(mc.sub_beam_kernel_source, "sub_beam_kernel")
        self.sub_steer_kernel = self.cuda_compile(mc.sub_steer_kernel_source, "sub_steer_kernel")
        self.sub_histogram_kernel = self.cuda_compile(mc.histogram_kernel_source, "sub_histogram_kernel")
        self.sub_mean_average_kernel = self.cuda_compile(mc.filter_kernel_source, "sub_mean_average_kernel")
        self.sub_matrix_mul_kernel = self.cuda_compile(mc.matrix_mul_kernel_source, "sub_matrix_mul_kernel")
        self.sub_dot_mul_kernel = self.cuda_compile(mc.dot_mul_kernel_source, "sub_dot_mul_kernel")
        self.sub_cycle_shift_kernel = self.cuda_compile(mc.cycle_shift_kernel_source, "sub_cycle_shift_kernel")
        self.sub_cuda_cyclic_shift_kernel = self.cuda_compile(mc.sub_cuda_cyclic_shift_kernel_source,
                                                              "sub_cuda_cyclic_shift_kernel")
        self.sub_rotate_image_kernel = self.cuda_compile(mc.sub_rotation_kernel_source, "sub_rotate_image_kernel")
        self.find_max_kernel = self.cuda_compile(mc.find_max_kernel_source, "find_max_kernel")

        self.module = nvcc.SourceModule(mc.clean_code)
        self.gridVis_wBM_kernel = self.module.get_function("gridVis_wBM_kernel")
        self.shiftGrid_kernel = self.module.get_function("shiftGrid_kernel")
        self.nrmGrid_kernel = self.module.get_function("nrmGrid_kernel")
        self.wgtGrid_kernel = self.module.get_function("wgtGrid_kernel")
        self.dblGrid_kernel = self.module.get_function("dblGrid_kernel")
        self.corrGrid_kernel = self.module.get_function("corrGrid_kernel")
        self.nrmBeam_kernel = self.module.get_function("nrmBeam_kernel")
        self.trimIm_kernel = self.module.get_function("trimIm_kernel")
        self.trimIm_kernel_2 = self.module.get_function("trimIm_kernel_2")
        self.trimDisk_kernel = self.module.get_function("trimDisk_kernel")
        self.trimDisk_kernel_2 = self.module.get_function("trimDisk_kernel_2")

        self.sub_image_kernel = self.module.get_function("sub_image_kernel")
        self.add_image_kernel = self.module.get_function("add_image_kernel")
        self.add_flat_kernel = self.module.get_function("add_flat_kernel")
        self.add_clean_kernel = self.module.get_function("add_clean_kernel")
        self.trim_float2_kernel = self.module.get_function("trim_float2_kernel")
        self.copyIm_kernel = self.module.get_function("copyIm_kernel")
        self.copyRIm_kernel = self.module.get_function("copyRIm_kernel")
        self.sun_disk_kernel = self.module.get_function("sun_disk_kernel")
        self.diskGrid_kernel = self.module.get_function("diskGrid_kernel")
        self.copyContour_kernel = self.module.get_function("copyContour_kernel")

        self.trim_float_image_kernel = self.module.get_function("trim_float_image_kernel")
        self.copy_float_kernel = self.module.get_function("copy_float_kernel")


    def cuda_compile(self, source_string, function_name):
        logger.debug('Compiling a CUDA kernel...')
        # Compile the CUDA Kernel at runtime
        source_module = nvcc.SourceModule(source_string)
        # Return a handle to the compiled CUDA kernel
        return source_module.get_function(function_name)

    def calc_gpu_thread(self, nx, imsize, gcount):
        self.blocksize_2D = (8, 16, 1)
        self.gridsize_2D = (
        np.int(np.ceil(1. * nx / self.blocksize_2D[0])), np.int(np.ceil(1. * nx / self.blocksize_2D[1])))
        self.blocksize_F2D = (16, 16, 1)
        self.gridsize_F2D = (
        np.int(np.ceil(1. * imsize / self.blocksize_F2D[0])), np.int(np.ceil(1. * imsize / self.blocksize_F2D[1])))
        self.blocksize_1D = (256, 1, 1)
        self.gridsize_1D = (np.int(np.ceil(1. * gcount / self.blocksize_1D[0])), 1)

    def iDivUp(self, a, b):
        # Round a / b to nearest higher integer value
        a = int(a)
        b = int(b)
        return int(a / b + 1) if (a % b != 0) else int(a / b)

    def convolution(self, a, b, plan):
        height, width = np.shape(a)
        blocksize = (8, 16, 1)
        gridsize = (np.int(np.ceil(1. * height / blocksize[0])), np.int(np.ceil(1. * width / blocksize[1])))

        max = gpu.max(a).get()
        print "MAX1:", max


        d_a =   gpu.zeros((np.int(height), np.int(width)), np.complex64)
        res = gpu.zeros((np.int(height), np.int(width)), np.float32)
        d_b = gpu.zeros_like(d_a)
        d_c = gpu.zeros_like(d_a)

        #plan = fft.Plan((np.int(height), np.int(width)), np.complex64, np.complex64)
        self.copyRIm_kernel(a, d_b, np.int32(height), block=blocksize, grid=gridsize)
        self.copyRIm_kernel(b, d_c, np.int32(height), block=blocksize, grid=gridsize)

        fft.fft(d_b, d_a, plan)
        fft.fft(d_c, d_b, plan)


        self.sub_dot_mul_kernel(d_a, d_b, d_c, np.int32(height), np.int32(width), block=blocksize,grid=gridsize)

        fft.fft(d_c, d_a, plan)
        self.shiftGrid_kernel(d_a, d_c, np.int32(height), block=blocksize,grid=gridsize)

        self.copyIm_kernel(d_c, res, np.int32(height), block=blocksize, grid=gridsize)

        max2 = gpu.max(res).get()

        bmax1 = np.float32(max*1.0 / max2)
        print "max2:", max2, bmax1
        self.nrmBeam_kernel(res, bmax1, np.int32(self.imsize), block=blocksize, grid=gridsize)
        return res

    ######################
    # Gridding functions
    ######################

    def spheroid(self, eta, m, alpha):
        """
        Calculates spheriodal wave functions. See Schwab 1984 for details.
        This implementation follows MIRIAD's grid.for subroutine.
        """

        twoalp = 2 * alpha
        if np.abs(eta) > 1:
            logger.debug('bad eta value!')
        if (twoalp < 1 or twoalp > 4):
            logger.debug('bad alpha value!')
        if (m < 4 or m > 8):
            logger.debug('bad width value!')

        etalim = np.float32([1., 1., 0.75, 0.775, 0.775])
        nnum = np.int8([5, 7, 5, 5, 6])
        ndenom = np.int8([3, 2, 3, 3, 3])
        p = np.float32(
                [
                    [[5.613913E-2, -3.019847E-1, 6.256387E-1,
                      -6.324887E-1, 3.303194E-1, 0.0, 0.0],
                     [6.843713E-2, -3.342119E-1, 6.302307E-1,
                      -5.829747E-1, 2.765700E-1, 0.0, 0.0],
                     [8.203343E-2, -3.644705E-1, 6.278660E-1,
                      -5.335581E-1, 2.312756E-1, 0.0, 0.0],
                     [9.675562E-2, -3.922489E-1, 6.197133E-1,
                      -4.857470E-1, 1.934013E-1, 0.0, 0.0],
                     [1.124069E-1, -4.172349E-1, 6.069622E-1,
                      -4.405326E-1, 1.618978E-1, 0.0, 0.0]
                     ],
                    [[8.531865E-4, -1.616105E-2, 6.888533E-2,
                      -1.109391E-1, 7.747182E-2, 0.0, 0.0],
                     [2.060760E-3, -2.558954E-2, 8.595213E-2,
                      -1.170228E-1, 7.094106E-2, 0.0, 0.0],
                     [4.028559E-3, -3.697768E-2, 1.021332E-1,
                      -1.201436E-1, 6.412774E-2, 0.0, 0.0],
                     [6.887946E-3, -4.994202E-2, 1.168451E-1,
                      -1.207733E-1, 5.744210E-2, 0.0, 0.0],
                     [1.071895E-2, -6.404749E-2, 1.297386E-1,
                      -1.194208E-1, 5.112822E-2, 0.0, 0.0]
                     ]
                ])
        q = np.float32(
                [
                    [[1., 9.077644E-1, 2.535284E-1],
                     [1., 8.626056E-1, 2.291400E-1],
                     [1., 8.212018E-1, 2.078043E-1],
                     [1., 7.831755E-1, 1.890848E-1],
                     [1., 7.481828E-1, 1.726085E-1]
                     ],
                    [[1., 1.101270, 3.858544E-1],
                     [1., 1.025431, 3.337648E-1],
                     [1., 9.599102E-1, 2.918724E-1],
                     [1., 9.025276E-1, 2.575337E-1],
                     [1., 8.517470E-1, 2.289667E-1]
                     ]
                ])

        i = m - 4
        if (np.abs(eta) - etalim[int(i)]>0.):
            ip = 1
            x = eta * eta - 1
        else:
            ip = 0
            x = eta * eta - etalim[int(i)] * etalim[int(i)]
            # numerator via Horner's rule
        mnp = nnum[int(i)] - 1
        num = p[int(ip), int(twoalp), int(mnp)]
        for j in np.arange(mnp):
            num = num * x + p[int(ip), int(twoalp), int(mnp - 1 - j)]
            # denominator via Horner's rule
        nq = ndenom[int(i)] - 1
        denom = q[int(ip), int(twoalp), int(nq)]
        for j in np.arange(nq):
            denom = denom * x + q[int(ip), int(twoalp), int(nq - 1 - j)]

        return np.float32(num / denom)

    def gcf(self, n, width):
        """
        Create table with spheroidal gridding function, C
        This implementation follows MIRIAD's grid.for subroutine.
        """
        alpha = 1.
        j = 2 * alpha
        p = 0.5 * j
        phi = np.zeros(int(n), dtype=np.float32)
        for i in np.arange(n):
            x = np.float32(2 * i - (n - 1)) / (n - 1)
            phi[int(i)] = (np.sqrt(1 - x * x) ** j) * self.spheroid(x, width, p)
        return phi

    def corrfun(self, n, width):
        """
        Create gridding correction function, c
        This implementation follows MIRIAD's grid.for subroutine.
        """
        alpha = 1.
        dx = 2. / n
        i0 = n / 2 + 1
        phi = np.zeros(n, dtype=np.float32)
        for i in np.arange(n):
            x = (i - i0 + 1) * dx
            phi[i] = self.spheroid(x, width, alpha)
        return phi

    def centroid(self, data):
        h, w = np.shape(data)
        x = np.arange(0, w) - w // 2
        y = np.arange(0, h) - h // 2

        X, Y = np.meshgrid(x, y)

        cx = np.sum(X * data) / np.sum(data)
        cy = np.sum(Y * data) / np.sum(data)

        return cx, cy

    def sub_sun_disk_offset(self, A, B):
        # import numpy
        # return np.fft.irfft2(np.fft.rfft2(A) * np.fft.rfft2(B, A.shape))


        nx = A.shape[0]
        plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)

        self.blocksize_2D = (8, 16, 1)
        self.gridsize_2D = (
        np.int(np.ceil(1. * nx / self.blocksize_2D[0])), np.int(np.ceil(1. * nx / self.blocksize_2D[1])))

        d_af = gpu.to_gpu(A)
        d_bf = gpu.to_gpu(B)

        d_grd = gpu.zeros((np.int(nx), np.int(nx)), np.complex64)
        d_a = gpu.zeros_like(d_grd)
        d_b = gpu.zeros_like(d_grd)
        d_am = gpu.zeros_like(d_grd)
        d_bm = gpu.zeros_like(d_grd)
        d_c = gpu.zeros_like(d_grd)
        d_cm = gpu.zeros_like(d_grd)
        d_im = gpu.zeros((np.int(nx), np.int(nx)), np.float32)

        self.copyRIm_kernel(d_af, d_a, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        self.copyRIm_kernel(d_bf, d_b, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        fft.fft(d_a, d_am, plan)
        # self.shiftGrid_kernel(d_am, d_a, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        fft.fft(d_b, d_bm, plan)
        # self.shiftGrid_kernel(d_bm, d_b, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        self.sub_dot_mul_kernel(d_am, d_bm, d_cm, np.int32(nx), np.int32(nx), block=self.blocksize_2D,
                                grid=self.gridsize_2D)
        # self.shiftGrid_kernel(d_c, d_cm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        fft.fft(d_cm, d_c, plan)
        self.shiftGrid_kernel(d_c, d_cm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        self.copyIm_kernel(d_cm, d_im, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        return d_im

    def flag(self, x_offset, y_offset):

        h_uu = np.float32(self.h_uu.ravel())
        h_vv = np.float32(self.h_vv.ravel())
        h_rere = np.float32(self.h_rere.ravel())
        h_imim = np.float32(self.h_imim.ravel())
        gcount = np.int32(np.size(h_uu))

        blen = 0
        bl_order = np.ndarray(shape=(self.baseline_number, 2), dtype=int)
        good = []

        if self.baseline_number == 780:  # MUSER-I
            antennas = 40
        else:
            antennas = 60

        if (x_offset != 0 or y_offset != 0):
            logger.debug("Phase correction for moving")
            logger.debug("Frequence: %f" % self.current_freq)
            for baseline in range(0, self.baseline_number):
                A = math.sqrt(
                    h_rere[baseline] * h_rere[baseline] +
                    h_imim[baseline] * h_imim[baseline])
                phai_sun = math.atan2(h_imim[baseline], h_rere[baseline])
                phai = phai_sun - 2 * np.pi * (h_uu[baseline] * x_offset + h_vv[baseline] * y_offset)
                h_rere[baseline] = A * math.cos(phai)
                h_imim[baseline] = A * math.sin(phai)

        for border1 in range(0, antennas - 1):
            for border2 in range(border1 + 1, antennas):
                bl_order[blen][0] = border1
                bl_order[blen][1] = border2
                blen = blen + 1

        self.h_u = []
        self.h_v = []
        self.h_re = []
        self.h_im = []
        for blen in range(0, self.baseline_number):
            if (bl_order[blen][0] not in self.Flag_Ant) and (bl_order[blen][1] not in self.Flag_Ant):
                good.append(blen)
                self.h_u.append(h_uu[blen])
                self.h_v.append(h_vv[blen])
                self.h_re.append(h_rere[blen])
                self.h_im.append(h_imim[blen])


    def cuda_gridvis(self, x_offset, y_offset, fullsize=1, preclean=0):
        """
        Grid the visibilities parallelized by pixel.
        References:
          - Chapter 10 in "Interferometry and Synthesis in Radio Astronomy"
              by Thompson, Moran, & Swenson
          - Daniel Brigg's PhD Thesis: http://www.aoc.nrao.edu/dissertations/dbriggs/

        If the size of the image is 1024x1024, the plan should be at least 1024*1.414 (about 25 degrees' rotation)
        And to satisfy the requirements of CLEAN, the dirty image should be 1024* 2.828
        """
        logger.debug("Gridding the visibilities")
        t_start = time.time()

        # f = pyfits.open(settings['vfile'])

        # unpack parameters

        nx = np.int32(2 * self.imsize)
        noff = np.int32((nx - self.imsize) / 2)

        ## constants

        arc2rad = np.float32(np.pi / 180. / 3600.)
        self.du = np.float32(1. / (arc2rad * self.cell)) / (self.imsize * 2.)
        logger.debug("1 Pixel DU  = %f" % self.du)

        # Flagging
        self.flag(x_offset, y_offset)
        gcount = np.int32(np.size(self.h_u))

        d_u = gpu.to_gpu(np.array(self.h_u, dtype='float32'))
        d_v = gpu.to_gpu(np.array(self.h_v, dtype='float32'))
        d_re = gpu.to_gpu(np.array(self.h_re, dtype='float32'))
        d_im = gpu.to_gpu(np.array(self.h_im, dtype='float32'))
        d_cnt = gpu.zeros((np.int(nx), np.int(nx)), np.int32)
        d_grd = gpu.zeros((np.int(nx), np.int(nx)), np.complex64)
        d_sample_tmp =   gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.complex64)
        d_sample_half =   gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.complex64)

        d_ngrd = gpu.zeros_like(d_grd)
        d_sample=gpu.zeros_like(d_grd)
        d_grd = gpu.zeros((np.int(nx), np.int(nx)), np.complex64)

        d_bm = gpu.zeros_like(d_grd)
        d_nbm = gpu.zeros_like(d_grd)
        d_cbm = gpu.zeros_like(d_grd)

        d_fbm = gpu.zeros((np.int(nx), np.int(nx)), np.float32)
        d_fim = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)
        d_dim = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)

        d_sun_disk = gpu.zeros_like(d_grd)
        d_fdisk = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)

        plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)

        ## define kernel parameters
        self.calc_gpu_thread(nx, self.imsize, gcount)

        # ------------------------
        # make gridding kernels
        # ------------------------
        ## make spheroidal convolution kernel (don't mess with these!)
        width = 6.
        ngcf = 24.
        h_cgf = self.gcf(ngcf, width)

        ## make grid correction
        h_corr = self.corrfun(nx, width)
        d_cgf = self.module.get_global('cgf')[0]

        d_corr = gpu.to_gpu(h_corr)
        cu.memcpy_htod(d_cgf, h_cgf)

        # ------------------------
        # grid it up
        # ------------------------
        d_umax = gpu.max(cumath.fabs(d_u))
        d_vmax = gpu.max(cumath.fabs(d_v))
        umax = np.int32(np.ceil(d_umax.get() / self.du))
        vmax = np.int32(np.ceil(d_vmax.get() / self.du))

        ## grid ($$)
        #  This should be improvable via:
        #    - shared memory solution? I tried...
        #    - better coalesced memory access? I tried...
        #    - reorganzing and indexing UV data beforehand?
        #       (i.e. http://www.nvidia.com/docs/IO/47905/ECE757_Project_Report_Gregerson.pdf)
        #    - storing V(u,v) in texture memory?
        self.gridVis_wBM_kernel(d_grd, d_bm, d_cbm, d_cnt, d_u, d_v, d_re, d_im, np.int32(nx), np.float32(self.du),
                                np.int32(gcount), np.int32(umax), np.int32(vmax),
                                np.int32(1 if self.correct_p_angle else 0),
                                block=self.blocksize_2D, grid=self.gridsize_2D)

        ## apply weights
        self.wgtGrid_kernel(d_bm, d_cnt, self.briggs, np.int32(nx), np.int32(fullsize), block=self.blocksize_2D, grid=self.gridsize_2D)
        hfac = np.int32(1)
        self.dblGrid_kernel(d_bm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
        self.dblGrid_kernel(d_cbm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)

        self.shiftGrid_kernel(d_bm, d_nbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        self.trimIm_kernel_2(d_cbm, d_sample_tmp, np.int32(nx), np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)
        # h_dim = d_dim.get()
        # self.write_fits(h_dim, '/Users/wangfeng/liuhuisample.fit', 'DIRTY_IMAGE')

        self.shiftGrid_kernel(d_sample_tmp, d_sample_half, np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)

        self.shiftGrid_kernel(d_cbm, d_sample, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        #


        ## normalize
        self.wgtGrid_kernel(d_grd, d_cnt, self.briggs, nx, np.int32(fullsize), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Reflect grid about v axis
        hfac = np.int32(-1)
        self.dblGrid_kernel(d_grd, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Shift both
        self.shiftGrid_kernel(d_grd, d_ngrd, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        # Sun Model
        # Sun disk radius = 16.1164 arcmin
        radius = 16.1164 * 60 / self.cell
        self.diskGrid_kernel(d_sun_disk, np.int32(self.imsize * 2), np.int32(radius), np.int32(100),
                             block=self.blocksize_2D,
                             grid=self.gridsize_2D)

        fft.fft(d_sun_disk, d_grd, plan)

        # ------------------------
        # Make the beam
        # ------------------------
        ## Transform to image plane



        # Sampling function and multiply disk
        self.sub_dot_mul_kernel(d_grd, d_sample, d_cbm, nx, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        fft.fft(d_cbm, d_sun_disk, plan)

        self.trimIm_kernel(d_sun_disk, d_fdisk, np.int32(nx), np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)
        d_bmax = gpu.max(d_fdisk)
        bmax = d_bmax.get()
        bmax1 = np.float32(1. / bmax)
        self.nrmBeam_kernel(d_fdisk, bmax1, np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)

        #
        fft.fft(d_nbm, d_bm, plan)
        ## Shift
        self.shiftGrid_kernel(d_bm, d_nbm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Correct for C
        self.corrGrid_kernel(d_nbm, d_corr, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        # Trim
        self.trimIm_kernel(d_nbm, d_fim, np.int32(nx), np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.copyIm_kernel(d_nbm, d_fbm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Normalize
        d_bmax = gpu.max(d_fim)
        bmax = d_bmax.get()
        bmax1 = np.float32(1. / bmax)
        self.nrmBeam_kernel(d_fim, bmax1, np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)
        d_bmax = gpu.max(d_fbm)
        bmax = d_bmax.get()
        bmax2 = np.float32(1. / bmax)
        self.nrmBeam_kernel(d_fbm, bmax2, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Pull onto CPU
        dpsf = d_fim.get()
        dpsf2 = d_fbm.get()

        # ------------------------
        # Make the map
        # ------------------------
        ## Transform to image plane
        if (x_offset <> 0 or y_offset <> 0):
            self.sub_cuda_cyclic_shift_kernel(d_ngrd, d_cbm, np.int32(nx), np.int32(y_offset), np.int32(x_offset),
                                              block=self.blocksize_2D, grid=self.gridsize_2D)
            # self.sub_cuda_cyclic_shift_kernel(d_ngrd, d_cbm, np.int32(nx), np.int32(200), np.int32(200), block=self.blocksize_2D, grid=self.gridsize_2D)
            fft.fft(d_cbm, d_grd, plan)
        else:
            fft.fft(d_ngrd, d_grd, plan)
        ## Shift
        self.shiftGrid_kernel(d_grd, d_ngrd, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Correct for C
        self.corrGrid_kernel(d_ngrd, d_corr, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Trim
        self.trimIm_kernel(d_ngrd, d_dim, np.int32(nx), np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.copyIm_kernel(d_ngrd, d_fbm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Normalize (Jy/beam)i
        self.nrmGrid_kernel(d_dim, bmax1, np.int32(self.imsize), block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.nrmGrid_kernel(d_fbm, bmax2, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        ## Finish timers
        t_end = time.time()
        t_full = t_end - t_start
        logger.debug("Gridding execution time %0.5f" % t_full + ' s')
        logger.debug("\t%0.5f" % (t_full / gcount) + ' s per visibility')

        # ----------------------

        # h_dim = d_dim.get()
        # self.write_fits(h_dim, '/Users/wangfeng/liuhuidirty.fit', 'DIRTY_IMAGE')

        # self.trim_float2_kernel(d_sample,d_sam, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        ## Return dirty psf (CPU), dirty image (GPU) and sun disk
        if self.clean_mode=='hogbom'  or preclean==1:
            return dpsf2, d_dim, d_fdisk, d_sample
        else:
            return dpsf, d_dim, d_fdisk, d_sample_half


    ######################
    # CLEAN functions
    ######################

    def get_clean_beam(self, dpsf, window=20):
        """
        Clean a dirty beam on the CPU
        A very simple approach - just extract the central beam #improvable#
        Another solution would be fitting a 2D Gaussian,
        e.g. http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
        """
        # print "Cleaning the dirty beam"

        # [ -7.43396394e-03   2.87406555e-01   5.12483288e+02   5.16996963e+02
        # 1.04011661e+02  -2.77956159e+01   5.52422629e+00]
        h, w = np.shape(dpsf)

        cpsf = np.zeros([h, w])
        # window = 50
        g_dpsf = np.zeros([window, window])
        g_dpsf = dpsf[w / 2 - window / 2:w / 2 + window / 2 - 1, h / 2 - window / 2:h / 2 + window / 2 - 1]
        # cpsf = gf.gaussian()
        fit = gf.fitgaussian(g_dpsf)
        fit[2] = w / 2  # fit[2] - window / 2 + w / 2
        fit[3] = h / 2  # fit[3] - window / 2 + h / 2
        cpsf = gf.twodgaussian(fit, shape=(h, w))
        cpsf = cpsf / np.max(cpsf)
        return np.float32(cpsf)

        h, w = np.shape(dpsf)

        cpsf = np.zeros([h, w])
        window = 100
        g_dpsf = np.zeros([window, window])
        g_dpsf = dpsf[w / 2 - window / 2:w / 2 + window / 2 - 1, h / 2 - window / 2:h / 2 + window / 2 - 1]
        # cpsf = gf.gaussian()
        fit = gf.fitgaussian(g_dpsf)
        d_cpsf = gf.twodgaussian(fit, shape=(window, window))

        # cpsf[w / 2 - window/2:w / 2 + window/2 -1, h / 2 - window/2:h / 2 + window/2 -1] = g_cpsf[0:window -1, 0:window -1]
        # cpsf=np.zeros([h,w])
        cpsf[w / 2 - window / 2:w / 2 + window / 2, h / 2 - window / 2:h / 2 + window / 2] = d_cpsf[:, :]  ##Normalize
        cpsf = cpsf / np.max(cpsf)
        return np.float32(cpsf)

    def gpu_getmax(self, map):
        """
        Use pycuda to get the maximum absolute deviation of the residual map,
        with the correct sign
        """
        #imax = gpu.max(cumath.fabs(map)).get()
        imax = gpu.max(map).get()
        #if gpu.max(map).get() != imax: imax *= -1
        return np.float32(imax)

    def cuda_steer(self, gpu_dirty, gpu_sample, gpu_dpsf, gpu_cpsf, thresh=0.2, gain=0.1, add_back=1):

        height, width = np.shape(gpu_dirty)
        print "HEIGHT, WIDTH:", width, height
        print "GPU_sample:", np.shape(gpu_sample)
        print "gpu_dpsf:", np.shape(gpu_dpsf)
        print "gpu_cpsf:", np.shape(gpu_cpsf)
        nx = self.imsize
        ## Grid parameters - #improvable#
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (int(width / tsize), int(height / tsize))  # The number of thread blocks     (x,y)
        blocksizehalf = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsizehalf = (int(self.imsize / tsize), int(self.imsize / tsize))  # The number of thread blocks     (x,y)

        ## Setup GPU constants
        # gpu_max_id = gpu.to_gpu(np.zeros(int32(0))
        gpu_max_id = gpu.to_gpu(np.zeros(1, dtype='int32'))
        imax = self.gpu_getmax(gpu_dirty)
        dmax = imax
        gainstep = imax*gain
        if (thresh<1 and thresh>0):
            thresh_val = np.float32(thresh * imax)
        else:
            thresh_val = thresh
        print "DMAX:", dmax, thresh_val


        ## Steps 1-3 - Iterate until threshold has been reached
        t_start = time.time()
        i = 0
        # Clean image till Disk
        logger.debug("Subtracting dirty beam...")
        clean_error = False


        d_grd = gpu.zeros((np.int(nx), np.int(nx)), np.complex64)
        d_contour = gpu.zeros_like(d_grd)
        d_contour_tmp = gpu.zeros_like(d_grd)

        d_bim = gpu.zeros([self.imsize, self.imsize], dtype=np.float32)
        d_fim = gpu.zeros([height, width], dtype=np.float32)
        d_clean = gpu.zeros([self.imsize, self.imsize], dtype=np.float32)
        gpu_clean = gpu.zeros([self.imsize, self.imsize], dtype=np.float32)
        d_dirty = gpu.zeros([self.imsize, self.imsize], dtype=np.float32)
        plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)

        self.copyRIm_kernel(gpu_dirty, d_grd, np.int32(nx), block=blocksize, grid=gridsize)

        #self.shiftGrid_kernel(gpu_sample, d_sample, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)


        while (abs(imax) > (thresh_val)) and (i < 100):
            '''if (np.mod(i, 100) == 0):
                print "Steer Clean iteration", i'''
            ## Step 1 - Check max value
            lastmax = imax

            ## Step 2 - Move Contour to An Empty Plan,
            ## 2.1
            print "MAX",imax
            self.copyContour_kernel(d_grd, d_contour, d_clean,  np.int32(nx), np.float32(dmax - (i+1)*np.float32(gainstep)),
                                 block=blocksize,
                                 grid=gridsize)

            fft.fft(d_contour, d_contour_tmp, plan)
            ## 2.2 Sampling function and multiply disk
            ## FFT, Degrid and IFFT
            self.sub_dot_mul_kernel(d_contour_tmp, gpu_sample, d_contour, np.int32(nx), np.int32(nx), block=blocksize, grid=gridsize)

            fft.fft(d_contour, d_contour_tmp, plan)


            ## 2.4 Norm
            self.copyIm_kernel(d_contour_tmp, d_fim, np.int32(nx), block=blocksize, grid=gridsize)

            d_bmax = gpu.max(d_fim)
            bmax = d_bmax.get()
            bmax1 = np.float32(1. / bmax)
            self.nrmBeam_kernel(d_fim, bmax1, np.int32(nx), block=blocksize, grid=gridsize)

            ## Step 3 - Subtract the beam (assume that it is normalized to have max 1)
            ##          This kernel simultaneously reconstructs the CLEANed image.
            #if self.Debug:
            logger.debug("Subtracting dirty beam " + str(i) + ", maxval=%0.8f" % imax + ' at x=' + str(
                        gpu_max_id.get() % width) + ', y=' + str(gpu_max_id.get() / width))
            gainstep = imax*gain
            self.sub_steer_kernel(d_grd, d_fim, np.float32(gainstep) ,
                                 np.int32(nx), block=blocksize, grid=gridsize)

            ## Step 4 - Find maximum value using gpuarray
            #self.copyIm_kernel(d_grd, d_fim, np.int32(nx), block=blocksize, grid=gridsize)
            self.trimDisk_kernel_2(d_grd, d_bim, np.int32(self.imsize), np.int32((16.1125 *1.02* 60*2 / self.cell)), block=blocksizehalf, grid=gridsizehalf)

            imax = self.gpu_getmax(d_bim)

            i += 1

            # if imax > lastmax:
            #     clean_error = True
            #     break;


        t_end = time.time()
        t_full = t_end - t_start
        print "Steer execution time %0.5f" % t_full + ' s', "Number of Iterations %d" % i
        logger.debug("Steer execution time %0.5f" % t_full + ' s')

        # CLEAN image convolution
        from scipy import signal
        from scipy import misc

        # fft.fft(d_contour, d_contour_tmp, plan)
        # fft.fft(gpu_cpsf, gpu_cpsf_tmp, plan)

        ## Step 5 - Add the residuals back in
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (int(self.imsize / tsize), int(self.imsize / tsize))  # The number of thread blocks     (x,y)

        #self.trimIm_kernel(d_grd, d_dirty, np.int32(nx), np.int32(self.imsize), block=blocksize, grid=gridsize)
        self.copyIm_kernel(d_grd, d_dirty, np.int32(nx), block=blocksize, grid=gridsize)


        #self.copyIm_kernel(d_clean, gpu_clean, nx, block=self.blocksize_F2D, grid=self.gridsize_F2D)

        if add_back == 1:
            mc.add_noise_kernel(gpu_dirty, d_clean, np.float32(width + height))

        # Trim
        d_clean3 = self.convolution(d_clean, gpu_cpsf, plan)

        # self.copyIm_kernel(d_clean,gpu_clean,np.int32(nx), block=blocksize, grid=gridsize)
        #self.trim_float_image_kernel(d_clean, gpu_clean, np.int32(nx), np.int32(self.imsize), block=blocksize, grid=gridsize)
        #self.nrmBeam_kernel(d_fim, np.float32(gainstep), np.int32(nx), block=blocksize, grid=gridsize)

        # h_disk = gpu_clean.get()
        # prefix = self.infile
        # prefix, ext = os.path.splitext(os.path.basename(prefix))
        #
        # pathPrefix = self.outdir
        # if pathPrefix == None:
        #     filename = prefix + '_disk_%dt.png' % self.chan
        # else:
        #     if pathPrefix[-1:] == '/':
        #         pathPrefix = pathPrefix[:-1]
        #     filename = pathPrefix + '/' + prefix + '_disk_%dt.png' % self.chan
        # # TODO , FITS
        # self.muser_draw.draw_one(filename, "DISK MAP", self.fov, h_disk, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,
        #                          self.dec + 0.5, 16.1, axistype=0)

        return d_dirty, d_clean3, clean_error

    def cuda_hogbom(self, gpu_dirty, gpu_pmodel, gpu_clean, gpu_dpsf, gpu_cpsf, thresh=0.2, damp=1, gain=0.05,
                    prefix='test', add_flag=1, add_back=1):
        """
        Use CUDA to implement the Hogbom CLEAN algorithm

        A nice description of the algorithm is given by the NRAO, here:
        http://www.cv.nrao.edu/~abridle/deconvol/node8.html

        Parameters:
        * dirty: The dirty image (2D numpy array)
        * dpsf: The dirty beam psf  (2D numpy array)
        * thresh: User-defined threshold to stop iteration, as a fraction of the max pixel intensity (float)
        * damp: The damping factor to scale the dirty beam by
        * prefix: prefix for output image file names
        """

        height, width = np.shape(gpu_dirty)

        ## Grid parameters - #improvable#
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (int(width / tsize), int(height / tsize))  # The number of thread blocks     (x,y)

        ## Setup cleam image and point source model
        ## Setup GPU constants
        # gpu_max_id = gpu.to_gpu(np.zeros(int32(0))
        gpu_max_id = gpu.to_gpu(np.zeros(1, dtype='int32'))
        imax = self.gpu_getmax(gpu_dirty)
        thresh_val = thresh  # np.float32(thresh * imax)

        ## Steps 1-3 - Iterate until threshold has been reached
        t_start = time.time()
        i = 0
        # Clean image till Disk
        logger.debug("Subtracting dirty beam...")
        clean_error = False

        while (abs(imax) > (thresh_val)) and (i < 200):
            '''if (np.mod(i, 100) == 0):
                print "Hogbom iteration", i'''
            ## Step 0 - Check unreasonable max value
            lastmax = imax

            ## Step 1 - Find max
            self.find_max_kernel(gpu_dirty, gpu_max_id, imax, np.int32(width), np.int32(height), gpu_pmodel,
                                 block=blocksize, grid=gridsize)
            ## Step 2 - Subtract the beam (assume that it is normalized to have max 1)
            ##          This kernel simultaneously reconstructs the CLEANed image.
            #if self.Debug:
            logger.debug("Subtracting dirty beam " + str(i) + ", maxval=%0.8f" % imax + ' at x=' + str(
                        gpu_max_id.get() % width) + ', y=' + str(gpu_max_id.get() / width))
            self.sub_beam_kernel(gpu_dirty, gpu_dpsf, gpu_max_id, gpu_clean, gpu_cpsf, np.float32(gain * imax),
                                 np.int32(width), \
                                 np.int32(height), np.int32(add_flag), block=blocksize, grid=gridsize)
            i += 1
            ## Step 3 - Find maximum value using gpuarray
            imax = self.gpu_getmax(gpu_dirty)
            # if imax > lastmax:
            #     clean_error = True
            #     break;

        t_end = time.time()
        t_full = t_end - t_start
        print "Hogbom execution time %0.5f" % t_full + ' s', "Number of Iterations %d" % i
        logger.debug("Hogbom execution time %0.5f" % t_full + ' s')

        ## Step 4 - Add the residuals back in
        if add_back == 1:
            mc.add_noise_kernel(gpu_dirty, gpu_clean, np.float32(width + height))
        return gpu_dirty, gpu_pmodel, gpu_clean, clean_error

    def cuda_histogram(self, image, binsize, no=1):

        ## Calculate histogram


        dirty_map_max = gpu.max(image).get()
        dirty_map_min = gpu.min(image).get()

        if dirty_map_min < 0:
            dirty_map_min = -int(round(abs(dirty_map_min) + 0.5))
        else:
            dirty_map_min = round(dirty_map_min)

        if dirty_map_max < 0:
            dirty_map_max = -int(round(abs(dirty_map_max) + 0.5))
        else:
            dirty_map_max = round(round(dirty_map_max + 0.5))

        gpu_histogram = gpu.zeros([binsize], np.int32)

        height, width = np.shape(image)
        ## Grid parameters - #improvable#
        tsize = 8
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (self.iDivUp(height, tsize), self.iDivUp(width, tsize))  # The number of thread blocks     (x,y)
        # gridsize = (int(height/tsize), int(width/ tsize))   # The number of thread blocks     (x,y)
        self.sub_histogram_kernel(image, np.int32(self.imsize), np.int32(self.imsize), gpu_histogram,
                                  np.int32(dirty_map_max), np.int32(dirty_map_min), np.int32(binsize), block=blocksize,
                                  grid=gridsize)

        gpu_smooth_histogram = gpu.zeros([binsize], np.int32)
        gpu_smooth_histogram2 = gpu.zeros([binsize], np.int32)
        # Temporary Testing
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)             # The number of threads per block (x,y,z)
        gridsize = (self.iDivUp(1, tsize), self.iDivUp(binsize, tsize))   # The number of thread blocks     (x,y)
        width = binsize
        radius = 32
        height = 1
        self.sub_mean_average_kernel(gpu_histogram, gpu_smooth_histogram, np.int32(height), np.int32(width), np.int32(radius), block=blocksize, grid=gridsize)
        width = binsize
        radius = 32
        height = 1
        blocksize = (int(tsize), int(tsize), 1)             # The number of threads per block (x,y,z)
        gridsize = (self.iDivUp(1, tsize), self.iDivUp(width, tsize))   # The number of thread blocks     (x,y)
        self.sub_mean_average_kernel(gpu_smooth_histogram, gpu_smooth_histogram2, np.int32(height), np.int32(width), np.int32(radius), block=blocksize, grid=gridsize)

        h_histogram = gpu_smooth_histogram2.get()
        # Histogram with NumPy
        # if (int(image.max()) - int(image.min()))>2000:
        #     bins = np.arange(int(image.min()), int(image.max()),(int(image.max())- int(image.min()))//2000 )
        # else:
        #     bins = np.arange(int(image.min()), int(image.max()))
        # item = image[:,:]
        # h_histogram,bins = np.histogram(item,bins)


        #
        # hist_smooth = gaussian_filter(h_histogram, 30)
        #
        # hist_smooth = hist_smooth[:-1]

        # Test the convolution kernel.
        # Generate or load a test image
        # You probably want to display the image using the tool of your choice here.
        # filterx = mfilter.gaussian_kernel()
        # destImage = h_histogram.copy()
        # destImage[:] = np.nan
        # destImage = mfilter.convolution_cuda(h_histogram,  filterx,  filterx)
        # destImage = destImage.reshape(2000)
        # destImage = destImage[:-1]

        bins = np.arange(0, binsize)
        data = np.vstack([bins * 1.0, h_histogram]).transpose()
        # plt.scatter(data[:,0],data[:,1])
        if no == 1:
            gmm = GMM(dim=2, ncomps=3, data=data, method="kmeans")
        else:
            gmm = GMM(dim=2, ncomps=2, data=data, method="kmeans")

        gmm_data = []
        for comp in gmm.comps:
            gmm_data.append([comp.mu[0], comp.mu[1]])
            # draw1dnormal(comp)

        gmm_data = sorted(gmm_data, key=lambda gmm_data: gmm_data[0])
        sky_peak = gmm_data[0][0]
        disk_peak = gmm_data[1][0]

        h_histogram = h_histogram[:-1]
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        if self.plot_me:
            # axs[1].imshow(cpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 99), cmap=cm.gray)
            prefix = self.infile
            prefix, ext = os.path.splitext(os.path.basename(prefix))

            pathPrefix = self.outdir
            if pathPrefix == None:
                plt.savefig(pathPrefix + '/' + prefix + '_his_%d_%d.png' % self.chan)
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                if not os.path.exists(pathPrefix):
                    os.makedirs(pathPrefix)
                plt.bar(center, h_histogram, align='center', width=width)
                if no == 1:
                    plt.savefig(pathPrefix + '/' + prefix + '_his_%d.png' % self.chan)
                else:
                    plt.savefig(pathPrefix + '/' + prefix + '_his2_%d.png' % self.chan)
            plt.close()

        #
        sky_peak = (dirty_map_max - dirty_map_min) * sky_peak / 2000.
        disk_peak = (dirty_map_max - dirty_map_min) * disk_peak / 2000.
        logger.debug("Peak Value: sky_peak:%f  disk_peak:%f" % (sky_peak, disk_peak))

        return sky_peak, disk_peak

    def write_fits(self, data, fitsfile, type):
        # create_fits(self, object, obs_date, obs_time,data, imagetype):
        self.muser_fits.create_fits(data, self.object, self.muser_date.strftime("%Y-%m-%d"),
                                    self.muser_date.strftime("%H:%M:%S.%f"), type)
        self.muser_fits.append_common_header(self.current_freq, self.polarization, self.ra, self.dec, self.p_angle)
        self.muser_fits.write_fits(self.outdir, fitsfile)

    def preclean(self):

        ## Create the PSF & dirty image
        #   dpsf - PSF, gpu_im ( dirty image)
        #   dpsf is computed by CPU, gpu_im is in the GPU
        # nx - 2 imsize, it means 2048 when imsize=1024

        dpsf, gpu_im, gpu_disk, gpu_sample = self.cuda_gridvis(0, 0, 0, 1)

        gpu_dpsf = gpu.to_gpu(dpsf)
        h_disk = gpu_disk.get()
        # gpu_dpsf2 = gpu.to_gpu(dpsf2)

        ## Clean the PSF
        if self.imsize >= 1024:
            cpsf = self.get_clean_beam(dpsf, 50)  # self.imsize / 32.)
        elif self.imsize >= 512:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 24.)
        elif self.imsize >= 256:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 16.)

        gpu_cpsf = gpu.to_gpu(cpsf)

        if self.plot_me: #and self.Debug:

            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True);
            plt.subplots_adjust(wspace=0)
            axs[0].imshow(cpsf, vmin=np.percentile(cpsf, 1), vmax=np.percentile(cpsf, 100), cmap=cm.hot)
            # fig.colorbar(cpsf)
            im = axs[1].imshow(dpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 100), cmap=cm.hot)
            # fig.colorbar(im)
            # axs[1].imshow(cpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 99), cmap=cm.gray)
            pathPrefix = self.outdir
            if pathPrefix == None:
                plt.savefig('cleanbeam_%dp.png' % self.chan)
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                if not os.path.exists(pathPrefix):
                    os.makedirs(pathPrefix)
                plt.savefig(pathPrefix + '/' + 'cleanbeam_%dp.png' % self.chan)
            plt.close()

            prefix = self.infile
            prefix, ext = os.path.splitext(os.path.basename(prefix))

            pathPrefix = self.outdir
            if pathPrefix == None:
                filename = prefix + '_disk_%dp.png' % self.chan
                fitsfile = prefix + '_disk_%dp.fits' % self.chan
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                filename = pathPrefix + '/' + prefix + '_disk_%dp.png' % self.chan
                fitsfile = pathPrefix + '/' + prefix + '_disk_%dp.fits' % self.chan
            # TODO , FITS
            self.muser_draw.draw_one(filename, "DISK MAP", self.fov, h_disk, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,
                                     self.dec + 0.5, 16.1, False, axistype=0)


        # Histogram
        sky_peak, disk_peak = self.cuda_histogram(gpu_im, 2000)

        if self.clean_mode=='steer':
            d_dirty = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)
            tsize = 16
            blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
            gridsize = (int(self.imsize / tsize), int(self.imsize / tsize))  # The number of thread blocks     (x,y)
            self.trim_float_image_kernel(gpu_im, d_dirty, np.int32(self.imsize*2), np.int32(self.imsize),
                                         block=blocksize, grid=gridsize)
            dirty = d_dirty.get()
        else:
            dirty = gpu_im.get()

        if self.Debug:
            logger.debug("Plotting dirty image")
            # TODO , FITS
        pathPrefix = self.outdir
        prefix = self.infile
        prefix, ext = os.path.splitext(os.path.basename(prefix))
        if pathPrefix == None:
            filename = prefix + '_dirty_%dp.pdf' % self.chan
            fitsfile = prefix + '_dirty_%dp.fits' % self.chan
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + prefix + '_dirty_%dp.pdf' % self.chan
            fitsfile = pathPrefix + '/' + prefix + '_dirty_%dp.fits' % self.chan

        if self.plot_me:
            title = ('DIRTY IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' if self.polarization == -2 else 'R', self.current_freq / 1e9)
            # self.muser_draw.draw_one(filename, title, self.fov, dirty, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5, self.dec + 0.5, 16.1, axistype=0)
            self.muser_draw.draw_one(filename, title, self.fov, dirty,
                                     self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell) * 2 / self.imsize, axis=False, axistype=0)

        if self.writefits:
            self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        ## Run CLEAN
        # Clean till >=Disk
        height, width = np.shape(gpu_im)
        imax= gpu.max(gpu_im)

        gpu_pmodel = gpu.zeros([height, width], dtype=np.float32)
        gpu_clean = gpu.zeros([height, width], dtype=np.float32)
        #gpu_dirty_shift = gpu.zeros([height, width], dtype=np.float32)

        gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean, gpu_dpsf,
                                                                          gpu_cpsf,
                                                                          thresh=disk_peak, gain=0.1, add_flag=1,
                                                                          add_back=0)

        dirty = gpu_dirty.get()
        if self.Debug:
            prefix = self.infile
            prefix, ext = os.path.splitext(os.path.basename(prefix))

            pathPrefix = self.outdir
            if pathPrefix == None:
                filename = prefix + '_dirty2_%dp.pdf' % self.chan
                fitsfile = prefix + '_dirty2_%dp.fits' % self.chan
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                filename = pathPrefix + '/' + prefix + '_dirty2_%dp.pdf' % self.chan
                fitsfile = pathPrefix + '/' + prefix + '_dirty2_%dp.fits' % self.chan

            if self.plot_me:
                logger.debug("Plotting dirty image")
                title = ('DIRTY IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' if self.polarization == -2 else 'R', self.current_freq / 1e9)
                # self.muser_draw.draw_one(filename, title, self.fov, dirty, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5, self.dec + 0.5, 16.1, axistype=0)
                self.muser_draw.draw_one(filename, title, self.fov, dirty,
                                         self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                         self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                         self.dec - (self.cell * self.imsize / 2) / 3600,
                                         self.dec + (self.cell * self.imsize / 2) / 3600,
                                         (16.1125 * 60 / self.cell) * 2 / self.imsize, axis=False, axistype=0)

                # TODO , FITS

            if self.writefits:
                self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        # h_disk_im  = self.sub_sun_disk_offset(h_disk, dirty)
        gpu_disk_im = self.sub_sun_disk_offset(h_disk, dirty)
        h_disk_im = gpu_disk_im.get()
        logger.debug("X,Y: %d %d " % (np.argmax(np.max(h_disk_im, axis=0)), np.argmax(np.max(h_disk_im, axis=1))))

        x_offset = -np.argmax(np.max(h_disk_im, axis=0)) + self.imsize / 2
        y_offset =  self.imsize / 2 - np.argmax( np.max(h_disk_im, axis=1))

        #self.du = np.float32(1. / (arc2rad * self.cell)) / (self.imsize * 2.
        x_offset = x_offset/self.du/3600.*2
        y_offset = -y_offset/self.du/3600.*2
        return x_offset, y_offset, sky_peak/imax, disk_peak/imax



    def clean(self, x_offset, y_offset, sky_peak_ratio, disk_peak_ratio, hybrid_mode=False):

        ## Create the PSF & dirty image
        #   dpsf - PSF, gpu_im ( dirty image)
        #   dpsf is computed by CPU, gpu_im is in the GPU
        # nx - 2 imsize, it means 2048 when imsize=1024

        nx = np.int32(2 * self.imsize)
        d_dirty = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)
        gpu_sun_disk = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)

        d_final = gpu.zeros((np.int(self.imsize / 2), np.int(self.imsize / 2)), np.float32)
        logger.debug("OFFSET: %f %f " % (x_offset, y_offset))
        print("OFFSET: %f %f %f %f" % (x_offset, y_offset, sky_peak_ratio, disk_peak_ratio))
        if self.weight_mode == 'natural':
            dpsf, gpu_im, gpu_disk, gpu_sample = self.cuda_gridvis(x_offset, y_offset, 0)
        else:
            dpsf, gpu_im, gpu_disk, gpu_sample = self.cuda_gridvis(x_offset, y_offset, 1)

        #dpsf, gpu_im, gpu_disk, gpu_sample = self.cuda_gridvis(plan, 0,0, 0)

        # , misc.minabs(gpu_im)

        d_imax = gpu.max(gpu_im)
        imax=d_imax.get()
        sky_peak = round(imax * sky_peak_ratio)
        disk_peak = round(imax * disk_peak_ratio)
        gpu_dpsf = gpu.to_gpu(dpsf)




        # gpu_dpsf2 = gpu.to_gpu(dpsf2)

        ## Clean the PSF
        if self.imsize >= 1024:
            cpsf = self.get_clean_beam(dpsf, 50) #self.imsize / 32.)
        elif self.imsize >= 512:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 24.)
        elif self.imsize >= 256:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 16.)

        gpu_cpsf = gpu.to_gpu(cpsf)

        if self.plot_me:

            print("Plotting dirty and cleaned beam")
            logger.debug("Plotting dirty and cleaned beam")

            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True);
            plt.subplots_adjust(wspace=0)
            axs[0].imshow(cpsf, vmin=np.percentile(cpsf, 1), vmax=np.percentile(cpsf, 100), cmap=cm.hot)
            # fig.colorbar(cpsf)
            im = axs[1].imshow(dpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 100), cmap=cm.hot)
            # fig.colorbar(im)
            # axs[1].imshow(cpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 99), cmap=cm.gray)
            pathPrefix = self.outdir
            if pathPrefix == None:
                plt.savefig('cleanbeam_%d.png' % self.chan)
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                if not os.path.exists(pathPrefix):
                    os.makedirs(pathPrefix)
                plt.savefig(pathPrefix + '/' + 'cleanbeam_%d.png' % self.chan)
            plt.close()

        if self.correct_p_angle:
            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[0])),
                                 np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[1])))

            self.sub_rotate_image_kernel(gpu_im, d_dirty, np.int32(self.imsize), np.int32(self.imsize),
                                         np.float32(self.p_angle), np.float32(1.), block=self.blocksize_F2D,
                                         grid=self.gridsize_F2D)

            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[0])),
                                 np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[1])))
            self.trim_float_image_kernel(d_dirty, d_final, np.int32(self.imsize), np.int32(self.imsize / 2),
                                         block=self.blocksize_F2D, grid=self.gridsize_F2D)
            dirty = d_final.get()
        else:
            dirty = gpu_im.get()


        prefix = self.infile
        prefix, ext = os.path.splitext(os.path.basename(prefix))
        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = prefix + '_dirty_%d.png' % self.chan
            fitsfile = prefix + '_dirty_%d.fits' % self.chan
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + prefix + '_dirty_%d.png' % self.chan
            fitsfile = pathPrefix + '/' + prefix + '_dirty_%d.fits' % self.chan

        if self.plot_me:
            logger.debug("Plotting final dirty image")
            title = ('DIRTY IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' if self.polarization == -2 else 'R', self.current_freq / 1e9)
            if self.correct_p_angle:
                self.muser_draw.draw_one(filename, title, self.fov, dirty, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell)*2 / self.imsize, axis=False, axistype=1)
            else:
                self.muser_draw.draw_one(filename, title, self.fov, dirty, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell)/ self.imsize, axis=False, axistype=1)

        if self.writefits:
            self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        ## Run CLEAN
        # Clsean till >=Disk
        height, width = np.shape(gpu_im)
        gpu_pmodel = gpu.zeros([height, width], dtype=np.float32)
        gpu_clean = gpu.zeros([height, width], dtype=np.float32)
        gpu_clean1 = gpu.zeros([height, width], dtype=np.float32)

        if hybrid_mode==True:
            if self.clean_mode=='hogbom':

                # tsize = 16
                # blocksize1 = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
                # gridsize1 = (int(self.imsize / tsize), int(self.imsize / tsize))  # The number of thread blocks     (x,y)
                #
                # self.sub_image_kernel(gpu_im, np.float32(disk_peak), np.int32((16.1125 * 60 / self.cell)), np.int32(self.imsize), block=blocksize1, grid=gridsize1)

                gpu_dirty, gpu_pmodel, gpu_clean1, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean1, gpu_dpsf,
                                                                              gpu_cpsf,
                                                                              thresh=disk_peak, gain=0.1, add_flag=1,
                                                                              add_back=0)
                gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_clean1, gpu_pmodel, gpu_clean, gpu_dpsf,
                                                                              gpu_cpsf,
                                                                              thresh=sky_peak, gain=0.1, add_flag=1,
                                                                              add_back=1)
                #self.add_image_kernel(gpu_clean, gpu_disk, np.int32(self.imsize), block=blocksize1, grid=gridsize1)

            else:

                # Solar Disk Processing
                #disk_max = disk_peak*np.float32(1. / disk_max)
                # tsize = 16
                # blocksize1 = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
                # gridsize1 = (int(height / tsize), int(width / tsize))  # The number of thread blocks     (x,y)
                #
                # self.sub_image_kernel(gpu_im, np.float32(disk_peak), np.int32((16.1125 * 60 / self.cell)*2), np.int32(nx), block=blocksize1, grid=gridsize1)

                gpu_dirty, gpu_clean, clean_result = self.cuda_steer(gpu_im, gpu_sample, gpu_dpsf, gpu_cpsf,
                                                                              thresh=0.2, gain=0.1,
                                                                              add_back=0)

        else:
            if self.clean_mode=='hogbom':
                gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean, gpu_dpsf,
                                                                              gpu_cpsf,
                                                                              thresh=sky_peak, gain=0.1, add_flag=1,
                                                                              add_back=1)
            else:
                gpu_dirty, gpu_clean, clean_result = self.cuda_steer(gpu_im, gpu_sample, gpu_dpsf, gpu_cpsf,
                                                                              thresh=0.2, gain=0.1,
                                                                              add_back=0)

        d_light = gpu.to_gpu(self.sun_disk_light)

        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (int(self.imsize / tsize), int(self.imsize / tsize))  # The number of thread blocks     (x,y)

        self.sun_disk_kernel(gpu_sun_disk, d_light, np.int32(self.imsize),np.int32((16.1125 * 1. * 60 / self.cell)), block=blocksize, grid=gridsize)

        # Trim
        plan = fft.Plan((np.int(height), np.int(width)), np.complex64, np.complex64)
        d_sun = self.convolution(gpu_sun_disk, gpu_cpsf, plan)

        h_disk = d_sun.get()
        prefix = self.infile
        prefix, ext = os.path.splitext(os.path.basename(prefix))

        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = prefix + '_disk_%dt2.png' % self.chan
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + prefix + '_disk_%dt2.png' % self.chan
        # TODO , FITS
        self.muser_draw.draw_one(filename, "DISK MAP", self.fov, h_disk, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,
                                 self.dec + 0.5, 16.1125, axis=False, axistype=1)



        # dirty = gpu_dirty.get()

        if self.correct_p_angle:
            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[0])),
                                 np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[1])))

            self.sub_rotate_image_kernel(gpu_clean, d_dirty, np.int32(self.imsize), np.int32(self.imsize),
                                         np.float32(self.p_angle), np.float32(1.), block=self.blocksize_F2D,
                                         grid=self.gridsize_F2D)

            noff = np.int32((nx - self.imsize) / 2)

            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[0])),
                                 np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[1])))
            self.trim_float_image_kernel(d_dirty, d_final, np.int32(self.imsize), np.int32(self.imsize / 2),
                                         block=self.blocksize_F2D, grid=self.gridsize_F2D)
            #clean = d_final.get()
            return d_final, d_sun
        else:
            return gpu_clean, d_sun


    def gpu_info(self):
        (free, total) = cu.mem_get_info()
        print "Global memory occupancy:%f free at %f " % (free, total)
        logger.debug("Global memory occupancy:%f free at %f " % (free, total))

    def gpu_sun_disk(self, sun_data, sun_light, sun_radius, w_offset, h_offset):
        """
        Use pycuda to get a 2D array with Sun disk
        """

        # draw a standard SUN disk
        h, w = np.shape(sun_data)
        for i in range(h):
            for j in range(w):
                if abs(i - h // 2 - h_offset) <= sun_radius and abs(j - w // 2 - w_offset) <= sun_radius:
                    sun_data[i, j] = 0
                else:
                    sun_data[i, j] = sun_light
        # fft

        return sun_data

    def set_size(self):

        LOW_FRE = {0: (400000000, 256), 1: (800000000, 512), 2: (1200000000, 1024), 3: (1600000000, 1024)}
        HIGH_FRE = {0: (2000000000, 1280), 1: (2400000000, 1280), 2: (2800000000, 1280), 3: (3200000000, 1280), 4: (3600000000, 1280),
                    5: (4000000000, 2560), 6: (4400000000, 2560), 7: (4800000000, 2560), 8: (5200000000, 2560), 9: (5600000000, 2560),
                    10: (6000000000, 2560), 11: (6400000000, 2560), 12: (6800000000, 2560), 13: (7200000000, 2560), 14: (7600000000, 2560),
                    15: (8000000000, 5120), 16: (8400000000, 5120), 17: (8800000000, 5120), 18: (9200000000, 5120), 19: (9600000000, 5120),
                    20: (10000000000, 5120), 21: (10400000000, 5120), 22: (10800000000, 5120), 23: (11200000000, 5120), 24: (11600000000, 5120),
                    25: (12000000000, 5120), 26: (12400000000, 5120), 27: (12800000000, 5120), 28: (13200000000, 5120), 29: (13600000000, 5120),
                    30: (14000000000, 5120), 31: (14400000000, 5120), 32: (14600000000, 5120)}

        # Retrieve Antennas information
        if self.object.upper().strip()=='MUSER-1':
            self.antennas = 40
            if self.band == None:
                if self.freq == 400E6:
                    self.imsize = 256
                elif self.freq == 800E6:
                    self.imsize = 512
                elif self.freq in [1200E6, 1600E6]:
                    self.imsize = 1024
            else:
                self.imsize = np.int32(LOW_FRE[self.band][1])

            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, self.muser_date.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            self.antennas = 60
            if self.band == None:
                if self.freq in [2000E6, 2400E6, 2800E6, 3200E6, 3600E6]:
                    self.imsize = 1280
                elif self.freq in [4000E6, 4400E6, 4800E6, 5200E6, 5600E6, 6000E6, 6400E6, 6800E6, 7200E6, 7600E6]:
                    self.imsize = 2560
                elif self.freq in [8000E6, 8400E6, 8800E6, 9200E6, 9600E6, 10000E6, 10400E6, 10800E6, 11200E6, 11600E6, 12000E6, 12400E6, 12800E6, 13200E6, 13600E6, 14000E6, 14400E6, 14800E6]:
                    self.imsize = 5120
            else:
                self.imsize = np.int32(HIGH_FRE[self.band][1])
            self.Flag_Ant =  self.muser_ant.get_flag_antenna(2, self.muser_date.strftime("%Y-%m-%d %H:%M:%S"))


    def clean_with_file(self, infile, outdir,  INP_CHANNEL, WEIGHT_MODE,CLEAN_MODE, AUTO_MOVE, MoveRA, MoveDEC, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):
        # clean.clean_with_fits(inputfile, outdir, channel, mode, automove, movera, movedec, plot, fits, correct, debug)
        # Load settings for each example
        self.infile = infile
        self.outdir = outdir

        if not os.path.exists(self.infile):
            logger.error("No file exist: %s." % self.infile)
            return

        self.briggs = np.float32(1e7)  # weight parameter
        self.light_speed = 299792458.  # Speed of light
        self.weight_mode = WEIGHT_MODE
        self.clean_mode=CLEAN_MODE
        self.auto_move = AUTO_MOVE
        self.inp_channel = INP_CHANNEL
        self.move_ra = MoveRA
        self.move_dec = MoveDEC
        self.plot_me = PLOT_ME
        self.writefits = WRITE_FITS
        self.Debug = DEBUG
        self.correct_p_angle = P_ANGLE
        if self.Debug:
            print "Start CLEAN..."

        self.fitsfile = pyfits.open(self.infile,ignore_missing_end=True)

        self.telescope = self.fitsfile[0].header['INSTRUME'].strip()
        if self.telescope != 'MUSER':
            logger.error("Current program can only support MUSER.")
            return

        self.channel = self.fitsfile[0].data.data.shape[3]
        self.baseline_number = self.fitsfile[0].header['GCOUNT']
        self.obs_date = self.fitsfile[0].header['DATE-OBS']
        self.muser_date = datetime.datetime.strptime(self.obs_date[:-3], "%Y-%m-%dT%H:%M:%S.%f")

        self.object = self.fitsfile[0].header['OBJECT']
        self.polarization = np.int32(self.fitsfile[0].header['CRVAL3'])
        self.basefreq = np.float32(self.fitsfile[0].header['CRVAL4'])
        self.bandwidth = np.float32(self.fitsfile[0].header['CDELT4'])
        self.ra = np.float32(self.fitsfile[0].header['OBSRA'])
        self.dec = np.float32(self.fitsfile[0].header['OBSDEC'])
        self.freq = self.basefreq + np.float32(self.fitsfile[1].data["IF FREQ"][0])
        logger.debug("File:       %s" % self.infile)
        logger.debug("Instrument: %s" % self.telescope)
        logger.debug("Obs date:   %s" % self.obs_date)
        logger.debug("Base Frequency:  %d" % self.basefreq)
        logger.debug("Bandwidth:  %d" % self.bandwidth)
        logger.debug("Channels:   %d" % self.channel)
        logger.debug("Polarization: %d" % self.polarization)
        logger.debug("Target RA:  %f" % self.ra)
        logger.debug("Target DEC: %f" % self.dec)
        self.set_size()

        logger.debug("FLag Antennas: %s " % self.Flag_Ant)
        print "FLag Antennas: %s " % self.Flag_Ant

        if self.correct_p_angle:
            self.imsize *= 1.5

        if self.inp_channel=='':
            self.channel_start = 0
            self.channel_end = 16
        else:
            self.channel_start=int(self.inp_channel)
            self.channel_end = self.channel_start +1

    def clean_read_data(self):
        if self.object.upper().strip()=='MUSER-1':
            self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
        else:
            self.current_freq = self.freq + (15-self.chan) * self.bandwidth + self.bandwidth // 2

        sun_disk_radius = MuserSunDisk()
        self.sun_disk_light= sun_disk_radius.getdata ((self.current_freq -  self.bandwidth // 2) / 1E6)

        self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi
        self.set_size()
        self.h_uu = np.ndarray(shape=(self.baseline_number), dtype='float64')
        self.h_vv = np.ndarray(shape=(self.baseline_number), dtype='float64')
        self.h_rere = np.ndarray(shape=(self.baseline_number), dtype='float32')
        self.h_imim = np.ndarray(shape=(self.baseline_number), dtype='float32')

        if self.infile.find('.fitsidi') != -1:
            self.h_uu = np.float64((self.freq + self.chan * 25000000) * self.fitsfile[0].data[:].UU)
            self.h_vv = np.float64((self.freq + self.chan * 25000000) * self.fitsfile[0].data[:].VV)

            for bl in range(0, self.baseline_number):
                self.h_rere[bl] = np.float32(self.fitsfile[0].data[:].data[bl][0][0][self.chan][0][0])
                self.h_imim[bl] = np.float32(self.fitsfile[0].data[:].data[bl][0][0][self.chan][0][1])
                ## make GPU arrays
                self.h_uu = np.float32(self.h_uu.ravel())
                self.h_vv = np.float32(self.h_vv.ravel())
                self.h_rere = np.float32(self.h_rere.ravel())
                self.h_imim = np.float32(self.h_imim.ravel())

        elif self.infile.find('.uvfits') != -1:
            # good = np.where(self.fitsfile[0].data.data[:, 0, 0, self.chan, 0, 0] != 0)
            # the unit of uu and  vv is seconds
            self.h_uu = np.float64(self.fitsfile[0].data.par('uu')) # [good])  # * self.current_freq)
            self.h_vv = np.float64(self.fitsfile[0].data.par('vv')) #[good])  # * self.current_freq)

            self.h_uu *= self.current_freq
            self.h_vv *= self.current_freq

            self.h_rere = np.float32(self.fitsfile[0].data.data[:, 0, 0, self.chan, 0, 0])
            self.h_imim = np.float32(self.fitsfile[0].data.data[:, 0, 0, self.chan, 0, 1])

    def write_result(self, clean):
        prefix = self.infile
        prefix, ext = os.path.splitext(os.path.basename(prefix))
        pathPrefix = self.outdir

        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = prefix + '_clean_%d.png' % self.chan
            fitsfile = prefix + '_clean_%d.fits' % self.chan
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + prefix + '_clean_%d.png' % self.chan
            fitsfile = pathPrefix + '/' + prefix + '_clean_%d.fits' % self.chan

        if self.plot_me:
            logger.debug("Plotting final clean image")

            title = ('CLEAN IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' if self.polarization == -2 else 'R', self.current_freq / 1e9)
            if self.correct_p_angle:
                self.muser_draw.draw_one(filename, title, self.fov, clean, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell) *2 / self.imsize, axis=False, axistype=1)
            else:
                self.muser_draw.draw_one(filename, title, self.fov, clean, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell)  / self.imsize, axis=False, axistype=1)


        if self.writefits:
            self.write_fits(clean, fitsfile, 'CLEANED_IMAGE')
        print "CLEAN finished and images saved."

    def clean_with_fits(self, infile, outdir, INP_CHANNEL, WEIGHT_MODE,  CLEAN_MODE, AUTO_MOVE, MoveRA, MoveDEC, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):
        # clean.clean_with_fits(inputfile, outdir, channel, mode, automove, movera, movedec, plot, fits, correct, debug)
        # Load settings for each example
        self.clean_with_file(infile, outdir, INP_CHANNEL,WEIGHT_MODE,  CLEAN_MODE, AUTO_MOVE, MoveRA, MoveDEC, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG)
        self.gpu_info()
        self.p_angle, b, sd = pb0r(self.obs_date[:-9])
        self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.

        for self.chan in range(self.channel_start, self.channel_end):
            self.clean_read_data()
            self.cell = self.angular_resolution / 3.
            self.fov = self.cell * self.imsize
            self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)
            nx = np.int32(2 * self.imsize)
            if self.auto_move == True:
                x_offset, y_offset, sky, disk = self.preclean()
                sky_ratio = sky.get()
                disk_ratio = disk.get()
                print "x_offset", x_offset, "y_offset", y_offset
            else:
                x_offset = self.move_ra/180.*np.pi
                y_offset = self.move_dec/180.*np.pi
                print "x_offset", x_offset, "y_offset", y_offset
                sky_ratio = 0.1
                disk_ratio = 0.2


            gpu_clean, gpu_sun = self.clean(x_offset, y_offset, sky_ratio, disk_ratio)
            clean = gpu_clean.get()

            self.write_result(clean)

    def hybrid_clean_with_fits(self, infile, outdir, INP_CHANNEL, WEIGHT_MODE, CLEAN_MODE, AUTO_MOVE, MoveRA, MoveDEC, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):

        self.clean_with_file(infile, outdir, INP_CHANNEL,WEIGHT_MODE,  CLEAN_MODE, AUTO_MOVE, MoveRA, MoveDEC, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG)
        self.gpu_info()
        self.p_angle, b, sd = pb0r(self.obs_date[:-9])
        self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.

        for self.chan in range(self.channel_start, self.channel_end):
            self.clean_read_data()

            self.cell = self.angular_resolution / 3.
            self.fov = self.cell * self.imsize
            self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)
            nx = np.int32(2 * self.imsize)
            logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq,self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

            self.weight_mode='natural'
            self.clean_mode='hogbom'
            if self.auto_move == True:
                x_offset, y_offset, sky, disk = self.preclean()
                print "x_offset", x_offset, "y_offset", y_offset
                sky_ratio = sky.get()
                disk_ratio = disk.get()
            else:
                x_offset = self.move_ra
                y_offset = self.move_dec
                print "x_offset", x_offset, "y_offset", y_offset
                sky_ratio = 0.1
                disk_ratio = 0.15
            self.weight_mode='uniform'
            self.clean_mode='hogbom'


            gpu_clean, gpu_sun = self.clean(x_offset, y_offset, sky_ratio, disk_ratio, hybrid_mode=True)

            self.weight_mode='natural'
            self.clean_mode='steer'
            gpu_clean1, gpu_sun = self.clean(0., 0., sky_ratio, disk_ratio, hybrid_mode=True)

            dmin_clean = gpu.min(gpu_clean)
            min_clean = dmin_clean.get()

            tsize = 16
            blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
            gridsize = (int(self.imsize / tsize), int(self.imsize / tsize))  # The number of thread blocks     (x,y)
            if (min_clean<0):
                print "Add flat ClEAN:", -min_clean
                self.add_flat_kernel(gpu_clean, np.int32(-min_clean), block=blocksize, grid=gridsize)

            dmin_clean1 = gpu.min(gpu_clean1)
            min_clean1 = dmin_clean1.get()
            if (min_clean1<0):
                print "Add flat CLEAN1:", -min_clean1
                self.add_flat_kernel(gpu_clean1, np.int32(-min_clean1), block=blocksize, grid=gridsize)

            dmax_clean = gpu.max(gpu_clean)
            max_clean = dmax_clean.get()

            dmax_clean1 = gpu.max(gpu_clean1)
            max_clean1 = dmax_clean1.get()

            #print np.int32(max_clean1),np.int32(max_clean1*disk_ratio)

            self.add_clean_kernel(gpu_clean1, gpu_clean, gpu_sun, np.int32(self.imsize),np.float32(1./max_clean1), np.float32(1./max_clean),np.int32(max_clean1*disk_ratio),  np.int32(max_clean1), np.int32((16.1125 *1.* 60 / self.cell)), block=blocksize,grid=gridsize)

            #__global__ void add_clean_kernel(float *im, float *nim, int nx, float s1, float s2, int back, int max, int r){
            clean = gpu_clean.get()
            clean1 = gpu_clean1.get()
            self.write_result(clean1)

