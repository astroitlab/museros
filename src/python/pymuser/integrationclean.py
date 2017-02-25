import numpy as np
import time, pdb, sys, pyfits
import matplotlib
import math
import os
import sys

#matplotlib.use('Agg')
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
from muserbase import *
from musersun import *
from muserdraw import *
from muserfits import *
import musercuda as mc
from muserfilter import *
from muserant import *
from muserfits import *
from muserfile import *

# from sklearn.mixture import GMM

from argparse import *

logger = logging.getLogger('muser')

GRID = lambda x, y, W: ((x) + ((y) * W))

IGRIDX = lambda tid, W: tid % W
IGRIDY = lambda tid, W: int(tid) / int(W)

MATRIX_SIZE = 4
TILE_SIZE = 2
BLOCK_SIZE = TILE_SIZE


class MuserClean_integration():
    ######################
    # CUDA kernels


    def __init__(self,):

        self.muser_draw = MuserDraw()
        self.muser_fits = MuserFits()
        self.muser_ant = MuserAntenna()
        self.muser_fits = MuserFits()

        # -------------------
        # Gridding kernels
        # -------------------
        self.sub_beam_kernel = self.cuda_compile(mc.sub_beam_kernel_source, "sub_beam_kernel")
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
        self.copyIm_kernel = self.module.get_function("copyIm_kernel")
        self.copyRIm_kernel = self.module.get_function("copyRIm_kernel")
        self.diskGrid_kernel = self.module.get_function("diskGrid_kernel")
        self.trim_float_image_kernel = self.module.get_function("trim_float_image_kernel")

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
        if (np.abs(eta) > etalim[i]):
            ip = 1
            x = eta * eta - 1
        else:
            ip = 0
            x = eta * eta - etalim[i] * etalim[i]
            # numerator via Horner's rule
        mnp = nnum[i] - 1
        num = p[ip, twoalp, mnp]
        for j in np.arange(mnp):
            num = num * x + p[ip, twoalp, mnp - 1 - j]
            # denominator via Horner's rule
        nq = ndenom[i] - 1
        denom = q[ip, twoalp, nq]
        for j in np.arange(nq):
            denom = denom * x + q[ip, twoalp, nq - 1 - j]

        return np.float32(num / denom)

    def gcf(self, n, width):
        """
        Create table with spheroidal gridding function, C
        This implementation follows MIRIAD's grid.for subroutine.
        """
        alpha = 1.
        j = 2 * alpha
        p = 0.5 * j
        phi = np.zeros(n, dtype=np.float32)
        for i in np.arange(n):
            x = np.float32(2 * i - (n - 1)) / (n - 1)
            phi[i] = (np.sqrt(1 - x * x) ** j) * self.spheroid(x, width, p)
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

    def cuda_gridvis(self, plan, x_offset, y_offset):
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

        nx = np.int32(2 * self.imsize)
        noff = np.int32((nx - self.imsize) / 2)
        arc2rad = np.float32(np.pi / 180. / 3600.)
        du = np.float32(1. / (arc2rad * self.cell)) / (self.imsize * 2.)
        logger.debug("1 Pixel DU  = %f" % du)
        ## grab data

        # h_uu = np.float32(self.h_uu.ravel())
        # h_vv = np.float32(self.h_vv.ravel())
        # h_uu = np.float32(self.h_uu[self.band, self.polar].ravel())
        # h_vv = np.float32(self.h_vv[self.band, self.polar].ravel())
        #
        # h_rere = np.float32(self.h_rere[self.band, self.polar, self.chan].ravel())
        # h_imim = np.float32(self.h_imim[self.band, self.polar, self.chan].ravel())

        # h_rere = np.float32(self.h_rere.ravel())
        # h_imim = np.float32(self.h_imim.ravel())
        gcount = np.int32(np.size(self.h_uu))
        print "GOUNT 1:", gcount, np.size(self.h_rere)

        # import matplotlib.pyplot as plt
        # plt.scatter(self.h_uu, self.h_vv)

        blen = 0
        bl_order = np.ndarray(shape=(self.baseline_base*self.integrate_frame_num, 2), dtype=int)
        good = []

        if self.subarray == 1:  # MUSER-I
            antennas = 40
        else:
            antennas = 60

        inter_num = 1
        while inter_num <= self.integrate_frame_num:
            for border1 in range(0, antennas - 1):
                for border2 in range(border1 + 1, antennas):
                    bl_order[blen][0] = border1
                    bl_order[blen][1] = border2
                    blen = blen + 1
            inter_num += 1

        h_u = []
        h_v = []
        h_re = []
        h_im = []
        for blen in range(0, self.baseline_base*self.integrate_frame_num):
            if (bl_order[blen][0] not in self.Flag_Ant) and (bl_order[blen][1] not in self.Flag_Ant):
                good.append(blen)
                h_u.append(self.h_uu[blen])
                h_v.append(self.h_vv[blen])
                h_re.append(self.h_rere[blen])
                h_im.append(self.h_imim[blen])

        gcount = np.int32(np.size(h_u))
        print "After Flagged:", gcount, np.size(h_re)

        # plt.scatter(h_u, h_v)

        d_u = gpu.to_gpu(np.array(h_u, dtype='float32'))
        d_v = gpu.to_gpu(np.array(h_v, dtype='float32'))
        d_re = gpu.to_gpu(np.array(h_re, dtype='float32'))
        d_im = gpu.to_gpu(np.array(h_im, dtype='float32'))
        d_cnt = gpu.zeros((np.int(nx), np.int(nx)), np.int32)
        d_grd = gpu.zeros((np.int(nx), np.int(nx)), np.complex64)
        d_ngrd = gpu.zeros_like(d_grd)
        d_bm = gpu.zeros_like(d_grd)
        d_nbm = gpu.zeros_like(d_grd)
        d_cbm = gpu.zeros_like(d_grd)

        d_fbm = gpu.zeros((np.int(nx), np.int(nx)), np.float32)
        d_fim = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)
        d_dim = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)

        d_sun_disk = gpu.zeros_like(d_grd)
        d_fdisk = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)

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
        umax = np.int32(np.ceil(d_umax.get() / du))
        vmax = np.int32(np.ceil(d_vmax.get() / du))

        self.gridVis_wBM_kernel(d_grd, d_bm, d_cbm, d_cnt, d_u, d_v, d_re, d_im, np.int32(nx), np.float32(du),
                                np.int32(gcount), np.int32(umax), np.int32(vmax),
                                np.int32(1 if self.correct_p_angle else 0),
                                block=self.blocksize_2D, grid=self.gridsize_2D)

        ## apply weights
        self.wgtGrid_kernel(d_bm, d_cnt, self.briggs, nx, np.int32(1),  block=self.blocksize_2D, grid=self.gridsize_2D)
        hfac = np.int32(1)
        self.dblGrid_kernel(d_bm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
        self.dblGrid_kernel(d_cbm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)

        self.shiftGrid_kernel(d_bm, d_nbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        self.shiftGrid_kernel(d_cbm, d_bm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        ## normalize
        self.wgtGrid_kernel(d_grd, d_cnt, self.briggs, nx, np.int32(1), block=self.blocksize_2D, grid=self.gridsize_2D)
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
        self.sub_dot_mul_kernel(d_grd, d_bm, d_cbm, nx, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        fft.fft(d_cbm, d_sun_disk, plan)

        self.trimIm_kernel(d_sun_disk, d_fdisk, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        d_bmax = gpu.max(d_fdisk)
        bmax = d_bmax.get()
        bmax1 = np.float32(1. / bmax)
        self.nrmBeam_kernel(d_fdisk, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)

        #
        fft.fft(d_nbm, d_bm, plan)
        ## Shift
        self.shiftGrid_kernel(d_bm, d_nbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Correct for C
        self.corrGrid_kernel(d_nbm, d_corr, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        # Trim
        self.trimIm_kernel(d_nbm, d_fim, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.copyIm_kernel(d_nbm, d_fbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Normalize
        d_bmax = gpu.max(d_fim)
        bmax = d_bmax.get()
        bmax1 = np.float32(1. / bmax)
        self.nrmBeam_kernel(d_fim, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        d_bmax = gpu.max(d_fbm)
        bmax = d_bmax.get()
        bmax2 = np.float32(1. / bmax)
        self.nrmBeam_kernel(d_fbm, bmax2, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Pull onto CPU
        # dpsf = d_fim.get()
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
        self.shiftGrid_kernel(d_grd, d_ngrd, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Correct for C
        self.corrGrid_kernel(d_ngrd, d_corr, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Trim
        self.trimIm_kernel(d_ngrd, d_dim, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.copyIm_kernel(d_ngrd, d_fbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Normalize (Jy/beam)i
        self.nrmGrid_kernel(d_dim, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.nrmGrid_kernel(d_fbm, bmax2, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        ## Finish timers
        t_end = time.time()
        t_full = t_end - t_start
        logger.debug("Gridding execution time %0.5f" % t_full + ' s')
        logger.debug("\t%0.5f" % (t_full / gcount) + ' s per visibility')

        # ----------------------

        ## Return dirty psf (CPU), dirty image (GPU) and sun disk

        return dpsf2, d_dim, d_fdisk

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
        g_dpsf = np.zeros([np.int32(window), np.int32(window)])
        g_dpsf = dpsf[np.int32(w / 2) - np.int32(window / 2): np.int32(w / 2) + np.int32(window / 2) - 1, np.int32(h / 2) - np.int32(window / 2): np.int32(h / 2) + np.int32(window / 2) - 1]
        # cpsf = gf.gaussian()
        fit = gf.fitgaussian(g_dpsf)
        fit[2] = w / 2  # fit[2] - window / 2 + w / 2
        fit[3] = h / 2  # fit[3] - window / 2 + h / 2
        cpsf = gf.twodgaussian(fit, shape=(h, w))
        cpsf = cpsf / np.max(cpsf)
        return np.float32(cpsf)

        # fit = gf.gaussfit(dpsf)
        # print fit


        h, w = np.shape(dpsf)

        cpsf = np.zeros([h, w])
        window = 100
        g_dpsf = np.zeros([window, window])
        g_dpsf = dpsf[w / 2 - window / 2:w / 2 + window / 2 - 1, h / 2 - window / 2:h / 2 + window / 2 - 1]
        # cpsf = gf.gaussian()
        fit = gf.fitgaussian(g_dpsf)
        print fit
        d_cpsf = gf.twodgaussian(fit, shape=(window, window))

        cpsf[w / 2 - window / 2:w / 2 + window / 2, h / 2 - window / 2:h / 2 + window / 2] = d_cpsf[:, :]  ##Normalize
        cpsf = cpsf / np.max(cpsf)
        return np.float32(cpsf)

    def gpu_getmax(self, map):
        """
        Use pycuda to get the maximum absolute deviation of the residual map,
        with the correct sign
        """
        imax = gpu.max(cumath.fabs(map)).get()
        if gpu.max(map).get() != imax: imax *= -1
        return np.float32(imax)

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
            dirty_map_min = int(dirty_map_min)

        if dirty_map_max < 0:
            dirty_map_max = -int(round(abs(dirty_map_max) + 0.5))
        else:
            dirty_map_max = int(round(dirty_map_max + 0.5))

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

        if self.Debug:
            print "Peak Value: sky_peak:", sky_peak,  "disk_peak:", disk_peak
        # print gmm

        h_histogram = h_histogram[:-1]
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        if self.plot_me:
            pathPrefix = self.outdir
            if pathPrefix == None:
                plt.savefig(pathPrefix + '/' + self.prefix + '_his_%d_%d.png' % (self.chan, self.integrate_frame_num))
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                if not os.path.exists(pathPrefix):
                    os.makedirs(pathPrefix)
                plt.bar(center, h_histogram, align='center', width=width)
                if no == 1:
                    plt.savefig(pathPrefix + '/' + self.prefix + '_his_%d_%d.png'% (self.chan, self.integrate_frame_num))
                else:
                    plt.savefig(pathPrefix + '/' + self.prefix + '_his2_%d_%d.png'% (self.chan, self.integrate_frame_num))
            plt.close()

        #
        sky_peak = (dirty_map_max - dirty_map_min) * sky_peak / 2000.
        disk_peak = (dirty_map_max - dirty_map_min) * disk_peak / 2000.
        logger.debug("Peak Value: sky_peak:%f  disk_peak:%f" % (sky_peak, disk_peak))
        # print sky_peak,disk_peak

        return sky_peak, disk_peak

    def write_fits(self, data, fitsfile, type):
        # create_fits(self, object, obs_date, obs_time,data, imagetype):
        self.muser_fits.create_fits(data, self.object, self.muser_date.strftime("%Y-%m-%d"),
                                    self.muser_date.strftime("%H:%M:%S.%f"), type)
        self.muser_fits.append_common_header(self.current_freq, self.polar, self.ra, self.dec, self.p_angle)
        self.muser_fits.write_fits(self.outdir, fitsfile)

    def preclean(self):

        ## Create the PSF & dirty image
        #   dpsf - PSF, gpu_im ( dirty image)
        #   dpsf is computed by CPU, gpu_im is in the GPU
        # nx - 2 imsize, it means 2048 when imsize=1024
        nx = np.int32(2 * self.imsize)

        # create fft plan nx*nx
        self.plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)

        d_dirty = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)

        dpsf, gpu_im, gpu_disk = self.cuda_gridvis(self.plan, 0, 0)
        gpu_dpsf = gpu.to_gpu(dpsf)
        h_disk = gpu_disk.get()

        ## Clean the PSF
        if self.imsize >= 1024:
            cpsf = self.get_clean_beam(dpsf, 50)  # self.imsize / 32.)
        elif self.imsize >= 512:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 24.)
        elif self.imsize >= 256:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 16.)

        gpu_cpsf = gpu.to_gpu(cpsf)
        fitsfile = ''

        # Histogram
        sky_peak, disk_peak = self.cuda_histogram(gpu_im, 2000)

        dirty = gpu_im.get()

        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = self.prefix + '_dirty_%dp_%d.png' % (self.chan, self.integrate_frame_num)
            fitsfile = self.prefix + '_dirty_%dp_%d.fits' % (self.chan, self.integrate_frame_num)
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + self.prefix + '_dirty_%dp_%d.png' % (self.chan, self.integrate_frame_num)
            fitsfile = pathPrefix + '/' + self.prefix + '_dirty_%dp_%d.fits' % (self.chan, self.integrate_frame_num)
            print fitsfile

            logger.debug("Plotting dirty image")
            title = ('DIRTY IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (
            self.obs_date, 'L' if self.polarization == -2 else 'R', self.current_freq / 1e9)
            self.muser_draw.draw_one(filename, title, self.fov, dirty,
                                     self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell) * 2 / self.imsize, axis=True, axistype=0)

        if self.writefits and self.Debug:
            self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        ## Run CLEAN
        # Clean till >=Disk
        height, width = np.shape(gpu_im)
        gpu_pmodel = gpu.zeros([height, width], dtype=np.float32)
        gpu_clean = gpu.zeros([height, width], dtype=np.float32)
        gpu_dirty_shift = gpu.zeros([height, width], dtype=np.float32)

        gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean, gpu_dpsf,
                                                                          gpu_cpsf,
                                                                          thresh=disk_peak, gain=0.1, add_flag=1,
                                                                          add_back=0)

        dirty = gpu_dirty.get()

        # h_disk_im  = self.sub_sun_disk_offset(h_disk, dirty)
        gpu_disk_im = self.sub_sun_disk_offset(h_disk, dirty)
        h_disk_im = gpu_disk_im.get()
        logger.debug("X,Y: %d %d " % (np.argmax(np.max(h_disk_im, axis=0)), np.argmax(np.max(h_disk_im, axis=1))))

        return -np.argmax(np.max(h_disk_im, axis=0)) + self.imsize / 2, -self.imsize / 2 + np.argmax(
            np.max(h_disk_im, axis=1)), sky_peak, disk_peak

    def clean(self, x_offset, y_offset, sky_peak, disk_peak):

        ## Create the PSF & dirty image
        #   dpsf - PSF, gpu_im ( dirty image)
        #   dpsf is computed by CPU, gpu_im is in the GPU
        # nx - 2 imsize, it means 2048 when imsize=1024
        nx = np.int32(2 * self.imsize)

        # create fft plan nx*nx
        self.plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)
        d_dirty = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)
        d_final = gpu.zeros((np.int(self.imsize / 2), np.int(self.imsize / 2)), np.float32)
        logger.debug("OFFSET: %f %f " % (x_offset, y_offset))
        dpsf, gpu_im, gpu_disk = self.cuda_gridvis(self.plan, x_offset, y_offset)

        # , misc.minabs(gpu_im)

        gpu_dpsf = gpu.to_gpu(dpsf)
        h_disk = gpu_disk.get()
        # gpu_dpsf2 = gpu.to_gpu(dpsf2)


        ## Clean the PSF
        if self.imsize >= 1024:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 32.)
        elif self.imsize >= 512:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 24.)
        elif self.imsize >= 256:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 16.)

        gpu_cpsf = gpu.to_gpu(cpsf)

        if self.plot_me:

            logger.debug("Plotting dirty and cleaned beam")

            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True);
            plt.subplots_adjust(wspace=0)
            axs[0].imshow(cpsf, vmin=np.percentile(cpsf, 1), vmax=np.percentile(cpsf, 100), cmap=cm.jet)
            # fig.colorbar(cpsf)
            im = axs[1].imshow(dpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 100), cmap=cm.jet)
            # fig.colorbar(im)
            # axs[1].imshow(cpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 99), cmap=cm.gray)
            pathPrefix = self.outdir
            if pathPrefix == None:
                plt.savefig('cleanbeam_%d_%d.png' % (self.chan, self.integration_number))
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                if not os.path.exists(pathPrefix):
                    os.makedirs(pathPrefix)
                plt.savefig(pathPrefix + '/' + 'cleanbeam_%d_%d.png' % (self.chan, self.integrate_frame_num))
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

        # prefix = self.start_frame_time.get_short_string()+' TO '+self.end_frame_time.get_short_string()
        # # prefix, ext = os.path.splitext(os.path.basename(prefix))
        # pathPrefix = self.outdir
        # if pathPrefix == None:
        #     filename = prefix + '_dirty_%d_%d.png' % (self.chan, self.integrate_frame_num)
        #     fitsfile = prefix + '_dirty_%d_%d.fits' % (self.chan, self.integrate_frame_num)
        # else:
        #     if pathPrefix[-1:] == '/':
        #         pathPrefix = pathPrefix[:-1]
        #     filename = pathPrefix + '/' + prefix + '_dirty_%d_%d.png' % (self.chan, self.integrate_frame_num)
        #     fitsfile = pathPrefix + '/' + prefix + '_dirty_%d_%d.fits' % (self.chan, self.integrate_frame_num)
        #
        # if self.plot_me:
        #     logger.debug("Plotting final dirty image")
        #     title = ('DIRTY IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' if self.polarization == 0 else 'R', self.current_freq / 1e9)
        #     if self.correct_p_angle:
        #         self.muser_draw.draw_one(filename, title, self.fov, dirty, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
        #                              self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
        #                              self.dec - (self.cell * self.imsize / 2) / 3600,
        #                              self.dec + (self.cell * self.imsize / 2) / 3600,
        #                              (16.1125 * 60 / self.cell)*2 / self.imsize, axis=True, axistype=1)
        #     else:
        #         self.muser_draw.draw_one(filename, title, self.fov, dirty, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
        #                              self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
        #                              self.dec - (self.cell * self.imsize / 2) / 3600,
        #                              self.dec + (self.cell * self.imsize / 2) / 3600,
        #                              (16.1125 * 60 / self.cell)/ self.imsize, axis=True, axistype=1)
        #
        # if self.writefits:
        #     self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        ## Run CLEAN
        # Clean till >=Disk
        height, width = np.shape(gpu_im)
        gpu_pmodel = gpu.zeros([height, width], dtype=np.float32)
        gpu_clean = gpu.zeros([height, width], dtype=np.float32)

        gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean, gpu_dpsf,
                                                                          gpu_cpsf,
                                                                          thresh=sky_peak, gain=0.1, add_flag=1,
                                                                          add_back=1)

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
            clean = d_final.get()
        else:
            clean = gpu_clean.get()

        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = pathPrefix + '_clean_%d_%d.png' % (self.chan, self.integrate_frame_num)
            fitsfile = pathPrefix + '_clean_%d_%d.fits' % (self.chan, self.integrate_frame_num)
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/'  + '_clean_%d_%d.png' % (self.chan, self.integrate_frame_num)
            fitsfile = pathPrefix + '/'  + '_clean_%d_%d.fits' % (self.chan, self.integrate_frame_num)

        if self.plot_me:
            logger.debug("Plotting final clean image")

            title = ('CLEAN IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' if self.polarization == 0 else 'R', self.current_freq / 1e9)
            if self.correct_p_angle:
                self.muser_draw.draw_one(filename, title, self.fov, clean, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell) *2 / self.imsize, axis=True, axistype=1)
            else:
                self.muser_draw.draw_one(filename, title, self.fov, clean, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,
                                     self.dec - (self.cell * self.imsize / 2) / 3600,
                                     self.dec + (self.cell * self.imsize / 2) / 3600,
                                     (16.1125 * 60 / self.cell)  / self.imsize, axis=True, axistype=1)


        if self.writefits:
            self.write_fits(clean, fitsfile, 'CLEANED_IMAGE')

    def gpu_info(self):
        (free, total) = cu.mem_get_info()
        logger.debug("Global memory occupancy:%f free at %f " % (free, total))
        print ("Global memory occupancy:%f free at %f " % (free, total))

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
        return sun_data


    def clean_integration_FITS(self, mode, infile, outdir, niter, BAND, channel, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):
        # Load settings for each example
        self.infile = infile
        self.outdir = outdir

        if not os.path.exists(self.infile):
            logger.error("No file exist: %s." % self.infile)
            return

        self.briggs = np.float32(1e7)  # weight parameter
        self.light_speed = 299792458.  # Speed of light
        self.plot_me = PLOT_ME
        self.writefits = WRITE_FITS
        self.Debug = DEBUG
        self.correct_p_angle = P_ANGLE
        self.integration_number = niter
        self.baseline_number = 0
        self.band = BAND
        # Retrieve Antennas information

        # if self.infile.find('.uvfits') != -1:

        self.fitsfile = pyfits.open(self.infile, ignore_missing_end=True)

        self.telescope = self.fitsfile[0].header['INSTRUME'].strip()
        if self.telescope != 'MUSER':
            logger.error("Current program can only support MUSER.")
            return

        self.channel = self.fitsfile[0].data.data.shape[3]
        self.baseline = self.fitsfile[0].header['GCOUNT']
        self.object = self.fitsfile[0].header['OBJECT']
        self.obs_date = self.fitsfile[0].header['DATE-OBS']
        self.muser_date = datetime.datetime.strptime(self.obs_date[:-3], "%Y-%m-%dT%H:%M:%S.%f")
        self.polarization = np.int32(self.fitsfile[0].header['CRVAL3'])
        self.basefreq = np.float32(self.fitsfile[0].header['CRVAL4'])
        self.bandwidth = np.float32(self.fitsfile[0].header['CDELT4'])
        self.ra = np.float32(self.fitsfile[0].header['OBSRA'])
        self.dec = np.float32(self.fitsfile[0].header['OBSDEC'])

        if self.object.upper().strip()=='MUSER-1':
            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, self.muser_date.strftime("%Y-%m-%d %H:%M:%S"))
            self.baseline_base =  780
            self.band_number = 4
        else:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(2, self.muser_date.strftime("%Y-%m-%d %H:%M:%S"))
            self.baseline_base = 1770
            self.band_number = 33

        logger.debug("File:       %s" % self.infile)
        logger.debug("Instrument: %s" % self.telescope)
        logger.debug("Obs date:   %s" % self.obs_date)
        logger.debug("Base Frequency:  %d" % self.basefreq)
        # logger.debug("Frequency:  %d" % self.freq)
        logger.debug("Bandwidth:  %d" % self.bandwidth)
        logger.debug("Channels:   %d" % self.channel)
        logger.debug("Polarization: %d" % self.polarization)
        logger.debug("Target RA:  %f" % self.ra)
        logger.debug("Target DEC: %f" % self.dec)

        if self.band != None:
            self.band_start = self.band
            self.band_end = self.band + 1
        else:
            self.band_start = 0
            self.band_end = self.band_number

        if channel != None:
            self.chan_start = channel
            self.chan_end = channel + 1
        else:
            self.chan_start = 0
            self.chan_end = 16

        self.baseline_number = self.integration_number*self.baseline_base
        self.record_num = self.integration_number*self.band_number
        logger.debug("FLag Antennas: %s" % self.Flag_Ant)
        print "Baseline Number, Record Number:", self.baseline_number, self.record_num
        #self.Flag_Ant = []

        self.h_uu = np.ndarray(shape=(self.band_number, self.baseline_number), dtype='float64')
        self.h_vv = np.ndarray(shape=(self.band_number, self.baseline_number), dtype='float64')
        self.h_rere = np.ndarray(shape=(self.band_number, 2, 16, self.baseline_number), dtype='float32')
        self.h_imim = np.ndarray(shape=(self.band_number, 2, 16, self.baseline_number), dtype='float32')


        if mode == "Loop":
            for integrate_Num in range(0, self.integration_number):
                for band in range(0, self.band_number):
                    start = (integrate_Num*self.band_number+band)*self.baseline_base
                    # UU, VV
                    self.h_uu[band, integrate_Num*self.baseline_base: (integrate_Num+1)*self.baseline_base] = np.float64(self.fitsfile[0].data.par('uu')[start: start+self.baseline_base])  # * self.current_freq)
                    self.h_vv[band, integrate_Num*self.baseline_base: (integrate_Num+1)*self.baseline_base] = np.float64(self.fitsfile[0].data.par('vv')[start: start+self.baseline_base])  # * self.current_freq)

                    for polar in range(0, 2):
                        for self.chan in range(0, 16):
                            self.h_rere[band, polar, self.chan, integrate_Num*self.baseline_base: (integrate_Num+1)*self.baseline_base] = np.float32(self.fitsfile[0].data.data[start: start+self.baseline_base, 0, 0, self.chan, polar, 0])
                            self.h_imim[band, polar, self.chan, integrate_Num*self.baseline_base: (integrate_Num+1)*self.baseline_base] = np.float32(self.fitsfile[0].data.data[start: start+self.baseline_base, 0, 0, self.chan, polar, 1])

            print "h_uu", self.h_uu[0,:]
            print "h_rere:", self.h_rere[0, 0, 4, :]
            print "CLEAN START..."
            for self.band in range(self.band_start, self.band_end):
                for self.polar in range(0, 1):
                    for self.chan in range(self.chan_start, self.chan_end):
                        self.freq = self.basefreq + np.float32(self.fitsfile[1].data["IF FREQ"][self.band])

                        self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
                        self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

                        print "BAND:", self.band, "CHANNEL:", self.chan, "FREQ:", self.current_freq,

                        # the unit of uu and  vv is seconds
                        self.h_uu[self.band] *= self.current_freq
                        self.h_vv[self.band] *= self.current_freq

                        if self.freq == 400E6:
                            ISIZE = 256
                        elif self.freq == 800E6:
                            ISIZE = 512
                        elif self.freq in [1200E6, 1600E6]:
                            ISIZE = 1024

                        if self.correct_p_angle:
                            self.imsize = np.int32(ISIZE * 1.5)
                        else:
                            self.imsize = ISIZE
                        self.cell = self.angular_resolution / 3.
                        self.fov = self.cell * self.imsize
                        self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)

                        logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

                        if self.correct_p_angle:
                            self.imsize = np.int32(ISIZE * 2)  # number of image pixels
                        else:
                            self.imsize = np.int32(ISIZE)  # number of image pixels

                        self.gpu_info()
                        self.p_angle, b, sd = pb0r(self.obs_date[:-9])
                        self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.

                        x_offset, y_offset, sky, disk = self.preclean()
                        if x_offset % 2 == 1:
                            x_offset -= 1
                        if y_offset % 2 == 1:
                            y_offset -= 1
                        self.clean(x_offset, y_offset, sky, disk)

        else:
            for frame_Num in range(0, self.integration_number):
                start = frame_Num*self.baseline_base
                delta = self.baseline_base

                # UU, VV
                self.h_uu[frame_Num%self.band_number][frame_Num*self.baseline_base: (frame_Num+1)*self.baseline_base] = np.float64(self.fitsfile[0].data.par('uu')[start: start+delta])
                self.h_vv[frame_Num%self.band_number][frame_Num*self.baseline_base: (frame_Num+1)*self.baseline_base] = np.float64(self.fitsfile[0].data.par('vv')[start: start+delta])

                for chan in range(0, 16):
                    # Visibility Data
                    self.h_rere[frame_Num%self.band_number][chan][frame_Num*self.baseline_base: (frame_Num+1)*self.baseline_base] = np.float32(self.fitsfile[0].data.data[start: start+delta, 0, 0, chan, self.polarization, 0])
                    self.h_imim[frame_Num%self.band_number][chan][frame_Num*self.baseline_base: (frame_Num+1)*self.baseline_base] = np.float32(self.fitsfile[0].data.data[start: start+delta, 0, 0, chan, self.polarization, 1])

            # CLEAN START.
            for band in range(0, self.band_number):
                for chan in range(0, 16):
                    self.freq = self.basefreq + np.float32(self.fitsfile[1].data["IF FREQ"][band])
                    self.current_freq = self.freq + chan * self.bandwidth + self.bandwidth // 2
                    self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

                    gcount = np.int32(np.size(self.h_uu))
                    # the unit of uu and  vv is seconds
                    self.h_uu *= self.current_freq
                    self.h_vv *= self.current_freq

                    if self.freq == 400E6:
                        ISIZE = 256
                    elif self.freq == 800E6:
                        ISIZE = 512
                    elif self.freq in [1200E6, 1600E6]:
                        ISIZE = 1024

                    if self.correct_p_angle:
                        self.imsize = np.int32(ISIZE * 1.5)
                    else:
                        self.imsize = ISIZE
                    self.cell = self.angular_resolution / 3.
                    self.fov = self.cell * self.imsize
                    self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)

                    logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

                    if self.correct_p_angle:
                        self.imsize = np.int32(ISIZE * 2)  # number of image pixels
                    else:
                        self.imsize = np.int32(ISIZE)  # number of image pixels

                    self.gpu_info()
                    self.p_angle, b, sd = pb0r(self.obs_date[:-9])
                    self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.

                    x_offset, y_offset, sky, disk = self.preclean()
                    if x_offset % 2 == 1:
                        x_offset -= 1
                    if y_offset % 2 == 1:
                        y_offset -= 1
                    self.clean(x_offset, y_offset, sky, disk)


    ################################################################
    # Read visibility data from rawdata. MODE: Average or Interval #
    ################################################################
    def clean_integration_RAWDATA(self, subarray, start_time, end_time, mode, outdir, band, channel, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):

        self.subArray = subarray
        self.outdir = outdir
        self.mode = mode
        self.band = band
        self.plot_me = PLOT_ME
        self.writefits = WRITE_FITS
        self.Debug = DEBUG
        self.correct_p_angle = P_ANGLE

        self.obs_date = ''
        self.object = "MUSER"
        self.bandwidth = 25000000.0
        self.briggs = np.float32(1e7)  # weight parameter
        self.light_speed = 299792458.  # Speed of light

        LOW_FRE = {0: (400000000, 256), 1: (800000000, 512), 2: (1200000000, 1024), 3: (1600000000, 1024)}
        HIGH_FRE = {0: (2000000000, 1280), 1: (2400000000, 1280), 2: (2800000000, 1280), 3: (3200000000, 1280),
                    4: (3600000000, 1280),
                    5: (4000000000, 2560), 6: (4400000000, 2560), 7: (4800000000, 2560), 8: (5200000000, 2560),
                    9: (5600000000, 2560),
                    10: (6000000000, 2560), 11: (6400000000, 2560), 12: (6800000000, 2560), 13: (7200000000, 2560),
                    14: (7600000000, 2560),
                    15: (8000000000, 5120), 16: (8400000000, 5120), 17: (8800000000, 5120), 18: (9200000000, 5120),
                    19: (9600000000, 5120),
                    20: (10000000000, 5120), 21: (10400000000, 5120), 22: (10800000000, 5120), 23: (11200000000, 5120),
                    24: (11600000000, 5120),
                    25: (12000000000, 5120), 26: (12400000000, 5120), 27: (12800000000, 5120), 28: (13200000000, 5120),
                    29: (13600000000, 5120),
                    30: (14000000000, 5120), 31: (14400000000, 5120), 32: (14600000000, 5120)}

        uvfits = MuserFile(self.subArray)

        # Convert datetime to MUSER time
        self.start_frame_time = MuserTime()
        self.start_frame_time.set_with_date_time(start_time)
        self.end_frame_time = MuserTime()
        self.end_frame_time.set_with_date_time(end_time)

        obs_day = ('%4d-%02d-%02d') % (self.start_frame_time.year, self.start_frame_time.month,  self.start_frame_time.day)
        # print "Integretion INFO:", "MUSER-"+ '%d'%self.subArray, self.mode, self.start_frame_time.get_date_time(), self.end_frame_time.get_date_time()

        if self.subArray == 1:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, obs_day+" 00:00:00")
            self.baseline_base = 780
            self.band_number = 4
        else:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(2, obs_day+" 00:00:00")
            self.baseline_base = 1770
            self.band_number = 33

        #Find file according to start and end time
        filenamelist = uvfits.get_file_info(self.start_frame_time, self.end_frame_time)
        print "FILE:", filenamelist
        self.integrate_frame_num = uvfits.get_frame_info(self.start_frame_time, self.end_frame_time)
        self.baseline_number = self.integrate_frame_num*self.baseline_base
        print "integration NUMBER(each band):", self.integrate_frame_num
        print "Baseline NUMBER:", self.baseline_number

        if self.band != None:
            self.band_start = self.band
            self.band_end = self.band + 1
        else:
            self.band_start = 0
            self.band_end = self.band_number

        if channel != None:
            self.chan_start = channel
            self.chan_end = channel + 1
        else:
            self.chan_start = 0
            self.chan_end = 16

        self.h_uu = np.zeros(shape=(self.band_number, 2, self.baseline_number), dtype='float64')
        self.h_vv = np.zeros(shape=(self.band_number, 2, self.baseline_number), dtype='float64')
        self.h_rere = np.zeros(shape=(self.band_number, 2, 16, self.baseline_number), dtype='float32')
        self.h_imim = np.zeros(shape=(self.band_number, 2, 16, self.baseline_number), dtype='float32')

        #Reading Data from rawdata files
        (framenum1, framenum2, framenum3, framenum4, framenum5, framenum6, framenum7,framenum8)  = (0, 0, 0, 0, 0, 0, 0, 0)
        for self.filename in filenamelist:
            if not os.path.exists(self.filename):
                logger.error("No file exist: %s." % self.filename)
                return
            if uvfits.open_raw_file(self.filename) == False:
                print "Cannot open observational data."
                return

            uvfits.in_file.seek(0,0)
            if self.filename == filenamelist[0]:
                if uvfits.search_first_frame() == False:
                    logger.error("Cannot find observational data.")
                    return str(False), []
                self.obs_date = uvfits.current_frame_time.get_fits_date_time()
                self.muser_date = datetime.datetime.strptime(self.obs_date[:-3], "%Y-%m-%dT%H:%M:%S.%f")
                self.ra = uvfits.ra_sum
                self.dec = uvfits.dec_sum

            while True:

                print framenum1, uvfits.current_frame_time.get_date_time(), self.end_frame_time.get_date_time()

                if self.filename == filenamelist[-1]:
                    print uvfits.current_frame_time.get_date_time() > self.end_frame_time.get_date_time()
                    if uvfits.current_frame_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

                uvfits.read_data()
                uvfits.calibration()
                uvw_data, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)

                if uvfits.sub_band == 0 and uvfits.polarization == 0:
                    self.h_uu[0, 0, framenum1*self.baseline_base: (framenum1+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[0, 0, framenum1*self.baseline_base: (framenum1+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[0, 0, chan, framenum1*self.baseline_base: (framenum1+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[0, 0, chan, framenum1*self.baseline_base: (framenum1+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum1 += 1
                elif uvfits.sub_band == 0 and uvfits.polarization == 1:
                    self.h_uu[0, 1, framenum2*self.baseline_base: (framenum2+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[0, 1, framenum2*self.baseline_base: (framenum2+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[0, 1, chan, framenum2*self.baseline_base: (framenum2+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[0, 1, chan, framenum2*self.baseline_base: (framenum2+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum2 += 1
                elif uvfits.sub_band == 1 and uvfits.polarization == 0:
                    self.h_uu[1, 0, framenum3*self.baseline_base: (framenum3+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[1, 0, framenum3*self.baseline_base: (framenum3+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[1, 0, chan, framenum3*self.baseline_base: (framenum3+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[1, 0, chan, framenum3*self.baseline_base: (framenum3+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum3 += 1
                elif uvfits.sub_band == 1 and uvfits.polarization == 1:
                    self.h_uu[1, 1, framenum4*self.baseline_base: (framenum4+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[1, 1, framenum4*self.baseline_base: (framenum4+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[1, 1, chan, framenum4*self.baseline_base: (framenum4+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[1, 1, chan, framenum4*self.baseline_base: (framenum4+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum4 += 1
                elif uvfits.sub_band == 2 and uvfits.polarization == 0:
                    self.h_uu[2, 0, framenum5*self.baseline_base: (framenum5+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[2, 0, framenum5*self.baseline_base: (framenum5+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[2, 0, chan, framenum5*self.baseline_base: (framenum5+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[2, 0, chan, framenum5*self.baseline_base: (framenum5+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum5 += 1
                elif uvfits.sub_band == 2 and uvfits.polarization == 1:
                    self.h_uu[2, 1, framenum6*self.baseline_base: (framenum6+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[2, 1, framenum6*self.baseline_base: (framenum6+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[2, 1, chan, framenum6*self.baseline_base: (framenum6+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[2, 1, chan, framenum6*self.baseline_base: (framenum6+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum6 += 1
                elif uvfits.sub_band == 3 and uvfits.polarization == 0:
                    self.h_uu[3, 0, framenum7*self.baseline_base: (framenum7+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[3, 0, framenum7*self.baseline_base: (framenum7+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[3, 0, chan, framenum7*self.baseline_base: (framenum7+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[3, 0, chan, framenum7*self.baseline_base: (framenum7+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum7 += 1
                elif uvfits.sub_band == 3 and uvfits.polarization == 1:
                    self.h_uu[3, 1, framenum8*self.baseline_base: (framenum8+1)*self.baseline_base] = uvw_data[:, 0]
                    self.h_vv[3, 1, framenum8*self.baseline_base: (framenum8+1)*self.baseline_base] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        self.h_rere[3, 1, chan, framenum8*self.baseline_base: (framenum8+1)*self.baseline_base] = uvfits.baseline_data[:, chan].real
                        self.h_imim[3, 1, chan, framenum8*self.baseline_base: (framenum8+1)*self.baseline_base] = uvfits.baseline_data[:, chan].imag
                    framenum8 += 1

                uvfits.read_one_frame()


        print self.h_rere
        print "UU", self.h_uu

        # CLEAN START
        for self.band in range(self.band_start, self.band_end):
            for self.polar in range(0, 2):
                for self.chan in range(self.chan_start, self.chan_end):
                    self.freq = LOW_FRE[self.band]
                    self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
                    self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

                    gcount = np.int32(np.size(self.h_uu))
                    # the unit of uu and  vv is seconds
                    self.h_uu[self.band, self.polar] *= self.current_freq
                    self.h_vv[self.band, self.polar] *= self.current_freq

                    if self.freq == 400E6:
                        ISIZE = 256
                    elif self.freq == 800E6:
                        ISIZE = 512
                    elif self.freq in [1200E6, 1600E6]:
                        ISIZE = 1024

                    if self.correct_p_angle:
                        self.imsize = np.int32(ISIZE * 1.5)
                    else:
                        self.imsize = ISIZE
                    self.cell = self.angular_resolution / 3.
                    self.fov = self.cell * self.imsize
                    self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)

                    logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

                    if self.correct_p_angle:
                        self.imsize = np.int32(ISIZE * 2)  # number of image pixels
                    else:
                        self.imsize = np.int32(ISIZE)  # number of image pixels

                    self.gpu_info()
                    self.p_angle, b, sd = pb0r(self.obs_date[:-9])
                    self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.

                    x_offset, y_offset, sky, disk = self.preclean()
                    if x_offset % 2 == 1:
                        x_offset -= 1
                    if y_offset % 2 == 1:
                        y_offset -= 1
                    self.clean(x_offset, y_offset, sky, disk)



    #####################################################################################
    # Read visibility data from Generated uvw and visibility file, and clean            #
    #####################################################################################
    def clean_integration_R(self, sub_ARRAY, is_loop_mode, start_time, end_time, TASK_TYPE, time_average, time_interval, BAND, CHANNEL, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG, outdir):
        # Read visibility data from rawdata file and clean

        self.subarray = sub_ARRAY
        self.is_loop_mode = is_loop_mode
        self.task_type = TASK_TYPE
        self.time_average = time_average
        self.time_interval = time_interval
        self.band = BAND
        self.plot_me = PLOT_ME
        self.writefits = WRITE_FITS
        self.correct_p_angle = P_ANGLE
        self.Debug = DEBUG
        self.outdir = outdir

        self.obs_date = ''
        self.object = "MUSER"
        self.bandwidth = 25000000.0
        self.briggs = np.float32(1e7)  # weight parameter
        self.light_speed = 299792458.  # Speed of light


        uvfits = MuserFile(self.subarray)

        # Convert datetime to MUSER time
        self.start_frame_time = MuserTime()
        self.start_frame_time.set_with_date_time(start_time)
        self.end_frame_time = MuserTime()
        self.end_frame_time.set_with_date_time(end_time)

        uvfits.set_data_date_time(self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day, self.start_frame_time.hour, self.start_frame_time.minute,
                                self.start_frame_time.second, self.start_frame_time.millisecond, self.start_frame_time.microsecond,
                                self.start_frame_time.nanosecond)
        obs_day = ('%4d-%02d-%02d') % (self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day)
        if self.subarray == 1:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, obs_day + " 00:00:00")
            self.baseline_base = 780
            self.band_number = 4
        else:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(2, obs_day + " 00:00:00")
            self.baseline_base = 1770
            self.band_number = 33
        print "FLag Antennas: %s " % self.Flag_Ant


        self.integrate_frame_num = int((end_time - start_time).seconds/self.time_interval) + 1
        print "integrate_frame_num:", self.integrate_frame_num

        if self.is_loop_mode == True:
            if self.band != None:
                self.band_start = self.band
                self.band_end = self.band + 1
            else:
                self.band_start = 0
                self.band_end = self.band_number

            if CHANNEL != None:
                self.chan_start = CHANNEL
                self.chan_end = CHANNEL + 1
            else:
                self.chan_start = 0
                self.chan_end = 16

            UU = np.zeros(shape=(self.integrate_frame_num, self.band_number, 2, self.baseline_base), dtype='float64')
            VV = np.zeros(shape=(self.integrate_frame_num, self.band_number, 2, self.baseline_base), dtype='float64')
            REAL = np.zeros(shape=(self.integrate_frame_num, self.band_number, 2, 16, self.baseline_base), dtype='float64')
            IMAG = np.zeros(shape=(self.integrate_frame_num, self.band_number, 2, 16, self.baseline_base), dtype='float64')

            h_uu = np.zeros(shape=(self.band_number, 2, self.baseline_base*self.integrate_frame_num), dtype='float64')
            h_vv = np.zeros(shape=(self.band_number, 2, self.baseline_base*self.integrate_frame_num), dtype='float64')
            h_rere = np.zeros(shape=(self.band_number, 2, 16, self.baseline_base*self.integrate_frame_num), dtype='float64')
            h_imim = np.zeros(shape=(self.band_number, 2, 16, self.baseline_base*self.integrate_frame_num), dtype='float64')

        else:
            if self.band != None:
                if self.Debug:
                    logger.info("This is NON LOOP MODE, Band choose is NOT available.")
                print "This is NON LOOP MODE, Band choose is NOT available."

            if CHANNEL != None:
                self.chan_start = CHANNEL
                self.chan_end = CHANNEL + 1
            else:
                self.chan_start = 0
                self.chan_end = 16

            UU = np.zeros(shape=(self.integrate_frame_num, self.baseline_base), dtype='float64')
            VV = np.zeros(shape=(self.integrate_frame_num, self.baseline_base), dtype='float64')
            REAL = np.zeros(shape=(self.integrate_frame_num, 16, self.baseline_base), dtype='float64')
            IMAG = np.zeros(shape=(self.integrate_frame_num, 16, self.baseline_base), dtype='float64')

            h_uu = np.zeros(shape=(self.baseline_base*self.integrate_frame_num), dtype='float64')
            h_vv = np.zeros(shape=(self.baseline_base*self.integrate_frame_num), dtype='float64')
            h_rere = np.zeros(shape=(16, self.baseline_base*self.integrate_frame_num), dtype='float64')
            h_imim = np.zeros(shape=(16, self.baseline_base*self.integrate_frame_num), dtype='float64')

        # Reading Data from rawdata files
        if uvfits.search_first_frame() == False:
            logger.error("Cannot find observational data.")

        self.obs_date = uvfits.current_frame_time.get_fits_date_time()
        # print uvfits.in_file
        # self.filename = uvfits.in_file
        self.muser_date = datetime.datetime.strptime(self.obs_date[:-3], "%Y-%m-%dT%H:%M:%S.%f")
        self.ra = uvfits.ra_sum
        self.dec = uvfits.dec_sum
        self.freq = uvfits.frequency
        self.polarization = uvfits.polarization

        uvfits.set_priority(0)
        uvfits.load_calibration_data()
        offset = [100000, 204800]

        niter = 0
        if self.is_loop_mode == True:
            (framenum1, framenum2, framenum3, framenum4, framenum5, framenum6, framenum7,framenum8) = (0, 0, 0, 0, 0, 0, 0, 0)
            if self.task_type == "average":
                while True:
                    print "FRAME INFO:", uvfits.polarization, uvfits.sub_band, uvfits.current_frame_time.get_date_time()

                    uvfits.read_data()
                    uvfits.calibration()
                    uvw_data, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)
                    self.previous_time = uvfits.current_frame_time

                    if uvfits.sub_band == 0 and uvfits.polarization == 0:
                        UU[niter, 0, 0, :] += uvw_data[:, 0]
                        VV[niter, 0, 0, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 0, 0, chan, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 0, 0, chan, :] += uvfits.baseline_data[:, chan].imag
                        framenum1 += 1
                    elif uvfits.sub_band == 0 and uvfits.polarization == 1:
                        UU[niter, 0, 1, :] += uvw_data[:, 0]
                        VV[niter, 0, 1, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 0, 1, chan, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 0, 1, chan, :] += uvfits.baseline_data[:, chan].imag
                        framenum2 += 1
                    elif uvfits.sub_band == 1 and uvfits.polarization == 0:
                        UU[niter, 1, 0, :] += uvw_data[:, 0]
                        VV[niter, 1, 0, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 1, 0, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 1, 0, :] += uvfits.baseline_data[:, chan].imag
                        framenum3 += 1
                    elif uvfits.sub_band == 1 and uvfits.polarization == 1:
                        UU[niter, 1, 1, :] += uvw_data[:, 0]
                        VV[niter, 1, 1, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 1, 1, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 1, 1, :] += uvfits.baseline_data[:, chan].imag
                        framenum4 += 1
                    elif uvfits.sub_band == 2 and uvfits.polarization == 0:
                        UU[niter, 2, 0, :] += uvw_data[:, 0]
                        VV[niter, 2, 0, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 2, 0, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 2, 0, :] += uvfits.baseline_data[:, chan].imag
                        framenum5 += 1
                    elif uvfits.sub_band == 2 and uvfits.polarization == 1:
                        UU[niter, 2, 1, :] += uvw_data[:, 0]
                        VV[niter, 2, 1, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 2, 1, chan, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 2, 1, chan, :] += uvfits.baseline_data[:, chan].imag
                        framenum6 += 1
                    elif uvfits.sub_band == 3 and uvfits.polarization == 0:
                        UU[niter, 3, 0, :] += uvw_data[:, 0]
                        VV[niter, 3, 0, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 3, 0, chan, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 3, 0, chan, :] += uvfits.baseline_data[:, chan].imag
                        framenum7 += 1
                    elif uvfits.sub_band == 3 and uvfits.polarization == 1:
                        UU[niter, 3, 1, :] += uvw_data[:, 0]
                        VV[niter, 3, 1, :] += uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, 3, 1, chan, :] += uvfits.baseline_data[:, chan].real
                            IMAG[niter, 3, 1, chan, :] += uvfits.baseline_data[:, chan].imag
                        framenum8 += 1

                    print uvfits.current_frame_time.get_date_time(), self.previous_time.get_date_time()
                    t_offset = uvfits.current_frame_time.get_date_time() - self.previous_time.get_date_time()
                    time_offset = t_offset.seconds * 1e6 + t_offset.microseconds
                    print t_offset
                    if time_offset >= self.time_interval*1e6:
                        print "Reached the time interval:", time_offset
                        print framenum1, framenum2, framenum3, framenum4, framenum5, framenum6, framenum7, framenum8
                        self.previous_time = uvfits.current_frame_time

                        REAL[niter, 0, 0, :] = REAL[niter, 0, 0, :]/framenum1
                        IMAG[niter, 0, 0, :] = IMAG[niter, 0, 0, :]/framenum1
                        UU[niter, 0, 0, :] = UU[niter, 0, 0, :]/framenum1
                        VV[niter, 0, 0, :] = VV[niter, 0, 0, :]/framenum1

                        REAL[niter, 0, 1, :] = REAL[niter, 0, 1, :]/framenum2
                        IMAG[niter, 0, 1, :] = IMAG[niter, 0, 1, :]/framenum2
                        UU[niter, 0, 1, :] = UU[niter, 0, 1, :]/framenum2
                        VV[niter, 0, 1, :] = VV[niter, 0, 1, :]/framenum2

                        REAL[niter, 1, 0, :] = REAL[niter, 1, 0, :]/framenum3
                        REAL[niter, 1, 0, :] = REAL[niter, 1, 0, :]/framenum3
                        UU[niter, 1, 0, :] = UU[niter, 1, 0, :]/framenum3
                        VV[niter, 1, 0, :] = VV[niter, 1, 0, :]/framenum3

                        REAL[niter, 1, 1, :] = REAL[niter, 1, 1, :]/framenum4
                        REAL[niter, 1, 1, :] = REAL[niter, 1, 1, :]/framenum4
                        UU[niter, 1, 1, :] = UU[niter, 1, 1, :]/framenum4
                        VV[niter, 1, 1, :] = VV[niter, 1, 1, :]/framenum4

                        REAL[niter, 2, 0, :] = REAL[niter, 2, 0, :]/framenum5
                        REAL[niter, 2, 0, :] = REAL[niter, 2, 0, :]/framenum5
                        UU[niter, 2, 0, :] = UU[niter, 2, 0, :]/framenum5
                        VV[niter, 2, 0, :] = VV[niter, 2, 0, :]/framenum5

                        REAL[niter, 2, 1, :] = REAL[niter, 2, 1, :]/framenum6
                        REAL[niter, 2, 1, :] = REAL[niter, 2, 1, :]/framenum6
                        UU[niter, 2, 1, :] = UU[niter, 2, 1, :]/framenum6
                        VV[niter, 2, 1, :] = VV[niter, 2, 1, :]/framenum6

                        REAL[niter, 3, 0, :] = REAL[niter, 3, 0, :]/framenum7
                        REAL[niter, 3, 0, :] = REAL[niter, 3, 0, :]/framenum7
                        UU[niter, 3, 0, :] = UU[niter, 3, 0, :]/framenum7
                        VV[niter, 3, 0, :] = VV[niter, 3, 0, :]/framenum7

                        REAL[niter, 3, 1, :] = REAL[niter, 3, 1, :]/framenum8
                        REAL[niter, 3, 1, :] = REAL[niter, 3, 1, :]/framenum8
                        UU[niter, 3, 1, :] = UU[niter, 3, 1, :]/framenum8
                        VV[niter, 3, 1, :] = VV[niter, 3, 1, :]/framenum8

                        niter += 1
                        (framenum1, framenum2, framenum3, framenum4, framenum5, framenum6, framenum7,framenum8) = (0, 0, 0, 0, 0, 0, 0, 0)

                        time_temp = self.previous_bigframe_time.get_date_time() + datetime.timedelta(seconds=self.time_interval)
                        uvfits.start_date_time.set_with_date_time(time_temp)
                        print "Next time:", uvfits.start_date_time.get_date_time()


                    if uvfits.check_next_file() == True:
                        uvfits.open_next_file()

                    uvfits.read_one_frame()
                    if uvfits.current_frame_time.get_date_time() > self.end_frame_time.get_date_time():
                        break


            elif self.task_type == "interval":
                print "TIME SPAN:", uvfits.current_frame_time.get_date_time(), self.end_frame_time.get_date_time()
                while True:
                    while uvfits.sub_band != 0 or uvfits.polarization != 0:
                        uvfits.in_file.seek(100000-192, 1)

                        if uvfits.check_next_file() == True:
                            uvfits.open_next_file()
                        uvfits.read_one_frame()
                    self.previous_bigframe_time = uvfits.current_frame_time
                    self.previous_time = uvfits.current_frame_time

                    (band, polar) = (0, 0)
                    (repeat, skip_frame_number) = (0, 0)
                    if self.Debug:
                        logger.info("Reading one big frame at %s" % uvfits.current_frame_time.get_string())
                    while repeat<self.band_number*2:
                        uvfits.read_data()
                        # Consider the condition of frame skipped.
                        t_offset = uvfits.current_frame_time.get_date_time() - self.previous_time.get_date_time()
                        time_offset = t_offset.seconds * 1e6 + t_offset.microseconds
                        if time_offset > 3125:
                            skip_frame_number = int(time_offset/3125)
                            for skip_num in range(0, skip_frame_number-1):
                                UU[niter, band, polar, :] = 0.
                                VV[niter, band, polar, :] = 0.
                                for chan in range(0, 16):
                                    # Visibility Data
                                    REAL[niter, band, polar, chan, :] = 0.
                                    IMAG[niter, band, polar, chan, :] = 0.
                                polar += 1
                                if polar == 2:
                                    band +=1
                                    polar = 0
                                if band == self.band_number:
                                    break
                            repeat += skip_frame_number
                            tmp = uvfits.current_frame_time.get_date_time() + timedelta(microseconds=3125*skip_frame_number)
                            self.previous_time.set_with_date_time(tmp)

                        self.previous_time = uvfits.current_frame_time

                        uvfits.calibration()
                        uvw_data, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)
                        UU[niter, band, polar, :] = uvw_data[:, 0]
                        VV[niter, band, polar, :] = uvw_data[:, 1]
                        for chan in range(0, 16):
                            # Visibility Data
                            REAL[niter, band, polar, chan, :] = uvfits.baseline_data[:, chan].real
                            IMAG[niter, band, polar, chan, :] = uvfits.baseline_data[:, chan].imag
                        polar += 1
                        if polar == 2:
                            band +=1
                            polar = 0
                        if band == self.band_number:
                            break

                        if uvfits.check_next_file() == True:
                            uvfits.open_next_file()
                        uvfits.read_one_frame()

                        if uvfits.current_frame_time.get_date_time() > self.end_frame_time.get_date_time():
                            if not (uvfits.sub_band == 0 and uvfits.polarization == 0):
                                niter -= 1
                            break
                    niter += 1
                    time_temp = self.previous_bigframe_time.get_date_time() + datetime.timedelta(seconds=self.time_interval)
                    uvfits.start_date_time.set_with_date_time(time_temp)
                    print "Next time:", uvfits.start_date_time.get_date_time()

                    uvfits.in_file.seek((self.time_interval*1e3/3.125+1)*100000, 1)

                    if uvfits.check_next_file() == True:
                        uvfits.open_next_file()

                    if uvfits.search_first_frame() == False:
                        logger.error("Cannot find observational data.")
                    if uvfits.current_frame_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

            (niter, band, polar) = (0, 0, 0)
            for niter in range(0, self.integrate_frame_num):
                for band in range(0, self.band_number):
                    for polar in range(0, 2):
                        h_uu[band, polar, niter * self.baseline_base:(niter + 1) * self.baseline_base] = UU[niter, band, polar, :]
                        h_vv[band, polar, niter * self.baseline_base:(niter + 1) * self.baseline_base] = VV[niter, band, polar, :]

                        for chan in range(0, 16):
                            h_rere[band, polar, chan, niter * self.baseline_base:(niter + 1) * self.baseline_base] = REAL[niter, band, polar, chan, :]
                            h_imim[band, polar, chan, niter * self.baseline_base:(niter + 1) * self.baseline_base] = IMAG[niter, band, polar, chan, :]

            # CLEAN START
            for self.band in range(self.band_start, self.band_end):
                for self.polar in range(0, 2):
                    for self.chan in range(self.chan_start, self.chan_end):
                        self.freq = FRE_LOW[self.band][0]
                        self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
                        self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

                        h_uu[self.band, self.polar] *= self.current_freq
                        h_vv[self.band, self.polar] *= self.current_freq
                        self.h_uu = np.float32(h_uu[self.band, self.polar].ravel())
                        self.h_vv = np.float32(h_vv[self.band, self.polar].ravel())

                        self.h_rere = np.float32(h_rere[self.band, self.polar, self.chan].ravel())
                        self.h_imim = np.float32(h_imim[self.band, self.polar, self.chan].ravel())


                        if self.correct_p_angle:
                            self.imsize = np.int32(self.imsize * 1.5)

                        self.cell = self.angular_resolution / 3.
                        self.fov = self.cell * self.imsize
                        self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)

                        logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

                        self.gpu_info()
                        self.p_angle, b, sd = pb0r(self.obs_date[:-9])
                        self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.

                        x_offset, y_offset, sky, disk = self.preclean()
                        if x_offset % 2 == 1:
                            x_offset -= 1
                        if y_offset % 2 == 1:
                            y_offset -= 1
                        self.clean(x_offset, y_offset, sky, disk)

        else: # self.is_loop_mode == False
            if self.task_type == "average":
                framenum = 0
                while True:
                    print "FRAME INFO:", uvfits.polarization, uvfits.sub_band, uvfits.current_frame_time.get_date_time()

                    uvfits.read_data()
                    if uvfits.current_frame_header.strip_switch == 0xCCCCCCCC:
                        uvfits.delay_process('sun')

                    uvfits.calibration()
                    uvw_data, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)

                    UU[niter, :] += uvw_data[:, 0]
                    VV[niter, :] += uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        REAL[niter, chan, :] += uvfits.baseline_data[:, chan].real
                        IMAG[niter, chan, :] += uvfits.baseline_data[:, chan].imag

                    framenum += 1

                    print uvfits.current_frame_time.get_date_time(), self.previous_time.get_date_time()
                    t_offset = uvfits.current_frame_time.get_date_time() - self.previous_time.get_date_time()
                    time_offset = t_offset.seconds * 1e6 + t_offset.microseconds
                    print t_offset
                    if time_offset >= self.time_interval * 1e6:
                        print "Reached the time interval:", time_offset
                        self.previous_time = uvfits.current_frame_time

                        REAL[niter, :, :] = REAL[niter, :, :] / framenum
                        IMAG[niter, :, :] = IMAG[niter, :, :] / framenum
                        UU[niter, :] = UU[niter, :] / framenum
                        VV[niter, :] = VV[niter, :] / framenum
                        niter += 1
                        framenum = 0

                    if uvfits.check_next_file() == True:
                        uvfits.open_next_file()
                    uvfits.read_one_frame()
                    if uvfits.current_frame_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

            elif self.task_type == "interval":
                print "TIME SPAN:", uvfits.current_frame_time.get_date_time(), self.end_frame_time.get_date_time()
                while True:
                    uvfits.read_one_frame()
                    self.previous_bigframe_time = uvfits.current_frame_time
                    self.previous_time = uvfits.current_frame_time

                    (repeat, skip_frame_number) = (0, 0)
                    if self.Debug:
                        logger.info("Reading one frame at %s" % uvfits.current_frame_time.get_string())

                    uvfits.read_data()
                    # DO NOT need to consider the condition of frame skipped.
                    self.previous_time = uvfits.current_frame_time

                    if uvfits.current_frame_header.strip_switch == 0xCCCCCCCC:
                        uvfits.delay_process('sun')

                    uvfits.calibration()
                    uvw_data, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)
                    UU[niter, :] = uvw_data[:, 0]
                    VV[niter, :] = uvw_data[:, 1]
                    for chan in range(0, 16):
                        # Visibility Data
                        REAL[niter, chan, :] = uvfits.baseline_data[:, chan].real
                        IMAG[niter, chan, :] = uvfits.baseline_data[:, chan].imag

                    niter += 1
                    time_temp = self.previous_bigframe_time.get_date_time() + datetime.timedelta(seconds=self.time_interval)
                    uvfits.start_date_time.set_with_date_time(time_temp)
                    uvfits.first_date_time.set_with_date_time(time_temp)
                    print "Next time:", uvfits.start_date_time.get_date_time(), self.end_frame_time.get_date_time()

                    if uvfits.start_date_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

                    if uvfits.search_first_frame() == False:
                        logger.error("Cannot find observational data.")

            elif self.task_type == "mixture":

                print "TIME SPAN:", uvfits.current_frame_time.get_date_time(), self.end_frame_time.get_date_time()

                while True:
                    uvfits.read_one_frame()
                    self.previous_bigframe_time = uvfits.current_frame_time
                    self.previous_time = uvfits.current_frame_time
                    time_offset = 0
                    framenum = 0
                    dra = 0.14 # 0.14 # degree
                    ddec = -0.1 # -0.33


                    while time_offset < self.time_average:
                        uvfits.read_data()
                        if uvfits.current_frame_header.strip_switch == 0xCCCCCCCC:
                            uvfits.delay_process('sun')
                        uvfits.calibration()
                        uvw_data, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)

                        UU[niter, :] += uvw_data[:, 0]
                        VV[niter, :] += uvw_data[:, 1]

                        for chan in range(0, 16):
                            # Visibility Data
                            for baseline in range(0, self.baseline_base):
                                REAL[niter, chan, baseline] += uvfits.baseline_data[baseline, chan].real
                                IMAG[niter, chan, baseline] += uvfits.baseline_data[baseline, chan].imag

                        framenum += 1
                        time_offset += 3 #miliseconds

                        if uvfits.check_next_file() == True:
                            uvfits.open_next_file()
                        uvfits.read_one_frame()
                        if time_offset >= self.time_average:
                            print "Reached the time interval:", time_offset, framenum
                            DEC = ddec + niter * 0.02
                            RA = dra/180.*np.pi   #radian
                            DEC = DEC / 180. * np.pi

                            for chan in range(0, 16):
                                if self.subarray == 1:
                                    self.current_freq = self.freq + chan * self.bandwidth + self.bandwidth // 2
                                else:
                                    self.current_freq = self.freq + (15 - chan) * self.bandwidth + self.bandwidth // 2

                                for baseline in range(0, self.baseline_base):

                                    A = math.sqrt(
                                        (REAL[niter, chan, baseline]/framenum) * (REAL[niter, chan, baseline]/framenum) +
                                        (IMAG[niter, chan, baseline]/framenum) * (IMAG[niter, chan, baseline]/framenum))
                                    phai_sun = math.atan2(IMAG[niter, chan, baseline]/framenum, REAL[niter, chan, baseline]/framenum)
                                    phai = phai_sun + 2*np.pi*(UU[niter, baseline]/framenum *self.current_freq*RA+VV[niter, baseline]/framenum *self.current_freq*DEC)
                                    REAL[niter, chan, baseline] = A * math.cos(phai)
                                    IMAG[niter, chan, baseline] = A * math.sin(phai)
                            print "Phase Calibration DONE."

                            niter += 1
                            break

                    time_temp = self.previous_bigframe_time.get_date_time() + datetime.timedelta(seconds=self.time_interval)
                    uvfits.start_date_time.set_with_date_time(time_temp)
                    uvfits.first_date_time.set_with_date_time(time_temp)
                    print "Next time:", uvfits.start_date_time.get_date_time(), self.end_frame_time.get_date_time()

                    if uvfits.start_date_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

                    if uvfits.search_first_frame() == False:
                        logger.error("Cannot find observational data.")

            for niter in range(0, self.integrate_frame_num):
                    h_uu[niter * self.baseline_base:(niter + 1) * self.baseline_base] = UU[niter, :]/framenum
                    h_vv[niter * self.baseline_base:(niter + 1) * self.baseline_base] = VV[niter, :]/framenum

                    for chan in range(0, 16):
                        h_rere[chan, niter * self.baseline_base:(niter + 1) * self.baseline_base] = REAL[niter, chan, :]
                        h_imim[chan, niter * self.baseline_base:(niter + 1) * self.baseline_base] = IMAG[niter, chan, :]

            # CLEAN START
            for self.chan in range(self.chan_start, self.chan_end):
                if self.subarray == 1:
                    self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
                else:
                    self.current_freq = self.freq + (15 - self.chan) * self.bandwidth + self.bandwidth // 2

                self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

                # the unit of uu and  vv is seconds
                h_uu *= self.current_freq
                h_vv *= self.current_freq
                self.h_uu = np.float32(h_uu.ravel())
                self.h_vv = np.float32(h_vv.ravel())

                self.h_rere = np.float32(h_rere[self.chan].ravel())
                self.h_imim = np.float32(h_imim[self.chan].ravel())

                if self.subarray == 1:
                    if self.freq == 400E6:
                        ISIZE = 256
                    elif self.freq == 800E6:
                        ISIZE = 512
                    elif self.freq in [1200E6, 1600E6]:
                        ISIZE = 1024
                else:
                    if self.freq in [2000E6, 3600E6]:
                        ISIZE = 1280
                    elif self.freq in [4000E6, 7600E6]:
                        ISIZE = 2560
                    elif self.freq in [8000E6, 15000E6]:
                        ISIZE = 5120


                if self.correct_p_angle:
                    self.imsize = np.int32(ISIZE * 1.5)
                else:
                    self.imsize = np.int32(ISIZE)

                self.cell = self.angular_resolution / 3.
                self.fov = self.cell * self.imsize
                self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)

                print ('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))


                self.gpu_info()
                self.p_angle, b, sd = pb0r(self.obs_date[:-9])
                self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.
                self.prefix = self.start_frame_time.get_short_string() + ' TO ' + self.end_frame_time.get_short_string()

                x_offset, y_offset, sky, disk = self.preclean()
                print "x_offset", x_offset, "y_offset", y_offset
                if x_offset % 2 == 1:
                    x_offset -= 1
                if y_offset % 2 == 1:
                    y_offset -= 1
                self.clean(x_offset, y_offset, sky, disk)
                print "CLEAN finished and images saved."





