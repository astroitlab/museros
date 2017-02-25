import numpy as np
import time, pdb, sys, pyfits
import math
import os
import sys

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import skcuda.fft as fft
import skcuda.misc as misc
import gaussfitter as gf
import logging
import gausspeak as peakutils

import matplotlib
import matplotlib.image as img
import matplotlib.cm as cm
from scipy import ndimage

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
from muserenv import *
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


class MuserDirty:
    ######################
    # CUDA kernels


    def __init__(self):

        self.muser_draw = MuserDraw()
        self.muser_fits = MuserFits()
        self.muser_ant = MuserAntenna()
        self.muser_fits = MuserFits()
        self.muser_env = MuserEnv()

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

        h_uu = np.float32(self.h_uu.ravel())
        h_vv = np.float32(self.h_vv.ravel())
        h_rere = np.float32(self.h_rere.ravel())
        h_imim = np.float32(self.h_imim.ravel())

        blen = 0
        bl_order = np.ndarray(shape=(self.baseline_number, 2), dtype=int)
        good = []

        if self.baseline_number == 780:  # MUSER-I
            antennas = 40
        else:
            antennas = 60
        # print antennas
        for border1 in range(0, antennas - 1):
            for border2 in range(border1 + 1, antennas):
                bl_order[blen][0] = border1
                bl_order[blen][1] = border2
                blen = blen + 1

        h_u = []
        h_v = []
        h_re = []
        h_im = []
        for blen in range(0, self.baseline_number):
            if (bl_order[blen][0] not in self.Flag_Ant) and (bl_order[blen][1] not in self.Flag_Ant):
                good.append(blen)

                h_u.append(h_uu[blen])
                h_v.append(h_vv[blen])
                h_re.append(h_rere[blen])
                h_im.append(h_imim[blen])

        gcount = np.int32(np.size(h_u))

        # h_ : host,  d_ : device
        # h_grd = np.zeros((nx, nx), dtype=np.complex64)
        # h_cnt = np.zeros((nx, nx), dtype=np.int32)
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
        self.wgtGrid_kernel(d_bm, d_cnt, self.briggs, nx, 0, block=self.blocksize_2D, grid=self.gridsize_2D)
        hfac = np.int32(1)
        self.dblGrid_kernel(d_bm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
        self.dblGrid_kernel(d_cbm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)

        self.shiftGrid_kernel(d_bm, d_nbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        self.shiftGrid_kernel(d_cbm, d_bm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        ## normalize
        self.wgtGrid_kernel(d_grd, d_cnt, self.briggs, nx, 0, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Reflect grid about v axis
        hfac = np.int32(-1)
        self.dblGrid_kernel(d_grd, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Shift both
        self.shiftGrid_kernel(d_grd, d_ngrd, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        fft.fft(d_ngrd, d_grd, plan)
        ## Shift
        self.shiftGrid_kernel(d_grd, d_ngrd, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Correct for C
        self.corrGrid_kernel(d_ngrd, d_corr, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Trim
        self.trimIm_kernel(d_ngrd, d_dim, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        self.copyIm_kernel(d_ngrd, d_fbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        ## Normalize (Jy/beam)i
        # self.nrmGrid_kernel(d_dim, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        # self.nrmGrid_kernel(d_fbm, bmax2, nx, block=self.blocksize_2D, grid=self.gridsize_2D)

        ## Finish timers
        t_end = time.time()
        t_full = t_end - t_start
        logger.debug("Gridding execution time %0.5f" % t_full + ' s')
        logger.debug("\t%0.5f" % (t_full / gcount) + ' s per visibility')

        # ----------------------

        ## Return dirty psf (CPU), dirty image (GPU) and sun disk

        return d_dim


    def write_fits(self, data, fitsfile, type):
        # create_fits(self, object, obs_date, obs_time,data, imagetype):
        self.muser_fits.create_fits(data, self.object, self.muser_date.strftime("%Y-%m-%d"),
                                    self.muser_date.strftime("%H:%M:%S.%f"), type)
        self.muser_fits.append_common_header(self.current_freq, self.polarization, self.ra, self.dec, self.p_angle)
        self.muser_fits.write_fits(self.outdir, fitsfile)

    def preclean(self):

        nx = np.int32(2 * self.imsize)

        # create fft plan nx*nx
        self.plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)
        d_dirty = gpu.zeros((np.int(self.imsize), np.int(self.imsize)), np.float32)
        gpu_im = self.cuda_gridvis(self.plan, 0, 0)

        dirty = gpu_im.get()

        if self.Debug:
            logger.debug("Plotting dirty image")

        if self.plot_me:
            pathPrefix = self.outdir
            prefix = self.uvfile
            prefix, ext = os.path.splitext(os.path.basename(prefix))
            if pathPrefix == None:
                filename = prefix + '_dirty_%dp.png' % self.chan
                fitsfile = prefix + '_dirty_%dp.fit' % self.chan
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                filename = pathPrefix + '/' + prefix + '_dirty_%dp.png' % self.chan
                fitsfile = pathPrefix + '/' + prefix + '_dirty_%dp.fit' % self.chan

            self.muser_draw.draw_one(filename, self.title, self.fov, dirty, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,
                                     self.dec + 0.5, 16.1, axis=False, axistype=0)

        if self.writefits:
            self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')
        return filename

    def gpu_info(self):
        (free, total) = cu.mem_get_info()
        logger.debug("Global memory occupancy:%f free at %f " % (free, total))

    def dirty_realtime(self, subarray, polarization, frequency, vis_file, uv_file, ra, dec, outdir, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):
        # Load settings for each example
        self.sub_array = subarray
        self.visfile = vis_file
        self.uvfile = uv_file
        self.outdir = outdir
        self.plot_me = PLOT_ME
        self.writefits = WRITE_FITS
        self.Debug = DEBUG
        self.correct_p_angle = P_ANGLE

        self.briggs = np.float32(1e7)  # weight parameter
        self.light_speed = 299792458.  # Speed of light
        self.bandwidth = 25000000.0

        if self.sub_array == 1:
            self.antennas = 40
            self.baseline_number = 780
            self.basefreq = 400000000
        elif self.sub_array == 2:
            self.antennas = 60
            self.baseline_number = 1660
            self.basefreq = 2000000000

        if self.muser_env.file_exist(self.visfile) == True:
            self.visdata = np.fromfile(self.visfile, dtype=complex)
            self.visdata.shape = self.baseline_number, 16

        if self.muser_env.file_exist(self.uvfile) == True:
            self.uvdata = np.fromfile(self.uvfile, dtype=float)
            self.uvdata.shape = self.baseline_number, 3

        self.telescope = "MUSER"
        obs_date_str = self.visfile.split('/')[-1].split("-")[0] + self.visfile.split('/')[-1].split("-")[1].split("_")[0]
        self.obs_date = datetime.datetime.strptime(obs_date_str, "%Y%m%d%H%M%S").strftime("%Y-%m-%d 00:00:00")
        obs_datetime = datetime.datetime.strptime(obs_date_str, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")
        dirtytime = datetime.datetime.strptime(obs_date_str, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        self.dirtytime = dirtytime+'.'+ self.visfile.split('/')[-1].split('_')[1]


        self.object = "MUSER-" + str(self.sub_array)
        self.polarization = polarization
        self.ra = ra
        self.dec = dec
        self.freq = frequency

        logger.debug("Instrument: %s" % self.telescope)
        logger.debug("Obs date:   %s" % self.obs_date)
        logger.debug("Base Frequency:  %d" % self.basefreq)
        logger.debug("Frequency:  %d" % self.freq)
        logger.debug("Bandwidth:  %d" % self.bandwidth)
        logger.debug("Polarization: %d" % self.polarization)
        logger.debug("Target RA:  %f" % self.ra)
        logger.debug("Target DEC: %f" % self.dec)

        # Retrieve Antennas information
        if self.sub_array == 1:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, self.obs_date)
        else:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(2, self.obs_date)
        logger.debug("FLag Antennas: %s " % self.Flag_Ant)

        if self.freq == 400E6:
            ISIZE = 256
        elif self.freq == 800E6:
            ISIZE = 512
        elif self.freq in [1200E6, 1600E6]:
            ISIZE = 1024

        if self.correct_p_angle:
            self.imsize = np.int32(ISIZE * 2)  # number of image pixels
        else:
            self.imsize = np.int32(ISIZE)  # number of image pixels

        pngfile=[]
        self.h_uu = np.ndarray(shape=self.baseline_number, dtype=float)
        self.h_vv = np.ndarray(shape=self.baseline_number, dtype=float)

        for self.chan in range(0, 16):
            self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
            self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi
            self.cell = self.angular_resolution / 3.
            self.fov = self.cell * self.imsize
            self.number_of_wavelentgh = 1. / (self.cell / 3600. / 180. * np.pi)
            logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))
            self.title = ('DIRTY IMAGE OF MUSER \n %s POL: %c @%.4fGHz') % (self.dirtytime, 'L' if self.polarization == 0 else 'R', self.current_freq / 1e9)

            for i in range(0, self.baseline_number):
                self.h_uu[i] = self.uvdata[i][0]
                self.h_vv[i] = self.uvdata[i][1]
            self.h_uu *= self.current_freq
            self.h_vv *= self.current_freq
            self.h_rere = np.float32(self.visdata[:, self.chan].real)
            self.h_imim = np.float32(self.visdata[:, self.chan].imag)

            self.gpu_info()
            self.p_angle, b, sd = pb0r(obs_datetime)
            self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.
            pngfile.append(self.preclean())

        return pngfile
