#coding=utf-8
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

#import the Pyopencl modules
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl import clmath
from pyfft.cl import Plan

from scipy.ndimage.filters import gaussian_filter
from slope import get_peaks
from musersun import *
from muserdraw import *
from muserfits import *
from muserfilter import *
from muserfits import *
from argparse import *
from datetime import *
from muserant import *

import logging
import gausspeak as peakutils
import gaussfitter as gf


logger = logging.getLogger('muser')
GRID   = lambda x, y, W: ((x) + ((y) * W))
IGRIDX = lambda tid, W: tid % W
IGRIDY = lambda tid, W: int(tid) / int(W)

MATRIX_SIZE = 4
TILE_SIZE = 2
BLOCK_SIZE = TILE_SIZE


class MuserClean:
    
    def __init__(self):	
	self.muser_draw = MuserDraw()
        self.muser_fits = MuserFits()
        self.muser_ant  = MuserAntenna()

	self.ctx = cl.create_some_context(interactive=False)
	self.queue = cl.CommandQueue(self.ctx)
        self.add_noise_kernel = ElementwiseKernel(self.ctx,\
	"float *a, float* b, int N",\
        "b[i] = a[i]+b[i]",\
        "gpunoise")

	src_kernel = self.LoadProgram("muserkernel_cl.py")
	prg = cl.Program(self.ctx,src_kernel).build()
		
	# -------------------
        # Gridding kernels
        # -------------------
	self.sub_beam_kernel = prg.sub_beam_kernel
	self.sub_histogram_kernel = prg.sub_histogram_kernel
	self.sub_mean_average_kernel = prg.sub_mean_average_kernel
	self.sub_matrix_mul_kernel = prg.sub_matrix_mul_kernel
	self.sub_dot_mul_kernel = prg.sub_dot_mul_kernel
	self.sub_cycle_shift_kernel = prg.sub_cycle_shift_kernel
	self.sub_cuda_cyclic_shift_kernel = prg.sub_cuda_cyclic_shift_kernel
	self.sub_rotate_image_kernel = prg.sub_rotate_image_kernel
	self.find_max_kernel = prg.find_max_kernel
				
	# -------------------
        # clean kernels 
        # -------------------
	self.gridVis_wBM_kernel = prg.gridVis_wBM_kernel
        self.shiftGrid_kernel = prg.shiftGrid_kernel
        self.nrmGrid_kernel = prg.nrmGrid_kernel
        self.wgtGrid_kernel = prg.wgtGrid_kernel
        self.dblGrid_kernel = prg.dblGrid_kernel
        self.corrGrid_kernel = prg.corrGrid_kernel
        self.nrmBeam_kernel = prg.nrmBeam_kernel
        self.trimIm_kernel = prg.trimIm_kernel
        self.copyIm_kernel = prg.copyIm_kernel
        self.copyRIm_kernel = prg.copyRIm_kernel
        self.diskGrid_kernel = prg.diskGrid_kernel
        self.trim_float_image_kernel = prg.trim_float_image_kernel
	
    def LoadProgram(self, filename):
        srcfile = open(filename,'r')
        src = "".join(srcfile.readlines())
        return src

    def calc_gpu_thread(self, nx, imsize, gcount):
        self.blocksize_2D = (8, 16, 1)
        self.gridsize_2D = (\
        np.int(np.ceil(1. * nx / self.blocksize_2D[0])), np.int(np.ceil(1. * nx / self.blocksize_2D[1])))
	self.globalsize_2D=(nx, nx)
	self.localsize_2D=(32,16)
        self.blocksize_F2D = (16, 16, 1)
        self.gridsize_F2D = (\
        np.int(np.ceil(1. * imsize / self.blocksize_F2D[0])), np.int(np.ceil(1. * imsize / self.blocksize_F2D[1])))
	self.globalsize_F2D=(imsize,imsize)
	self.localsize_F2D=(16,16)
        self.blocksize_1D = (256, 1)
        self.gridsize_1D = (np.int(np.ceil(1. * gcount / self.blocksize_1D[0])), 1)
	self.globalsize_1D=(gcount,1)
	self.localsize_1D=(64,1)

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

        nx = A.shape[0]
        plan = Plan((np.int(nx), np.int(nx)),queue=self.queue)
        self.blocksize_2D = (8, 16, 1)
        self.gridsize_2D = (np.int(np.ceil(1. * nx / self.blocksize_2D[0])), np.int(np.ceil(1. * nx / self.blocksize_2D[1])))
	self.globalsize_2D=(nx,nx)

        d_af = cl_array.to_device(self.queue,A)
        d_bf = cl_array.to_device(self.queue,B)

        d_grd = cl_array.zeros(self.queue,(np.int(nx), np.int(nx)), np.complex64)
        d_a = cl_array.zeros_like(d_grd)
        d_b = cl_array.zeros_like(d_grd)
        d_am = cl_array.zeros_like(d_grd)
        d_bm = cl_array.zeros_like(d_grd)
        d_c = cl_array.zeros_like(d_grd)
        d_cm = cl_array.zeros_like(d_grd)
        d_im = cl_array.zeros(self.queue,(np.int(nx), np.int(nx)), np.float32)
	self.copyRIm_kernel(self.queue,self.globalsize_2D,None,d_af.data, d_a.data, np.int32(nx))
        self.copyRIm_kernel(self.queue,self.globalsize_2D,None,d_bf.data, d_b.data, np.int32(nx))
	plan.execute(d_a.data)
	d_am=d_a.get()
	d_am=cl_array.to_device(self.queue,d_am)
	plan.execute(d_b.data)
	d_bm=d_b.get()
	d_bm=cl_array.to_device(self.queue,d_bm)
        # self.shiftGrid_kernel(d_bm, d_b, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        #self.sub_dot_mul_kernel(d_am, d_bm, d_cm, np.int32(nx), np.int32(nx), block=self.blocksize_2D,grid=self.gridsize_2D)
	#self.sub_dot_mul_kernel(d_am, d_bm, d_cm, np.int32(nx), np.int32(nx), block=self.blocksize_2D,grid=self.gridsize_2D)
	self.sub_dot_mul_kernel(self.queue,self.globalsize_2D,None,d_am.data, d_bm.data, d_cm.data, np.int32(nx), np.int32(nx))
	self.sub_dot_mul_kernel(self.queue,self.globalsize_2D,None,d_am.data, d_bm.data, d_cm.data, np.int32(nx), np.int32(nx))
        #self.shiftGrid_kernel(d_c, d_cm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)

        #fft.fft(d_cm, d_c, plan)
        plan.execute(d_cm.data)
	d_c=d_cm.get()
	d_c=cl_array.to_device(self.queue,d_c)
        #self.shiftGrid_kernel(d_c, d_cm, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
        #self.copyIm_kernel(d_cm, d_im, np.int32(nx), block=self.blocksize_2D, grid=self.gridsize_2D)
	self.shiftGrid_kernel(self.queue,self.globalsize_2D,None,d_c.data, d_cm.data, np.int32(nx))
        self.copyIm_kernel(self.queue,self.globalsize_2D,None,d_cm.data, d_im.data, np.int32(nx))
        return d_im

    def cuda_gridvis(self , plan , x_offset, y_offset):
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
	#set plan
	#plan = Plan((np.int(nx), np.int(nx)),queue=self.queue)
        noff = np.int32((nx - self.imsize) / 2)

        ## constants
        arc2rad = np.float32(np.pi / 180. / 3600.)
        du = np.float32(1. / (arc2rad * self.cell)) / (self.imsize * 2.)
        logger.debug("1 Pixel DU  = %f" % du)
        ## grab data

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

        d_u = cl_array.to_device(self.queue,np.array(h_u, dtype='float32'))
        d_v = cl_array.to_device(self.queue,np.array(h_v, dtype='float32'))
        d_re = cl_array.to_device(self.queue,np.array(h_re, dtype='float32'))
        d_im = cl_array.to_device(self.queue,np.array(h_im, dtype='float32'))
        d_cnt = cl_array.zeros(self.queue,(np.int(nx), np.int(nx)), np.int32)
        d_grd = cl_array.zeros(self.queue,(np.int(nx), np.int(nx)), np.complex64)
	
        d_ngrd = cl_array.zeros_like(d_grd)
        d_bm = cl_array.zeros_like(d_grd)
        d_nbm = cl_array.zeros_like(d_grd)
        d_cbm = cl_array.zeros_like(d_grd)
        d_fbm = cl_array.zeros(self.queue,(np.int(nx), np.int(nx)), np.float32)
        d_fim = cl_array.zeros(self.queue,(np.int(self.imsize), np.int(self.imsize)), np.float32)
        d_dim = cl_array.zeros(self.queue,(np.int(self.imsize), np.int(self.imsize)), np.float32)
	
        d_sun_disk = cl_array.zeros_like(d_grd)
        d_fdisk = cl_array.zeros(self.queue,(np.int(self.imsize), np.int(self.imsize)), np.float32)
	
        ## define kernel parameters
        self.calc_gpu_thread(nx, self.imsize, gcount)
        # Testing WFWFWF - Used in temp
        # Generate imsize * imsize Sample FUnction
        # h_grid = np.zeros((np.int(nx / 2), np.int(nx /2)), np.int32)
        #
        # du1 = np.float32(1 / (arc2rad * self.cell * nx/2))
        #
        # u0 = nx / 2
        # for ivis in range(gcount):
        #     mu = h_u[ivis]
        #     mv = h_v[ivis]
        #     hflag = 1
        #     if (mu < 0.) == True:
        #         hflag = -1
        #         mu = -1*mu
        #         mv = -1*mv
        #     uu = mu/du1
        #     vv = mv/du1
        #     ind = vv*nx/2+uu
        #     h_grid[u0/2+uu][u0/2+vv] = 1
        #     h_grid[u0/2-uu][u0/2-vv] = 1
        #
        # #h_nbm = np.int32(d_nbm.get())
        # hdunew = pyfits.PrimaryHDU(data=h_grid)
        # hdunew.header.set('OBSERVER','MUSER')
        # hdunew.header.set('COMMENT' ,  "Here's some commentary about this FITS file.")
        # hdulistnew = pyfits.HDUList([hdunew])
        # pathPrefix = self.outdir
        # if pathPrefix[-1:] == '/':
        #     pathPrefix = pathPrefix[:-1]
        # if not os.path.exists(pathPrefix):
        #     os.makedirs(pathPrefix)
        # gridfitsname = pathPrefix + '/' + 'grid.fit'
        #
        # hdulistnew.writeto(gridfitsname,clobber=True)
        # try:
        #     vra
        # except NameError:
        #     vra = [np.percentile(h_grid, 1), np.percentile(h_grid, 99)]
        #
        # fig, axs = plt.subplots(figsize=(6.1, 6)) #1, 2, sharex=True, sharey=True, figsize=(12.2, 6));
        # plt.subplots_adjust(wspace=0)
        # #axs.imshow(dirty, vmin=vra[0], vmax=vra[1], cmap=cm.gray, origin='lower')
        # #axs.set_title('Original dirty image')
        # axs.imshow(h_grid, vmin=0, vmax=1, cmap=cm.gray, origin='lower')
        # axs.set_title('Cleaned image')
        # plt.savefig('grid.png')

        # ------------------------
        # make gridding kernels
        # ------------------------
        ## make spheroidal convolution kernel (don't mess with these!)
        width = 6.
        ngcf = 24.
        h_cgf = self.gcf(ngcf, width)
        ## make grid correction
	h_corr = self.corrfun(nx, width)
        #d_cgf = self.module.get_global('cgf')[0]
        #d_corr = gpu.to_gpu(h_corr)
        #cu.memcpy_htod(d_cgf, h_cgf)
        d_corr = cl_array.to_device(self.queue,h_corr)
	d_cgf =  cl_array.to_device(self.queue,h_cgf)
       
        # ------------------------
        # grid it up
        # ------------------------

	d_umax = cl_array.max(clmath.fabs(d_u))
        d_vmax = cl_array.max(clmath.fabs(d_v))
        umax = np.int32(np.ceil(d_umax.get()/ du))
        vmax = np.int32(np.ceil(d_vmax.get()/ du))
        ## grid ($$)
        #  This should be improvable via:
        #    - shared memory solution? I tried...
        #    - better coalesced memory access? I tried...
        #    - reorganzing and indexing UV data beforehand?
        #       (i.e. http://www.nvidia.com/docs/IO/47905/ECE757_Project_Report_Gregerson.pdf)
        #    - storing V(u,v) in texture memory?
        #self.gridVis_wBM_kernel(d_grd, d_bm, d_cbm, d_cnt, d_u, d_v, d_re, d_im, np.int32(nx), np.float32(du),
        #                       np.int32(gcount), np.int32(umax), np.int32(vmax),
        #                        np.int32(1 if self.correct_p_angle else 0),
        #                        block=self.blocksize_2D, grid=self.gridsize_2D)
	
	self.gridVis_wBM_kernel(self.queue,self.globalsize_2D,None,d_cgf.data,d_grd.data, d_bm.data, d_cbm.data,\
		                d_cnt.data, d_u.data, d_v.data, d_re.data, d_im.data, np.int32(nx), np.float32(du),\
                                np.int32(gcount), np.int32(umax), np.int32(vmax),\
                                np.int32(1 if self.correct_p_angle else 0))
	## apply weights
        #self.wgtGrid_kernel(d_bm, d_cnt, self.briggs, nx, 0,  block=self.blocksize_2D, grid=self.gridsize_2D)
	self.wgtGrid_kernel(self.queue,self.globalsize_2D,None,d_bm.data, d_cnt.data, self.briggs, nx, 0)
        hfac = np.int32(1)
        #self.dblGrid_kernel(d_bm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
        #self.dblGrid_kernel(d_cbm, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.dblGrid_kernel(self.queue,self.globalsize_2D,None,d_bm.data, nx, hfac)
        self.dblGrid_kernel(self.queue,self.globalsize_2D,None,d_cbm.data, nx, hfac)
        #self.shiftGrid_kernel(d_bm, d_nbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
        #self.shiftGrid_kernel(d_cbm, d_bm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.shiftGrid_kernel(self.queue,self.globalsize_2D,None,d_bm.data, d_nbm.data, nx)
        self.shiftGrid_kernel(self.queue,self.globalsize_2D,None,d_cbm.data, d_bm.data, nx)
        ## normalize
        #self.wgtGrid_kernel(d_grd, d_cnt, self.briggs, nx, 0, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.wgtGrid_kernel(self.queue,self.globalsize_2D,None,d_grd.data, d_cnt.data, self.briggs, nx, 0)
        ## Reflect grid about v axis
        hfac = np.int32(-1)
        #self.dblGrid_kernel(d_grd, nx, hfac, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.dblGrid_kernel(self.queue,self.globalsize_2D,None,d_grd.data, nx, hfac)
        ## Shift both
        #self.shiftGrid_kernel(d_grd, d_ngrd, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.shiftGrid_kernel(self.queue,self.globalsize_2D,None,d_grd.data, d_ngrd.data, nx)
        # Sun Model
        # Sun disk radius = 16.1164 arcmin
	#print self.cell
        radius = 16.1164 * 60 / self.cell
	'''
        self.diskGrid_kernel(d_sun_disk, np.int32(self.imsize * 2), np.int32(radius), np.int32(100),
                             block=self.blocksize_2D,
                             grid=self.gridsize_2D)'''
	self.diskGrid_kernel(self.queue,self.globalsize_2D,None,d_sun_disk.data, np.int32(self.imsize * 2), np.int32(radius), np.int32(100))
        #fft.fft(d_sun_disk, d_grd, plan)
	plan.execute(d_sun_disk.data)
	d_grd = d_sun_disk.get()
	d_grd = cl_array.to_device(self.queue,d_grd)

        # ------------------------
        # Make the beam
        # ------------------------
        ## Transform to image plane
        # Sampling function and multiply disk
        #self.sub_dot_mul_kernel(d_grd, d_bm, d_cbm, nx, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.sub_dot_mul_kernel(self.queue,self.globalsize_2D,None,d_grd.data, d_bm.data, d_cbm.data, nx, nx)
        #fft.fft(d_cbm, d_sun_disk, plan)
	plan.execute(d_cbm.data)
	d_sun_disk = d_cbm.get()
	d_sun_disk = cl_array.to_device(self.queue,d_sun_disk)
	
        #self.trimIm_kernel(d_sun_disk, d_fdisk, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
	self.trimIm_kernel(self.queue,self.globalsize_F2D,None,d_sun_disk.data, d_fdisk.data, nx, self.imsize)
        d_bmax = cl_array.max(d_fdisk)
        bmax = d_bmax.get()
        bmax1 = np.float32(1. / bmax)
        #self.nrmBeam_kernel(d_fdisk, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
	self.nrmBeam_kernel(self.queue,self.globalsize_F2D,None,d_fdisk.data, bmax1.data, self.imsize)
        #
        #fft.fft(d_nbm, d_bm, plan)
	plan.execute(d_nbm.data)
	d_bm = d_nbm.get()
	d_bm = cl_array.to_device(self.queue,d_bm)
        ## Shift
        #self.shiftGrid_kernel(d_bm, d_nbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.shiftGrid_kernel(self.queue,self.globalsize_2D,None,d_bm.data, d_nbm.data, nx)
        ## Correct for C
        #self.corrGrid_kernel(d_nbm, d_corr, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.corrGrid_kernel(self.queue,self.globalsize_2D,None,d_nbm.data, d_corr.data, nx)
        # Trim
        #self.trimIm_kernel(d_nbm, d_fim, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
	self.trimIm_kernel(self.queue,self.globalsize_F2D,None,d_nbm.data, d_fim.data, nx, self.imsize)
        #self.copyIm_kernel(d_nbm, d_fbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.copyIm_kernel(self.queue,self.globalsize_2D,None,d_nbm.data, d_fbm.data, nx)
        ## Normalize
        d_bmax = cl_array.max(d_fim)
        bmax = d_bmax.get()
        bmax1 = np.float32(1. / bmax)
        #self.nrmBeam_kernel(d_fim, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
	self.nrmBeam_kernel(self.queue,self.globalsize_F2D,None,d_fim.data, bmax1.data, self.imsize)
        d_bmax = cl_array.max(d_fbm)
        bmax = d_bmax.get()
        bmax2 = np.float32(1. / bmax)
        #self.nrmBeam_kernel(d_fbm, bmax2, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.nrmBeam_kernel(self.queue,self.globalsize_2D,None,d_fbm.data, bmax2, nx)
        ## Pull onto CPU
        # dpsf = d_fim.get()
        dpsf2 = d_fbm.get()

        # ------------------------
        # Make the map
        # ------------------------
        ## Transform to image plane
        if (x_offset <> 0 or y_offset <> 0):
	    '''
            self.sub_cuda_cyclic_shift_kernel(d_ngrd, d_cbm, np.int32(nx), np.int32(y_offset), np.int32(x_offset),
                                              block=self.blocksize_2D, grid=self.gridsize_2D)'''
	    self.sub_cuda_cyclic_shift_kernel(self.queue,self.globalsize_2D,None,d_ngrd.data, d_cbm.data, np.int32(nx),\
                                              np.int32(y_offset), np.int32(x_offset))
            # self.sub_cuda_cyclic_shift_kernel(d_ngrd, d_cbm, np.int32(nx), np.int32(200), np.int32(200),
            #                                   block=self.blocksize_2D, grid=self.gridsize_2D)
            #fft.fft(d_cbm, d_grd, plan)
	    plan.execute(d_cbm.data)
	    d_grd = d_cbm.get()
            d_grd = cl_array.to_device(self.queue,d_grd)
        else:
            #fft.fft(d_ngrd, d_grd, plan)
	    plan.execute(d_ngrd.data)
	    d_grd = d_ngrd.get()
	    d_grd = cl_array.to_device(self.queue,d_grd)
        ## Shift
        #self.shiftGrid_kernel(d_grd, d_ngrd, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.shiftGrid_kernel(self.queue,self.globalsize_2D,None,d_grd.data, d_ngrd.data, nx)
        ## Correct for C
        #self.corrGrid_kernel(d_ngrd, d_corr, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.corrGrid_kernel(self.queue,self.globalsize_2D,None,d_ngrd.data, d_corr.data, nx)
        ## Trim
        #self.trimIm_kernel(d_ngrd, d_dim, nx, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        #self.copyIm_kernel(d_ngrd, d_fbm, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.trimIm_kernel(self.queue,self.globalsize_F2D,None,d_ngrd.data, d_dim.data, nx, self.imsize)
        self.copyIm_kernel(self.queue,self.globalsize_2D,None,d_ngrd.data, d_fbm.data, nx)
        ## Normalize (Jy/beam)i
        #self.nrmGrid_kernel(d_dim, bmax1, self.imsize, block=self.blocksize_F2D, grid=self.gridsize_F2D)
        #self.nrmGrid_kernel(d_fbm, bmax2, nx, block=self.blocksize_2D, grid=self.gridsize_2D)
	self.nrmGrid_kernel(self.queue,self.globalsize_F2D,None,d_dim.data, bmax1.data, self.imsize)
        self.nrmGrid_kernel(self.queue,self.globalsize_2D,None,d_fbm.data, bmax2.data, nx)

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
        g_dpsf = np.zeros([window, window])
        g_dpsf = dpsf[w / 2 - window / 2:w / 2 + window / 2 - 1, h / 2 - window / 2:h / 2 + window / 2 - 1]
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

        # cpsf[w / 2 - window/2:w / 2 + window/2 -1, h / 2 - window/2:h / 2 + window/2 -1] = g_cpsf[0:window -1, 0:window -1]
        # cpsf=np.zeros([h,w])
        cpsf[w / 2 - window / 2:w / 2 + window / 2, h / 2 - window / 2:h / 2 + window / 2] = d_cpsf[:, :]  ##Normalize
     
        ean, gpu_dpsf, gpu_cpsf, psf = cpsf / np.max(cpsf)
        return np.float32(cpsf)

    def gpu_getmax(self, map):
        """
        Use pycuda to get the maximum absolute deviation of the residual map,
        with the correct sign
        """
        imax = cl_array.max(clmath.fabs(map)).get()
        if cl_array.max(map).get() != imax: imax *= -1
        return np.float32(imax)

    def cuda_steer(self, gpu_dirty, gpu_pmodel, gpu_clean, gpu_dpsf, gpu_cpsf, thresh=0.2, damp=1, gain=0.05,\
                    prefix='test', add_flag=1, add_back=1):
        #def cuda_hogbom(self, thresh=0.2,damp=1,gain=0.05,prefix='test', add_flag=1, add_back=1):
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
	'''
	gpu_dirty=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_pmodel=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_clean=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_dpsf=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_cpsf=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	'''
        height, width = np.shape(gpu_dirty)


        ## Grid parameters - #improvable#
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (int(width / tsize), int(height / tsize))  # The number of thread blocks     (x,y)
        globalsize=(width,height)
        ## Setup cleam image and point source model
        ## Setup GPU constants
        # gpu_max_id = cl_array.to_device(np.zeros(int32(0))
        gpu_max_id = cl_array.to_device(self.queue,np.zeros(1, dtype='int32'))
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
	    '''
            self.find_max_kernel(gpu_dirty, gpu_max_id, imax, np.int32(width), np.int32(height), gpu_pmodel,
                                 block=blocksize, grid=gridsize)'''
	    self.find_max_kernel(self.queue,globalsize,None,gpu_dirty.data, gpu_max_id.data,\
			         imax, np.int32(width), np.int32(height),gpu_pmodel.data)
            ## Step 2 - Subtract the beam (assume that it is normalized to have max 1)
            ##          This kernel simultaneously reconstructs the CLEANed image.
            #if self.Debug:
            logger.debug("Subtracting dirty beam " + str(i) + ", maxval=%0.8f" % imax + ' at x=' + str(\
                        gpu_max_id.get() % width) + ', y=' + str(gpu_max_id.get() / width))
	    '''
            self.sub_beam_kernel(gpu_dirty, gpu_dpsf, gpu_max_id, gpu_clean, gpu_cpsf, np.float32(gain * imax),
                                 np.int32(width), \
                                 np.int32(height), np.int32(add_flag), block=blocksize, grid=gridsize)'''
	    self.sub_beam_kernel(self.queue,globalsize,None,gpu_dirty.data, gpu_dpsf.data, gpu_max_id.data,\
			         gpu_clean.data, gpu_cpsf.data, np.float32(gain * imax),
                                 np.int32(width), np.int32(height), np.int32(add_flag))
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
            #mc.add_noise_kernel(gpu_dirty, gpu_clean, np.float32(width + height))
	    self.add_noise_kernel(gpu_dirty, gpu_clean, np.float32(width + height))
        return gpu_dirty, gpu_pmodel, gpu_clean, clean_error


    def cuda_hogbom(self, gpu_dirty, gpu_pmodel, gpu_clean, gpu_dpsf, gpu_cpsf, thresh=0.2, damp=1, gain=0.05,\
                    prefix='test', add_flag=1, add_back=1):
        #def cuda_hogbom(self, thresh=0.2,damp=1,gain=0.05,prefix='test', add_flag=1, add_back=1):
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
	'''
	gpu_dirty=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_pmodel=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_clean=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_dpsf=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	gpu_cpsf=cl_array.to_device(self.queue,np.random.rand(4,4).astype(np.float32))
	'''
        height, width = np.shape(gpu_dirty)
	

        ## Grid parameters - #improvable#
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (int(width / tsize), int(height / tsize))  # The number of thread blocks     (x,y)
        globalsize=(width,height)
        ## Setup cleam image and point source model
        ## Setup GPU constants
        # gpu_max_id = cl_array.to_device(np.zeros(int32(0))
        gpu_max_id = cl_array.to_device(self.queue,np.zeros(1, dtype='int32'))
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
	    '''
            self.find_max_kernel(gpu_dirty, gpu_max_id, imax, np.int32(width), np.int32(height), gpu_pmodel,
                                 block=blocksize, grid=gridsize)'''
	    self.find_max_kernel(self.queue,globalsize,None,gpu_dirty.data, gpu_max_id.data,\
			         imax, np.int32(width), np.int32(height),gpu_pmodel.data)
            ## Step 2 - Subtract the beam (assume that it is normalized to have max 1)
            ##          This kernel simultaneously reconstructs the CLEANed image.
            #if self.Debug:
            logger.debug("Subtracting dirty beam " + str(i) + ", maxval=%0.8f" % imax + ' at x=' + str(\
                        gpu_max_id.get() % width) + ', y=' + str(gpu_max_id.get() / width))
	    '''
            self.sub_beam_kernel(gpu_dirty, gpu_dpsf, gpu_max_id, gpu_clean, gpu_cpsf, np.float32(gain * imax),
                                 np.int32(width), \
                                 np.int32(height), np.int32(add_flag), block=blocksize, grid=gridsize)'''
	    self.sub_beam_kernel(self.queue,globalsize,None,gpu_dirty.data, gpu_dpsf.data, gpu_max_id.data,\
			         gpu_clean.data, gpu_cpsf.data, np.float32(gain * imax),
                                 np.int32(width), np.int32(height), np.int32(add_flag))
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
            #mc.add_noise_kernel(gpu_dirty, gpu_clean, np.float32(width + height))
	    self.add_noise_kernel(gpu_dirty, gpu_clean, np.float32(width + height))
        return gpu_dirty, gpu_pmodel, gpu_clean, clean_error

    def cuda_histogram(self, image, binsize, no=1):
        ## Calculate histogram
        dirty_map_max = cl_array.max(image).get()
        dirty_map_min = cl_array.min(image).get()
        if dirty_map_min < 0:
            dirty_map_min = -int(round(abs(dirty_map_min) + 0.5))
        else:
            dirty_map_min = int(dirty_map_min)

        if dirty_map_max < 0:
            dirty_map_max = -int(round(abs(dirty_map_max) + 0.5))
        else:
            dirty_map_max = int(round(dirty_map_max + 0.5))

        gpu_histogram = cl_array.zeros(self.queue,(binsize), np.int32)

        height, width = np.shape(image)
	
        ## Grid parameters - #improvable#
        tsize = 8
        blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
        gridsize = (self.iDivUp(height, tsize), self.iDivUp(width, tsize))  # The number of thread blocks     (x,y)
	globalsize=(tsize*self.iDivUp(height, tsize),tsize*self.iDivUp(width, tsize))
        # gridsize = (int(height/tsize), int(width/ tsize))   # The number of thread blocks     (x,y)
	'''
        self.sub_histogram_kernel(image, np.int32(self.imsize), np.int32(self.imsize), gpu_histogram,
                                  np.int32(dirty_map_max), np.int32(dirty_map_min), np.int32(binsize), block=blocksize,
                                  grid=gridsize)'''
	self.sub_histogram_kernel(self.queue,image.shape,None,image.data, np.int32(self.imsize), np.int32(self.imsize), gpu_histogram.data,\
                                  np.int32(dirty_map_max), np.int32(dirty_map_min), np.int32(binsize))

	
        gpu_smooth_histogram = cl_array.zeros(self.queue,(binsize), np.int32)
        gpu_smooth_histogram2 = cl_array.zeros(self.queue,(binsize), np.int32)
        # Temporary Testing
        tsize = 16
        blocksize = (int(tsize), int(tsize), 1)             # The number of threads per block (x,y,z)
        gridsize = (self.iDivUp(1, tsize), self.iDivUp(binsize, tsize))   # The number of thread blocks     (x,y)
	globalsize=(tsize*self.iDivUp(1, tsize),tsize*self.iDivUp(binsize, tsize))

        width = binsize
        radius = 32
        height = 1
	'''
        self.sub_mean_average_kernel(gpu_histogram, gpu_smooth_histogram, np.int32(height), np.int32(width),\
                                     np.int32(radius), block=blocksize, grid=gridsize)'''
	self.sub_mean_average_kernel(self.queue, globalsize, None,gpu_histogram.data,\
                                     gpu_smooth_histogram.data, np.int32(height), np.int32(width), np.int32(radius))
        width = binsize
        radius = 32
        height = 1
        blocksize = (int(tsize), int(tsize), 1)             # The number of threads per block (x,y,z)
        gridsize = (self.iDivUp(1, tsize), self.iDivUp(width, tsize))   # The number of thread blocks     (x,y)
	globalsize=(tsize*self.iDivUp(1, tsize),tsize*self.iDivUp(width, tsize))
	'''
        self.sub_mean_average_kernel(gpu_smooth_histogram, gpu_smooth_histogram2, np.int32(height), np.int32(width),\
                                     np.int32(radius), block=blocksize, grid=gridsize)'''
	self.sub_mean_average_kernel(self.queue, globalsize, None,gpu_smooth_histogram.data,\
                                     gpu_smooth_histogram2.data, np.int32(height), np.int32(width), np.int32(radius))

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

        if self.Debug:
            print gmm
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

        sky_peak = (dirty_map_max - dirty_map_min) * sky_peak / 2000.
        disk_peak = (dirty_map_max - dirty_map_min) * disk_peak / 2000.
        logger.debug("Peak Value: sky_peak:%f  disk_peak:%f" % (sky_peak, disk_peak))
        return sky_peak, disk_peak

    def write_fits(self, data, fitsfile, type):
        # create_fits(self, object, obs_date, obs_time,data, imagetype):
        self.muser_fits.create_fits(data, self.object, self.muser_date.strftime("%Y-%m-%d"),self.muser_date.strftime("%H:%M:%S.%f"), type)
        self.muser_fits.append_common_header(self.current_freq, self.polarization, self.ra, self.dec, self.p_angle)
        self.muser_fits.write_fits(self.outdir, fitsfile)

    def preclean(self):
        ## Create the PSF & dirty image
        #   dpsf - PSF, gpu_im ( dirty image)
        #   dpsf is computed by CPU, gpu_im is in the GPU
        # nx - 2 imsize, it means 2048 when imsize=1024
        nx = np.int32(2 * self.imsize)

        # create fft plan nx*nx
        self.plan = Plan((np.int(nx), np.int(nx)),queue = self.queue)
        d_dirty = cl_array.zeros(self.queue,(np.int(self.imsize), np.int(self.imsize)), np.float32)
        dpsf, gpu_im, gpu_disk = self.cuda_gridvis(self.plan, 0, 0)
        # , misc.minabs(gpu_im)

        gpu_dpsf = cl_array.to_device(self.queue,dpsf)
        h_disk = gpu_disk.get()
        # gpu_dpsf2 = cl_array.to_device(dpsf2)

        ## Clean the PSF
        if self.imsize >= 1024:
            cpsf = self.get_clean_beam(dpsf,50)  # self.imsize / 32.)
        elif self.imsize >= 512:
            cpsf = self.get_clean_beam(dpsf, np.int32(self.imsize / 24.))
        elif self.imsize >= 256:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 16.)
	
        gpu_cpsf = cl_array.to_device(self.queue,cpsf)

        if self.plot_me and self.Debug:

            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True);
            plt.subplots_adjust(wspace=0)
            axs[0].imshow(cpsf, vmin=np.percentile(cpsf, 1), vmax=np.percentile(cpsf, 100), cmap=cm.jet)
            # fig.colorbar(cpsf)
            im = axs[1].imshow(dpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 100), cmap=cm.jet)
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
                fitsfile = prefix + '_disk_%dp.fit' % self.chan
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                filename = pathPrefix + '/' + prefix + '_disk_%dp.png' % self.chan
                fitsfile = pathPrefix + '/' + prefix + '_disk_%dp.fit' % self.chan
            # TODO , FITS
            self.muser_draw.draw_one(filename, "DISK MAP", h_disk, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,\
                                     self.dec + 0.5, 16.1, axistype=0)

        # Histogram
	sky_peak, disk_peak = self.cuda_histogram(gpu_im, 2000)
        dirty = gpu_im.get()

        if self.plot_me and self.Debug:

            # dirty = np.roll(np.fliplr(gpu_im.get()), 1, axis=1)
            # dirty = d_dirty.get()

            pathPrefix = self.outdir
            if pathPrefix == None:
                filename = prefix + '_dirty_%dp.png' % self.chan
                fitsfile = prefix + '_dirty_%dp.fit' % self.chan
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                filename = pathPrefix + '/' + prefix + '_dirty_%dp.png' % self.chan
                fitsfile = pathPrefix + '/' + prefix + '_dirty_%dp.fit' % self.chan

            if self.Debug:
                logger.debug("Plotting dirty image")
                # TODO , FITS
                self.muser_draw.draw_one(filename, "DIRTY MAP", dirty, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,
                                         self.dec + 0.5, 16.1, axistype=0)

        if self.writefits and self.Debug:
            self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        ## Run CLEAN
        # Clean till >=Disk
        height, width = np.shape(gpu_im)
        gpu_pmodel = cl_array.zeros(self.queue,(height, width), dtype=np.float32)
        gpu_clean = cl_array.zeros(self.queue,(height, width), dtype=np.float32)
        gpu_dirty_shift = cl_array.zeros(self.queue,(height, width), dtype=np.float32)
        gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean, gpu_dpsf,\
                                                                          gpu_cpsf,\
                                                                          thresh=disk_peak, gain=0.1, add_flag=1,\
                                                                          add_back=1)

        dirty = gpu_dirty.get()
        if self.Debug:
            prefix = self.infile
            prefix, ext = os.path.splitext(os.path.basename(prefix))

            pathPrefix = self.outdir
            if pathPrefix == None:
                filename = prefix + '_dirty2_%dp.png' % self.chan
                fitsfile = prefix + '_dirty2_%dp.fit' % self.chan
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                filename = pathPrefix + '/' + prefix + '_dirty2_%dp.png' % self.chan
                fitsfile = pathPrefix + '/' + prefix + '_dirty2_%dp.fit' % self.chan

            if self.plot_me:
                logger.debug("Plotting dirty image")
                self.muser_draw.draw_one(filename, "DIRTY MAP", dirty, self.ra - 0.5, self.ra + 0.5, self.dec - 0.5,\
                                         self.dec + 0.5, 16.1, axistype=0)

                # TODO , FITS

            if self.writefits:
                self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        # h_disk_im  = self.sub_sun_disk_offset(h_disk, dirty)
        gpu_disk_im = self.sub_sun_disk_offset(h_disk, dirty)
        h_disk_im = gpu_disk_im.get()
        logger.debug("X,Y: %d %d " % (np.argmax(np.max(h_disk_im, axis=0)), np.argmax(np.max(h_disk_im, axis=1))))

        #sky_peak, disk_peak = self.cuda_histogram(gpu_im, 2000)
        #
        # print "Peak:", sky_peak, disk_peak

        return -np.argmax(np.max(h_disk_im, axis=0)) + self.imsize / 2, -self.imsize / 2 + np.argmax(\
               np.max(h_disk_im, axis=1)), sky_peak, disk_peak
        
            
        # self.blocksize_F2D = (16, 16, 1)
        # self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[0])),\
        #                      np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[1])))
        #
        # self.sub_cycle_shift_kernel(gpu_dirty, gpu_dirty_shift, np.int32(self.imsize),
        #                                np.int32(-self.imsize //2 + np.argmax(np.max(h_disk_im, axis=0))),
        #                                np.int32(-self.imsize //2 + np.argmax(np.max(h_disk_im, axis=1))), block=self.blocksize_F2D,
        #                                grid=self.gridsize_F2D)
        # #print "Disk Offset:", self.centroid(dirty)
        # dirty2 = gpu_dirty_shift.get()
        #
        # if self.plot_me:
        #     prefix = self.infile
        #     prefix, ext = os.path.splitext(os.path.basename(prefix))
        #     try:
        #         vra
        #     except NameError:
        #         vra = [np.percentile(dirty2, 1), np.percentile(dirty2, 100)]
        #
        #     print "Plotting dirty image"
        #     fig, axs = plt.subplots()  # 1, 2, sharex=True, sharey=True, figsize=(12.2, 6));
        #     plt.subplots_adjust(wspace=0)
        #     im = axs.imshow(dirty2, vmin=vra[0], vmax=vra[1], cmap=cm.jet, origin='lower')
        #     axs.set_title(self.obs_date + ' dirty image')
        #     # axs[1].imshow(np.roll(np.fliplr(gpu_dirty.get()), 1, axis=1), vmin=vra[0], vmax=vra[1], cmap=cm.gray,
        #     #          origin='lower')
        #     # axs[1].set_title('Dirty image cleaned of sources')
        #     fig.colorbar(im)
        #     pathPrefix = self.outdir
        #     if pathPrefix == None:
        #         plt.savefig(prefix + '_dirty3_%dp.png' % self.chan)
        #         fitsfile = prefix + '_dirty3_%dp.fit' % self.chan
        #     else:
        #         if pathPrefix[-1:] == '/':
        #             pathPrefix = pathPrefix[:-1]
        #         plt.savefig(pathPrefix + '/' + prefix + '_dirty3_%dp.png' % self.chan)
        #         fitsfile = pathPrefix + '/' + prefix + '_dirty3_%dp.fit' % self.chan
        #     plt.close()
        #     # TODO , FITS
        #
        #     self.write_fits(dirty, fitsfile, self.obs_date)

    def clean(self, x_offset, y_offset, sky_peak, disk_peak):
        ## Create the PSF & dirty image
        #   dpsf - PSF, gpu_im ( dirty image)
        #   dpsf is computed by CPU, gpu_im is in the GPU
        # nx - 2 imsize, it means 2048 when imsize=1024
        nx = np.int32(2 * self.imsize)

        # create fft plan nx*nx
        #self.plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)
	self.plan = Plan((np.int(nx), np.int(nx)),queue=self.queue)

        d_dirty = cl_array.zeros(self.queue,(np.int(self.imsize), np.int(self.imsize)), np.float32)
        d_final = cl_array.zeros(self.queue,(np.int(self.imsize / 2), np.int(self.imsize / 2)), np.float32)
        logger.debug("OFFSET: %f %f " % (x_offset, y_offset))
        dpsf, gpu_im, gpu_disk = self.cuda_gridvis(self.plan, x_offset, y_offset)
	#print dpsf
	#print gpu_im
	#print gpu_disk
        #misc.minabs(gpu_im)

        gpu_dpsf = cl_array.to_device(self.queue,dpsf)
        h_disk = gpu_disk.get()
        # gpu_dpsf2 = cl_array.to_device(dpsf2)


        ## Clean the PSF
	
        if self.imsize >= 1024:
            cpsf = self.get_clean_beam(dpsf,np.int32( self.imsize / 32.))
        elif self.imsize >= 512:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 24.)
        elif self.imsize >= 256:
            cpsf = self.get_clean_beam(dpsf, self.imsize / 16.)
	
	#set cpsf
	#W,H = dpsf.shape
	#cpsf =np.random.rand(W,H).astype(np.float32)

        gpu_cpsf = cl_array.to_device(self.queue,cpsf)

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
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[0])),\
                                 np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[1])))
	    self.globalsize_F2D=(self.imsize,self.imsize)
	    '''
            self.sub_rotate_image_kernel(gpu_im, d_dirty, np.int32(self.imsize), np.int32(self.imsize),
                                         np.float32(self.p_angle), np.float32(1.), block=self.blocksize_F2D,
                                         grid=self.gridsize_F2D)'''
	    self.sub_rotate_image_kernel(self.queue, self.globalsize_F2D, None, gpu_im.data, d_dirty.data,\
                                         np.int32(self.imsize), np.int32(self.imsize), np.float32(self.p_angle), np.float32(1.))

            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[0])),\
                                 np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[1])))
	    '''
            self.trim_float_image_kernel(d_dirty, d_final, np.int32(self.imsize), np.int32(self.imsize / 2),
                                         block=self.blocksize_F2D, grid=self.gridsize_F2D)'''
	    self.trim_float_image_kernel(self.queue, self.globalsize_F2D, None,d_dirty.data, d_final.data,\
                                         np.int32(self.imsize), np.int32(self.imsize / 2))
            dirty = d_final.get()
        else:
            #
            # self.blocksize_F2D = (16, 16, 1)
            # self.gridsize_F2D = ( np.int(np.ceil(1. * self.imsize  / self.blocksize_F2D[0])),\
            #                       np.int(np.ceil(1. * self.imsize/ self.blocksize_F2D[1])))
            # self.trim_float_image_kernel(gpu_im, d_final, np.int32(2*self.imsize), np.int32(self.imsize),
            #                              block=self.blocksize_F2D, grid=self.gridsize_F2D)
            dirty = gpu_im.get()

        # dirty = d_dirty.get()
        # dirty = gpu_im.get()
        # Histogram
        # sky_peak, disk_peak = self.cuda_histogram(gpu_im, 2000)
        # print "Peak:", sky_peak, disk_peak

        prefix = self.infile
        prefix, ext = os.path.splitext(os.path.basename(prefix))
        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = prefix + '_dirty_%d.png' % self.chan
            fitsfile = prefix + '_dirty_%d.fit' % self.chan
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + prefix + '_dirty_%d.png' % self.chan
            fitsfile = pathPrefix + '/' + prefix + '_dirty_%d.fit' % self.chan

        if self.plot_me:
            logger.debug("Plotting final dirty image")
            title = ('DIRTY IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' \
                     if self.polarization == -2 else 'R', self.current_freq / 1e9)
            if self.correct_p_angle:
                self.muser_draw.draw_one(filename, title, dirty, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.dec - (self.cell * self.imsize / 2) / 3600,\
                                     self.dec + (self.cell * self.imsize / 2) / 3600,\
                                     (16.1125 * 60 / self.cell)*2 / self.imsize, axis=True, axistype=1)
            else:
                self.muser_draw.draw_one(filename, title, dirty, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.dec - (self.cell * self.imsize / 2) / 3600,\
                                     self.dec + (self.cell * self.imsize / 2) / 3600,\
                                     (16.1125 * 60 / self.cell)/ self.imsize, axis=True, axistype=1)

            # self.muser_draw.draw_one(filename,title, dirty,self.ra - 0.5, self.ra+0.5,self.dec-0.5, 
            #                          self.dec+0.5, 16.1, axis=False, axistype = 1)

        if self.writefits:
            self.write_fits(dirty, fitsfile, 'DIRTY_IMAGE')

        ## Run CLEAN
        # Clean till >=Disk
        height, width = np.shape(gpu_im)
        gpu_pmodel = cl_array.zeros(self.queue,(height, width), dtype=np.float32)
        gpu_clean = cl_array.zeros(self.queue,(height, width), dtype=np.float32)

        gpu_dirty, gpu_pmodel, gpu_clean, clean_result = self.cuda_hogbom(gpu_im, gpu_pmodel, gpu_clean, gpu_dpsf,\
                                                                          gpu_cpsf,\
                                                                          thresh=sky_peak, gain=0.1, add_flag=1,\
                                                                          add_back=1)

        if self.correct_p_angle:
            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[0])),\
                                 np.int(np.ceil(1. * self.imsize / self.blocksize_F2D[1])))
            '''
            self.sub_rotate_image_kernel(gpu_clean, d_dirty, np.int32(self.imsize), np.int32(self.imsize),
                                         np.float32(self.p_angle), np.float32(1.), block=self.blocksize_F2D,
                                         grid=self.gridsize_F2D)'''
	    self.sub_rotate_image_kernel(self.queue,self.globalsize_F2D,None, gpu_clean.data, d_dirty.data,\
                                         np.int32(self.imsize), np.int32(self.imsize), np.float32(self.p_angle), np.float32(1.))
            noff = np.int32((nx - self.imsize) / 2)

            self.blocksize_F2D = (16, 16, 1)
            self.gridsize_F2D = (np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[0])),\
                                 np.int(np.ceil(1. * self.imsize / 2 / self.blocksize_F2D[1])))
	    '''
            self.trim_float_image_kernel(d_dirty, d_final, np.int32(self.imsize), np.int32(self.imsize / 2),
                                         block=self.blocksize_F2D, grid=self.gridsize_F2D)'''
	    self.trim_float_image_kernel(self.queue,self.globalsize_F2D,None, d_dirty.data, d_final.data,\
                                         np.int32(self.imsize), np.int32(self.imsize / 2))
            clean = d_final.get()
        else:
            clean = gpu_clean.get()

        pathPrefix = self.outdir
        if pathPrefix == None:
            filename = prefix + '_clean_%d.png' % self.chan
            fitsfile = prefix + '_clean_%d.fit' % self.chan
        else:
            if pathPrefix[-1:] == '/':
                pathPrefix = pathPrefix[:-1]
            filename = pathPrefix + '/' + prefix + '_clean_%d.png' % self.chan
            fitsfile = pathPrefix + '/' + prefix + '_clean_%d.fit' % self.chan

        if self.plot_me:
            logger.debug("Plotting final clean image")

            title = ('CLEAN IMAGE OF MUSER \n TIME: %s POL: %c @%.4fGHz') % (self.obs_date, 'L' \
                     if self.polarization == -2 else 'R', self.current_freq / 1e9)
            if self.correct_p_angle:
                self.muser_draw.draw_one(filename, title, clean, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.dec - (self.cell * self.imsize / 2) / 3600,\
                                     self.dec + (self.cell * self.imsize / 2) / 3600,\
                                     (16.1125 * 60 / self.cell) *2 / self.imsize, axis=True, axistype=1)
            else:
                self.muser_draw.draw_one(filename, title, clean, self.ra - (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.ra + (self.cell * self.imsize / 2) / 3600 / 15,\
                                     self.dec - (self.cell * self.imsize / 2) / 3600,\
                                     self.dec + (self.cell * self.imsize / 2) / 3600,\
                                     (16.1125 * 60 / self.cell)  / self.imsize, axis=True, axistype=1)


        if self.writefits:
            self.write_fits(clean, fitsfile, 'CLEANED_IMAGE')

    def gpu_info(self):
        (free, total) = cu.mem_get_info()
        logger.debug("Global memory occupancy:%f free at %f " % (free, total))

        # for devicenum in range(cu.Device.count()):
        #     device=cu.Device(devicenum)
        #     attrs=device.get_attributes()
        #
        #     #Beyond this point is just pretty printing
        #     print("\n===Attributes for device %d"%devicenum)
        #     for (key,value) in attrs.iteritems():
        #         print("%s:%s"%(str(key),str(value)))

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

    def clean_with_fits(self, infile, outdir, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG):
        # Load settings for each example
	logger = logging.getLogger('muser')

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

        self.fitsfile = pyfits.open(self.infile,ignore_missing_end=True)
        self.telescope = self.fitsfile[0].header['INSTRUME'].strip()
        if self.telescope != 'MUSER':
            logger.error("Current program can only support MUSER.")
            return
        self.channel = self.fitsfile[0].data.data.shape[3]
        self.baseline_number = self.fitsfile[0].header['GCOUNT']
        self.obs_date = self.fitsfile[0].header['DATE-OBS']
        self.muser_date = datetime.datetime.strptime(self.obs_date[:-3], "%Y-%m-%dT%H:%M:%S.%f")

        if self.infile.find('.fitsidi') != -1:
            self.freq = np.float32(self.fitsfile[7].header['CRVAL3'])  # 299792458vvvv
        elif self.infile.find('.uvfits') != -1:
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
            logger.debug("Frequency:  %d" % self.freq)
            logger.debug("Bandwidth:  %d" % self.bandwidth)
            logger.debug("Channels:   %d" % self.channel)
            logger.debug("Polarization: %d" % self.polarization)
            logger.debug("Target RA:  %f" % self.ra)
            logger.debug("Target DEC: %f" % self.dec)
  
        # Retrieve Antennas information
	
        if self.object.upper().strip()=='MUSER-1':
            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, self.muser_date.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(2, self.muser_date.strftime("%Y-%m-%d %H:%M:%S"))
        logger.debug("FLag Antennas: %s " % self.Flag_Ant)
	
        #self.Flag_Ant = []
        # determin the file type (uvfits or fitsidi)
        self.h_uu = np.ndarray(shape=(self.baseline_number), dtype='float64')
        self.h_vv = np.ndarray(shape=(self.baseline_number), dtype='float64')
        self.h_rere = np.ndarray(shape=(self.baseline_number), dtype='float32')
        self.h_imim = np.ndarray(shape=(self.baseline_number), dtype='float32')

        for self.chan in range(4, 5):

            self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
            self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

            if self.infile.find('.fitsidi') != -1:

                # good  = np.where(f[0].data.data[:,0,0,0,0,0,0] != 0)
                # h_u   = np.float32(freq*f[0].data.par('uu')[good])
                # h_v   = np.float32(freq*f[0].data.par('vv')[good])

                self.h_uu = np.float64((self.freq + self.chan * 25000000) * self.fitsfile[0].data[:].UU)
                self.h_vv = np.float64((self.freq + self.chan * 25000000) * self.fitsfile[0].data[:].VV)

                for bl in range(0, self.baseline_number):
                    # gcount += np.int32(np.size(h_u[bl]))
                    ## assume data is unpolarized
                    # h_re   = np.float32(0.5*(f[0].data.data[good,0,0,0,0,0,0]+f[0].data.data[good,0,0,0,0,1,0]))
                    # h_im   = np.float32(0.5*(f[0].data.data[good,0,0,0,0,0,1]+f[0].data.data[good,0,0,0,0,1,1]))
                    self.h_rere[bl] = np.float32(self.fitsfile[0].data[:].data[bl][0][0][self.chan][0][0])
                    self.h_imim[bl] = np.float32(self.fitsfile[0].data[:].data[bl][0][0][self.chan][0][1])
                    ## make GPU arrays
                    self.h_uu = np.float32(self.h_uu.ravel())
                    self.h_vv = np.float32(self.h_vv.ravel())
                    gcount = np.int32(np.size(self.h_uu))
                    # gcount = len(gcount.ravel())
                    self.h_rere = np.float32(self.h_rere.ravel())
                    self.h_imim = np.float32(self.h_imim.ravel())

                    # print len(h_re),len(h_im)
            elif self.infile.find('.uvfits') != -1:
                good = np.where(self.fitsfile[0].data.data[:, 0, 0, self.chan, 0, 0] != 0)
                # the unit of uu and  vv is seconds
                self.h_uu = np.float64(self.fitsfile[0].data.par('uu')[good])  # * self.current_freq)
                self.h_vv = np.float64(self.fitsfile[0].data.par('vv')[good])  # * self.current_freq)
                gcount = np.int32(np.size(self.h_uu))
                # print self.h_uu
                ## assume data is unpolarized
                self.h_uu *= self.current_freq
                self.h_vv *= self.current_freq

                self.h_rere = np.float32(self.fitsfile[0].data.data[good, 0, 0, self.chan, 0, 0])
                self.h_imim = np.float32(self.fitsfile[0].data.data[good, 0, 0, self.chan, 0, 1])

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

            logger.debug('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq,self.imsize,\
                          self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

            if self.correct_p_angle:
                self.imsize = np.int32(ISIZE * 2)  # number of image pixels
            else:
                self.imsize = np.int32(ISIZE)  # number of image pixels
         
            #self.gpu_info()
            self.p_angle, b, sd = pb0r(self.obs_date[:-9])
            self.p_angle = np.float32(-self.p_angle) * 3.1415926535 / 180.
            
            x_offset, y_offset, sky, disk = self.preclean()
            if x_offset % 2 == 1:
                x_offset -= 1
            if y_offset % 2 == 1:
                y_offset -= 1
            self.clean(x_offset, y_offset, sky, disk)


#test
#muserclean = MuserClean()
#muserclean.clean_with_fits( '/home/ckun/data/20151101-120849_354161240.uvfits', '/home/ckun/work/',  True ,  False  ,  False  , 0)


