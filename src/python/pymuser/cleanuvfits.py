import numpy as np
import time, pdb, sys, pyfits
import matplotlib
import math
import os
import sys

import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage

import logging
from musersundisk import *

from muserclean import *
from muserant import *

logger = logging.getLogger('muser')

######################################
# Integration Clean from UVFTIS File #
######################################

class Muser_CleanUVFITS(MuserClean):
    # MUSER Dirty and Clean images from integrated UVFITS file

    def __init__(self):
        MuserClean.__init__(self)
        self.muser_draw = MuserDraw()
        self.muser_ant = MuserAntenna()

    def clean_read_data(self):

        self.baseline_base = self.antennas*(self.antennas-1)/2
        print "record_num:", self.baseline_number
        repeat_num = 1

        if self.is_loop_mode == True:
            self.h_uu = np.ndarray(shape=(self.baseline_number / 2), dtype='float64')
            self.h_vv = np.ndarray(shape=(self.baseline_number / 2), dtype='float64')
            self.h_rere = np.ndarray(shape=(self.baseline_number / 2), dtype='float32')
            self.h_imim = np.ndarray(shape=(self.baseline_number / 2), dtype='float32')
            data_start = self.baseline_base * (self.band * repeat_num - 1)
            data_end = self.baseline_base * self.band * repeat_num

            self.freqsel = np.float32(self.fitsfile[0].data[data_start][0])
            self.freq = self.basefreq + np.float32(self.fitsfile[1].data["IF FREQ"][int(self.freqsel) - 1])

            if self.object.upper().strip() == 'MUSER-1':
                self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
            else:
                self.current_freq = self.freq + (15 - self.chan) * self.bandwidth + self.bandwidth // 2

            self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi

            while data_end <= self.baseline_number:
                self.h_uu[data_start: data_end] = np.float64(self.fitsfile[0].data[data_start: data_end].par('uu'))
                self.h_vv[data_start: data_end] = np.float64(self.fitsfile[0].data[data_start: data_end].par('vv'))
                self.h_uu[data_start: data_end] *= self.current_freq
                self.h_vv[data_start: data_end] *= self.current_freq

                self.h_rere[data_start: data_end] = np.float32(
                    self.fitsfile[0].data[data_start: data_end].data[:, 0, 0, self.chan, self.polarization, 0])
                self.h_imim[data_start: data_end] = np.float32(
                    self.fitsfile[0].data[data_start: data_end].data[:, 0, 0, self.chan, self.polarization, 1])

                repeat_num += 1
                data_start = self.baseline_base * (self.band * repeat_num - 1)
                data_end = self.baseline_base * self.band * repeat_num

        else:
            self.h_uu = np.ndarray(shape=(self.baseline_number), dtype='float64')
            self.h_vv = np.ndarray(shape=(self.baseline_number), dtype='float64')
            self.h_rere = np.ndarray(shape=(self.baseline_number), dtype='float32')
            self.h_imim = np.ndarray(shape=(self.baseline_number), dtype='float32')
            data_start = self.baseline_base * (repeat_num - 1)
            data_end = self.baseline_base * repeat_num
            self.freqsel = np.float32(self.fitsfile[0].data[data_start][7])

            self.freq = self.basefreq + np.float32(self.fitsfile[1].data["IF FREQ"][int(self.freqsel) - 1])

            if self.object.upper().strip() == 'MUSER-1':
                self.current_freq = self.freq + self.chan * self.bandwidth + self.bandwidth // 2
            else:
                self.current_freq = self.freq + (15 - self.chan) * self.bandwidth + self.bandwidth // 2

            sun_disk_radius = MuserSunDisk()
            self.sun_disk_light = sun_disk_radius.getdata((self.current_freq - self.bandwidth // 2) / 1E6)

            print 'Current Freq at cleanuvfits:', self.current_freq
            self.angular_resolution = self.light_speed / self.current_freq / 3000 * 180. * 3600 / np.pi
            self.set_size()
            while data_end <= self.baseline_number:

                self.h_uu[data_start: data_end] = np.float64(self.fitsfile[0].data[data_start: data_end].par('uu'))
                self.h_vv[data_start: data_end] = np.float64(self.fitsfile[0].data[data_start: data_end].par('vv'))
                self.h_uu[data_start: data_end] *= self.current_freq
                self.h_vv[data_start: data_end] *= self.current_freq

                self.h_rere[data_start: data_end] = np.float32(self.fitsfile[0].data[data_start: data_end].data[:, 0, 0, self.chan, self.polarization, 0])
                self.h_imim[data_start: data_end] = np.float32(self.fitsfile[0].data[data_start: data_end].data[:, 0, 0, self.chan, self.polarization, 1])

                repeat_num += 1
                data_start = self.baseline_base * (repeat_num - 1)
                data_end = self.baseline_base * repeat_num

        sun_disk_radius = MuserSunDisk()
        self.sun_disk_light= sun_disk_radius.getdata ((self.current_freq -  self.bandwidth // 2) / 1E6)

        if self.correct_p_angle:
            self.imsize *= 1.5

    def flag(self, x_offset, y_offset):

        gcount = np.int32(np.size(self.h_uu))
        print "GOUNT 1:", gcount, np.size(self.h_rere)

        blen = 0
        bl_order = np.ndarray(shape=(self.baseline_number, 2), dtype=int)
        good = []

        inter_num = 1
        while inter_num <= self.baseline_number/self.baseline_base:
            for border1 in range(0, self.antennas - 1):
                for border2 in range(border1 + 1, self.antennas):
                    bl_order[blen][0] = border1
                    bl_order[blen][1] = border2
                    blen = blen + 1
            inter_num += 1

        self.h_u = []
        self.h_v = []
        self.h_re = []
        self.h_im = []
        for blen in range(0, self.baseline_number):
            if (bl_order[blen][0] not in self.Flag_Ant) and (bl_order[blen][1] not in self.Flag_Ant):
                good.append(blen)
                self.h_u.append(self.h_uu[blen])
                self.h_v.append(self.h_vv[blen])
                self.h_re.append(self.h_rere[blen])
                self.h_im.append(self.h_imim[blen])

        gcount = np.int32(np.size(self.h_u))
        print "After Flagged:", gcount, np.size(self.h_re)


    def cleanuvfits(self, infile, is_loop_mode, mode, weighting, band, channel, polarization, movera, movedec, fits, P_ANGLE, DEBUG, outdir):

        self.is_loop_mode = is_loop_mode
        self.band = band
        self.polarization = polarization
        self.inp_channel = channel
        AUTO_MOVE = False
        PLOT_ME = True


        if mode == 'hybrid':
            self.hybrid_clean_with_fits(infile, outdir, channel, weighting, mode, AUTO_MOVE, movera, movedec, PLOT_ME, fits, P_ANGLE, DEBUG)
        else:
            self.clean_with_fits(infile, outdir, channel, weighting, mode, AUTO_MOVE, movera, movedec, PLOT_ME, fits, P_ANGLE, DEBUG)















