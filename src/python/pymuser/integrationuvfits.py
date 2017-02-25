import numpy as np
import time, pdb, sys, pyfits
import matplotlib
import math
import os
import sys
import logging

from muserant import *
from muserfits import *
from muserfile import *

logger = logging.getLogger('muser')



class Muser_IntegrationUVFITS():
    ######################
    # CUDA kernels


    def __init__(self,):
        self.muser_ant = MuserAntenna()
        self.muser_fits = MuserFits()

    def iDivUp(self, a, b):
        # Round a / b to nearest higher integer value
        a = int(a)
        b = int(b)
        return int(a / b + 1) if (a % b != 0) else int(a / b)


    #####################################################################################
    # Read visibility data from Generated uvw and visibility file, and clean            #
    #####################################################################################
    def integrationuvfits(self, sub_ARRAY, is_loop_mode, start_time, end_time, TASK_TYPE, time_average, time_interval, DEBUG):
        # Read visibility data from rawdata file and clean

        self.subarray = sub_ARRAY
        self.is_loop_mode = is_loop_mode
        self.task_type = TASK_TYPE
        self.time_average = time_average
        self.time_interval = time_interval
        self.Debug = DEBUG

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

        if self.start_frame_time.get_date_time() > self.end_frame_time.get_date_time() :
            logger.info("Please give the reasonable start and end time.")
            return

        if self.task_type == "interval" or self.task_type == "mixture":
            if self.time_interval == None:
                logger.info("Time interval is required in 'intervel' or 'mixture' mode.")
                return
            self.integrate_frame_num = int((end_time - start_time).seconds/self.time_interval) + 1
        else:
            if self.time_average == None:
                logger.info("Average time is required in 'average' mode.")
                return
            self.integrate_frame_num = 1

        if self.Debug:
            logger.info("TASK: %s   TIME SPAN: %s ~ %s" % (self.task_type, start_time, end_time))


        uvfits.set_data_date_time(self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day,
                                  self.start_frame_time.hour, self.start_frame_time.minute,
                                  self.start_frame_time.second, self.start_frame_time.millisecond,
                                  self.start_frame_time.microsecond,
                                  self.start_frame_time.nanosecond)
        obs_day = ('%4d-%02d-%02d') % (self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day)

        print obs_day, uvfits.start_date_time.get_date_time(), uvfits.first_date_time.get_date_time()
        if self.subarray == 1:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(1, obs_day + " 00:00:00")
            self.baseline_base = 780
            self.band_number = 4
        else:
            self.Flag_Ant = self.muser_ant.get_flag_antenna(2, obs_day + " 00:00:00")
            self.baseline_base = 1770
            self.band_number = 33
        # print "FLag Antennas: %s " % self.Flag_Ant

        if self.is_loop_mode == True:
            uvfits.pyuvfits.set_visibility_data((self.integrate_frame_num+1)/2)
            UU = np.zeros(shape=(self.band_number, 2, self.baseline_base), dtype='float64')
            VV = np.zeros(shape=(self.band_number, 2, self.baseline_base), dtype='float64')
            REAL = np.zeros(shape=(self.band_number, 2, self.baseline_base, 16), dtype='float64')
            IMAG = np.zeros(shape=(self.band_number, 2, self.baseline_base, 16), dtype='float64')

        else:
            uvfits.pyuvfits.set_visibility_data(self.integrate_frame_num)

        if uvfits.search_first_frame() == False:
            logger.error("Cannot find observational data.")

        self.obs_date = uvfits.current_frame_time.get_fits_date_time()
        self.muser_date = datetime.datetime.strptime(self.obs_date[:-3], "%Y-%m-%dT%H:%M:%S.%f")
        self.ra = uvfits.ra_sum
        self.dec = uvfits.dec_sum
        self.freq = uvfits.frequency
        self.polarization = uvfits.polarization
        uvfits.set_priority(0)
        uvfits.load_calibration_data()

        niter = 0
        if self.is_loop_mode == True:
            (framenum1, framenum2, framenum3, framenum4, framenum5, framenum6, framenum7,framenum8) = (0, 0, 0, 0, 0, 0, 0, 0)
            if self.task_type == "average":
                while True:
                    print "FRAME INFO:", uvfits.polarization, uvfits.sub_band, uvfits.current_frame_time.get_date_time()

                    uvfits.read_data()
                    uvfits.calibration()
                    uvw_data = uvfits.compute_UVW()
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
                        uvw_data = uvfits.compute_UVW()
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

                        logger.info('Freq: %d Imsize: %d AR: %f FOV: %f CELL: %f NW: %f' % (self.current_freq, self.imsize, self.angular_resolution,self.fov, self.cell, self.number_of_wavelentgh))

        else: # self.is_loop_mode == False
            if self.task_type == "average":
                if self.Debug:
                    logger.info("Average time: %d(ms) " % time_average)
                uvfits.read_one_frame()
                self.previous_bigframe_time = uvfits.current_frame_time
                self.previous_time = uvfits.current_frame_time
                time_offset = 0
                framenum = 0
                dra = 0.14  # 0.14 # degree
                ddec = -0.1  # -0.33
                starttime = 0.

                while time_offset < self.time_average:
                    # print "FRAME INFO:", uvfits.polarization, uvfits.frequency, uvfits.current_frame_time.get_date_time()
                    uvfits.read_data()

                    if framenum == 0:
                        starttime = uvfits.current_frame_time.get_julian_date()

                    if uvfits.current_frame_header.strip_switch == 0xCCCCCCCC:
                        uvfits.delay_process('sun')

                    uvfits.calibration()
                    uvw_data, source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)

                    UU[:] += uvw_data[:, 0]
                    VV[:] += uvw_data[:, 1]

                    for chan in range(0, 16):
                        # Visibility Data
                        for baseline in range(0, self.baseline_base):
                            REAL[baseline, chan] += uvfits.baseline_data[baseline, chan].real
                            IMAG[baseline, chan] += uvfits.baseline_data[baseline, chan].imag

                    framenum += 1
                    time_offset += 3  # miliseconds

                    if uvfits.check_next_file() == True:
                        uvfits.open_next_file()
                    uvfits.read_one_frame()
                # print "Reached the time interval:" , time_offset, framenum
                endtime = uvfits.current_frame_time.get_julian_date()
                avgtime = (starttime + endtime) / 2

                uvfits.current_frame_time.from_julian_date(avgtime)
                uvfits.current_frame_utc_time.copy(uvfits.current_frame_time)
                uvfits.current_frame_utc_time.set_with_date_time(
                    uvfits.current_frame_utc_time.get_date_time() + datetime.timedelta(hours=-8))

                obs_date = ('%4d-%02d-%02d') % (
                    uvfits.current_frame_utc_time.year, uvfits.current_frame_utc_time.month,
                    uvfits.current_frame_utc_time.day)
                obs_time = ('%02d:%02d:%02d.%03d%03d%03d') % (
                    uvfits.current_frame_utc_time.hour, uvfits.current_frame_utc_time.minute,
                    uvfits.current_frame_utc_time.second,
                    uvfits.current_frame_utc_time.millisecond, uvfits.current_frame_utc_time.microsecond,
                    uvfits.current_frame_utc_time.nanosecond)

                DEC = ddec + niter * 0.02
                RA = dra / 180. * np.pi  # radian
                DEC = DEC / 180. * np.pi

                for chan in range(0, 16):

                    if self.subarray == 1:
                        self.current_freq = self.freq + chan * self.bandwidth + self.bandwidth // 2
                    else:
                        self.current_freq = self.freq + (15 - chan) * self.bandwidth + self.bandwidth // 2

                    for baseline in range(0, self.baseline_base):
                        A = math.sqrt(
                            (REAL[baseline, chan] / framenum) * (REAL[baseline, chan] / framenum) +
                            (IMAG[baseline, chan] / framenum) * (IMAG[baseline, chan] / framenum))
                        phai_sun = math.atan2(IMAG[baseline, chan] / framenum, REAL[baseline, chan] / framenum)
                        phai = phai_sun + 2 * np.pi * (UU[baseline] / framenum * self.current_freq * RA + VV[
                            baseline] / framenum * self.current_freq * DEC)
                        REAL[baseline, chan] = A * math.cos(phai)
                        IMAG[baseline, chan] = A * math.sin(phai)
                # print "Phase Calibration DONE."

                uvfits.baseline_data.real = REAL[:, :]
                uvfits.baseline_data.imag = IMAG[:, :]
                uvfits.uvws_sum, uvfits.source = uvfits.compute_UVW(obs_date, obs_time)  # units: SECONDS
                uvfits.obs_date_sum = uvfits.source.midnightJD
                uvfits.obs_time_sum = uvfits.source.JD - uvfits.source.midnightJD
                uvfits.ra_sum = uvfits.source.appra
                uvfits.dec_sum = uvfits.source.appdec

                fitsfile = ('%04d%02d%02d_%02d:%02d:%02d-%02d:%02d:%02d_I.uvfits' % (
                            self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day,
                            self.start_frame_time.hour, self.start_frame_time.minute, self.start_frame_time.second,
                            self.end_frame_time.hour, self.end_frame_time.minute, self.end_frame_time.second))
                fits_file = uvfits.env.uvfits_file(self.subarray, fitsfile)
                uvfits.pyuvfits.write_single_uvfits(fits_file)
                if self.Debug:
                    logger.info("Integration UVFITS file saved: %s" % (os.path.basename(fits_file)))

            elif self.task_type == "interval":

                if self.Debug:
                    logger.info("Time interval: %d(s)  Integration frame number: %d" % (time_interval, self.integrate_frame_num))

                dra = 0.14  # 0.14 # degree
                ddec = -0.1  # -0.33

                while True:
                    uvfits.read_one_frame()
                    uvfits.read_data()
                    self.previous_frame_time = uvfits.current_frame_time
                    # print "FRAME INFO:", uvfits.polarization, uvfits.frequency, uvfits.current_frame_time.get_date_time()

                    if uvfits.current_frame_header.strip_switch == 0xCCCCCCCC:
                        uvfits.delay_process('sun')

                    uvfits.calibration()
                    uvw_data, source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)

                    UU[:] += uvw_data[:, 0]
                    VV[:] += uvw_data[:, 1]

                    for chan in range(0, 16):
                        # Visibility Data
                        for baseline in range(0, self.baseline_base):
                            REAL[baseline, chan] += uvfits.baseline_data[baseline, chan].real
                            IMAG[baseline, chan] += uvfits.baseline_data[baseline, chan].imag

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
                                (REAL[baseline, chan]) * (REAL[baseline, chan]) +
                                (IMAG[baseline, chan]) * (IMAG[baseline, chan]))
                            phai_sun = math.atan2(IMAG[baseline, chan], REAL[baseline, chan])
                            phai = phai_sun + 2*np.pi*(UU[baseline] *self.current_freq*RA+VV[baseline]*self.current_freq*DEC)
                            REAL[baseline, chan] = A * math.cos(phai)
                            IMAG[baseline, chan] = A * math.sin(phai)
                    # print "Phase Calibration DONE."

                    uvfits.baseline_data.real = REAL[:, :]
                    uvfits.baseline_data.real = IMAG[:, :]
                    uvfits.uvws_sum, uvfits.source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)  # units: SECONDS
                    uvfits.obs_date_sum = uvfits.source.midnightJD
                    uvfits.obs_time_sum = uvfits.source.JD - uvfits.source.midnightJD
                    uvfits.ra_sum = uvfits.source.appra
                    uvfits.dec_sum = uvfits.source.appdec

                    uvfits.pyuvfits.config_primary_big(niter, error=0, if_append=True)
                    uvfits.pyuvfits.config_source_big(uvfits.source, niter)

                    niter += 1

                    time_temp = self.previous_frame_time.get_date_time() + datetime.timedelta(seconds=self.time_interval)
                    uvfits.start_date_time.set_with_date_time(time_temp)
                    uvfits.first_date_time.set_with_date_time(time_temp)
                    if self.Debug:
                        logger.info("Next time: %s  End time: %s" % (uvfits.start_date_time.get_date_time(), self.end_frame_time.get_date_time()))

                    if uvfits.start_date_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

                    if uvfits.search_first_frame() == False:
                        logger.error("Cannot find observational data.")

                hdu = uvfits.pyuvfits.make_primary_big()
                tbl_frequency = uvfits.pyuvfits.make_frequency_big()
                tbl_antenna = uvfits.pyuvfits.make_antenna(num_rows=uvfits.antennas)
                tbl_antenna = uvfits.pyuvfits.config_antenna(tbl_antenna)
                tbl_source = uvfits.pyuvfits.make_source_big()

                hdulist = pf.HDUList(
                    [hdu,
                     tbl_frequency,
                     tbl_antenna,
                     tbl_source,
                     ])
                hdulist.verify()  # Verify all values in the instance. Output verification option.

                fitsfile = ('%04d%02d%02d_%02d:%02d:%02d-%02d:%02d:%02d_I.uvfits' % (self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day, self.start_frame_time.hour, self.start_frame_time.minute, self.start_frame_time.second,
                                                                            self.end_frame_time.hour, self.end_frame_time.minute, self.end_frame_time.second))
                fits_file = uvfits.env.uvfits_file(self.subarray, fitsfile)
                if (os.path.isfile(fits_file)):
                    os.remove(fits_file)
                if self.Debug:
                    logger.info("Integration UVFITS file saved: %s" % (os.path.basename(fits_file)))
                hdulist.writeto(fits_file)

            elif self.task_type == "mixture":

                if self.Debug:
                    logger.info("Average time: %d(ms)  Time interval: %d(s)  Integration frame number: %d" % (time_average, time_interval, self.integrate_frame_num))

                while True:
                    uvfits.read_one_frame()
                    self.previous_bigframe_time = uvfits.current_frame_time
                    self.previous_time = uvfits.current_frame_time
                    time_offset = 0
                    framenum = 0
                    dra = 0.14 # 0.14 # degree
                    ddec = -0.1 # -0.33
                    starttime = 0.

                    UU = np.zeros(shape=(self.baseline_base), dtype='float64')
                    VV = np.zeros(shape=(self.baseline_base), dtype='float64')
                    REAL = np.zeros(shape=(self.baseline_base, 16), dtype='float64')
                    IMAG = np.zeros(shape=(self.baseline_base, 16), dtype='float64')

                    while time_offset < self.time_average:
                        # print "FRAME INFO:", uvfits.polarization, uvfits.frequency, uvfits.current_frame_time.get_date_time()
                        uvfits.read_data()
                        if uvfits.current_frame_header.strip_switch == 0xCCCCCCCC:
                            uvfits.delay_process('sun')
                        uvfits.calibration()

                        if framenum == 0:
                            starttime = uvfits.current_frame_time.get_julian_date()

                        uvw_data, source = uvfits.compute_UVW(uvfits.obs_date, uvfits.obs_time)

                        UU[:] += uvw_data[:, 0]
                        VV[:] += uvw_data[:, 1]

                        for chan in range(0, 16):
                            # Visibility Data
                            for baseline in range(0, self.baseline_base):
                                REAL[baseline, chan] += uvfits.baseline_data[baseline, chan].real
                                IMAG[baseline, chan] += uvfits.baseline_data[baseline, chan].imag

                        framenum += 1
                        time_offset += 3 #miliseconds

                        if uvfits.check_next_file() == True:
                            uvfits.open_next_file()
                        uvfits.read_one_frame()
                    # print "Reached the time interval:" , time_offset, framenum
                    endtime = uvfits.current_frame_time.get_julian_date()
                    avgtime = (starttime+endtime)/2

                    uvfits.current_frame_time.from_julian_date(avgtime)
                    uvfits.current_frame_utc_time.copy(uvfits.current_frame_time)
                    uvfits.current_frame_utc_time.set_with_date_time(
                        uvfits.current_frame_utc_time.get_date_time() + datetime.timedelta(hours=-8))

                    obs_date = ('%4d-%02d-%02d') % (
                        uvfits.current_frame_utc_time.year, uvfits.current_frame_utc_time.month,
                        uvfits.current_frame_utc_time.day)
                    obs_time = ('%02d:%02d:%02d.%03d%03d%03d') % (
                        uvfits.current_frame_utc_time.hour, uvfits.current_frame_utc_time.minute,
                        uvfits.current_frame_utc_time.second,
                        uvfits.current_frame_utc_time.millisecond, uvfits.current_frame_utc_time.microsecond,
                        uvfits.current_frame_utc_time.nanosecond)

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
                                (REAL[baseline, chan]/framenum) * (REAL[baseline, chan]/framenum) +
                                (IMAG[baseline, chan]/framenum) * (IMAG[baseline, chan]/framenum))
                            phai_sun = math.atan2(IMAG[baseline, chan]/framenum, REAL[baseline, chan]/framenum)
                            phai = phai_sun + 2*np.pi*(UU[baseline]/framenum *self.current_freq*RA+VV[baseline]/framenum *self.current_freq*DEC)
                            REAL[baseline, chan] = A * math.cos(phai)
                            IMAG[baseline, chan] = A * math.sin(phai)
                    # print "Phase Calibration DONE."

                    uvfits.baseline_data.real = REAL[:, :]
                    uvfits.baseline_data.imag = IMAG[:, :]
                    uvfits.uvws_sum, uvfits.source = uvfits.compute_UVW(obs_date, obs_time)  # units: SECONDS
                    uvfits.obs_date_sum = uvfits.source.midnightJD
                    uvfits.obs_time_sum = uvfits.source.JD - uvfits.source.midnightJD
                    uvfits.ra_sum = uvfits.source.appra
                    uvfits.dec_sum = uvfits.source.appdec

                    niter += 1
                    uvfits.pyuvfits.config_primary_big(niter, error=0, if_append=True)
                    uvfits.pyuvfits.config_source_big(uvfits.source, niter)

                    time_temp = self.previous_bigframe_time.get_date_time() + datetime.timedelta(seconds=self.time_interval)
                    uvfits.start_date_time.set_with_date_time(time_temp)
                    uvfits.first_date_time.set_with_date_time(time_temp)
                    if self.Debug:
                        logger.info("Next time: %s  End time: %s" % (uvfits.start_date_time.get_date_time(), self.end_frame_time.get_date_time()))

                    if uvfits.start_date_time.get_date_time() > self.end_frame_time.get_date_time():
                        break

                    if uvfits.search_first_frame() == False:
                        logger.error("Cannot find observational data.")

                uvfits.pyuvfits.config_frequency_big()
                hdu = uvfits.pyuvfits.make_primary_big()
                tbl_frequency = uvfits.pyuvfits.make_frequency_big()
                tbl_antenna = uvfits.pyuvfits.make_antenna(num_rows=uvfits.antennas)
                tbl_antenna = uvfits.pyuvfits.config_antenna(tbl_antenna)
                tbl_source = uvfits.pyuvfits.make_source_big()

                hdulist = pf.HDUList(
                    [hdu,
                     tbl_frequency,
                     tbl_antenna,
                     tbl_source,
                     ])
                hdulist.verify()  # Verify all values in the instance. Output verification option.

                fitsfile = ('%04d%02d%02d_%02d:%02d:%02d-%02d:%02d:%02d_I.uvfits' % (self.start_frame_time.year, self.start_frame_time.month, self.start_frame_time.day, self.start_frame_time.hour, self.start_frame_time.minute, self.start_frame_time.second,
                                                                            self.end_frame_time.hour, self.end_frame_time.minute, self.end_frame_time.second))
                fits_file = uvfits.env.uvfits_file(self.subarray, fitsfile)
                if (os.path.isfile(fits_file)):
                    os.remove(fits_file)
                if self.Debug:
                    logger.info("Integration UVFITS file saved: %s" % (os.path.basename(fits_file)))
                hdulist.writeto(fits_file)













