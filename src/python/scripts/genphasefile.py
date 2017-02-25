#! /usr/bin/env python
# encoding: utf-8
"""
"""

from __future__ import division
import sys,os, string, datetime


import numpy as np
from argparse import *
path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserenv import *
from pymuser.muserdata import *

class Phase:
    def __init__(self, sub_array, is_loop_mode, obs_date_time, frame_number, calibration_sequence):
        self.sub_array = sub_array
        self.is_loop_mode = is_loop_mode
        self.obs_date_time = obs_date_time
        self.frame_number = frame_number
        self.calibration_sequence = calibration_sequence
        self.data_source = 0
        self.env = MuserEnv()
        self.year = obs_date_time.date().year
        self.month = obs_date_time.date().month
        self.day = obs_date_time.date().day
        self.hour = obs_date_time.time().hour
        self.minute = obs_date_time.time().minute

    def Calibration(self):
        if self.frame_number <= 0:
            print "You should input a positive number!"
            return False

        muser_calibration = MuserData(self.sub_array, 0)
        muser_calibration.set_data_date_time(self.year, self.month, self.day, self.hour, self.minute,0,0,0,0)
        print('\nReading Visibility Data of calibration......')
        if muser_calibration.open_data_file() == False:
            print 'Error: cannot find the data file.'
            exit(0)

        muser_calibration.skip_frames(self.frame_number)

        calibration_Data = np.ndarray(
            shape=(muser_calibration.frame_number, muser_calibration.polarization_number,
                   muser_calibration.antennas * (muser_calibration.antennas - 1) // 2, 16),
            dtype=complex)

        self.last_sub_band = -1
        self.last_polarization = -1
        if is_loop_mode==True:
            repeat_number = 1
        else:
            repeat_number = muser_calibration.frame_number * 2

        for i in range(repeat_number):
            if (muser_calibration.read_one_frame() == False):  # 32*8bits
                print('Cannot read a frame.')
                return False
            muser_calibration.read_data()
            print "Reading No.", i, muser_calibration.current_frame_time.get_string(), " ", muser_calibration.sub_band, " ", muser_calibration.polarization
            muser_calibration.delay_process('satellite')
            self.last_sub_band =  muser_calibration.sub_band
            self.last_polarization = muser_calibration.polarization

            bl = 0
            for antenna1 in range(0, muser_calibration.antennas - 1):
                for antenna2 in range(antenna1 + 1, muser_calibration.antennas):
                    # muser_calibration.csrhData[muser_calibration.Polarization][antenna1][antenna2][channel] = c1
                    for channel in range(0, muser_calibration.sub_channels):

                        calibration_Data[muser_calibration.sub_band][muser_calibration.polarization][bl][channel] = muser_calibration.baseline_data[bl][channel]
                        # if channel ==0 and antenna1==0:
                        #        print channel, antenna1, antenna2, muser_calibration.baseline_data[bl][channel]
                    bl = bl + 1
            print "*******calibration_Data******:"

        file_name = self.env.cal_file(self.sub_array,self.year, self.month, self.day, self.calibration_sequence)
        print "Writing to file: ", os.path.basename(file_name)
        calibration_Data.tofile(file_name)
        print "Done."
        return True

def valid_date(split_s):
    try:
        s = string.join(split_s, ' ')
        if len(split_s) == 1:
            return datetime.datetime.strptime(s, "%Y-%m-%d")
        elif len(split_s) == 2:
            return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        elif len(split_s) == 3:
            s = string.join(split_s, ' ')
            return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %f")
        else:
            msg = "Not a valid date: '{0}'.".format(s)
            raise ArgumentTypeError(msg)
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise ArgumentTypeError(msg)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-m', '--muser', help="Define the sub array (Muser-I/II)", dest='sub_array', choices=[1, 2],
                        type=int, default=1)
    parser.add_argument('-l', "--is_loop_mode", help="Loop Mode or not", dest='is_loop_mode',
                        type=int, required=True)
    parser.add_argument('-s', "--startdate", help="Date format YYYY-MM-DD [HH:MM:SS]", dest='start_date',
                        nargs='+', required=True)

    parser.add_argument('-f', "--frame", help="Skip frames", dest='frames', type=int, default = 0)
    parser.add_argument('-c', "--calibration", help="Calibration sequence number", dest='calibration', type=int, default = 0)
    parser.add_argument('-d', "--debug", help="DEBUG", dest='debug', type=int, default = 0)

    args = parser.parse_args()

    sub_array = args.sub_array
    obsFileName = valid_date(args.start_date)
    frameNum = args.frames
    calibration_sequence = args.calibration
    is_loop_mode = args.is_loop_mode

    cal = Phase(sub_array, is_loop_mode, obsFileName, frameNum, calibration_sequence)
    cal.Calibration()
