#! /usr/bin/env python
# encoding: utf-8

from __future__ import division
import math
import time, datetime, string
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python') + 7]
sys.path.append(path1)

from pymuser.muserdata import *
#from pymuser.muserenv import *
from pymuser.musertime import *
from argparse import *


# Trainset Class
class Trainset:
    def __init__(self, sub_array, filename, outfile):
        """
        Main function call. Process raw data: delay process and sum
        """

        self.sub_array = sub_array
        self.manualOrAuto = 0
        self.filename = filename
        self.outfilename = outfile

        # CSRHRawData Class
        self.muser = MuserData(self.sub_array, self.manualOrAuto)

    def open_raw_data(self):
        self.muser.set_data_date_time()
        if self.muser.open_raw_file(self.filename) == False:
            print "Cannot find observational data."
            return False

        return True

    def write_file(self):

        if self.open_raw_data() == True:
            print "Source file: ", self.filename
        else:
            exit(0)

        print "Writing training set file:", self.outfilename

        #Modify here
        band_wanted = 0
        channel_wanted = 0
        pol_wanted = 0
        train = open(self.outfilename, 'w')

        NFrame = 0
        while (NFrame < 19200):     # 19200 frames/minute
            if self.muser.read_one_frame() == False:
                print ("Cannot read enough data. ")
                self.muser.close_raw_file()
                exit(0)
            # Read data
            if self.muser.sub_band == band_wanted and self.muser.polarization == pol_wanted:
                self.muser.read_data()
                print self.muser.current_frame_time.get_string()

                (column1, column2, column3, column4, column5) = (1, 2, 3, 4, 5)
                for chan in range(channel_wanted, channel_wanted+1):  #0, self.muser.sub_channels):
                    bl = 0
                    for ant1 in range(0, self.muser.antennas - 1):
                        for ant2 in range(ant1 + 1, self.muser.antennas):  # Add class label for each visibility
                            #print "VISIBILITY*****************",csrh.csrhData[bl][chan].real, csrh.csrhData[bl][chan].imag

                            amplitude = math.sqrt(
                                self.muser.baseline_data[bl][chan].real * self.muser.baseline_data[bl][chan].real +
                                self.muser.baseline_data[bl][chan].imag * self.muser.baseline_data[bl][chan].imag)
                            #bl = bl + 1
                            pha = math.atan2(self.muser.baseline_data[bl][chan].imag,
                                             self.muser.baseline_data[bl][chan].real)
                            if ant1 in [4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 24, 25, 26, 36, 38] or ant2 in [4, 7, 10, 11, 12,
                                                                                                              36, 38, 39]:
                                label = -1
                            else:
                                label = +1

                            # print csrh.csrhData[bl][chan].real
                            train.write("%+d %d%c%d %d%c%d %d%c%d %d%s%8.10f %d%s%8.10f \n" % (
                                label, column1, ':', chan, column2, ':', ant1, column3, ':', ant2, column4, ':', amplitude,
                                column5,
                                ':', pha))
                            bl = bl+1
                        # train.write("%d  %d  %d  %8.10f  %8.10f  %8.10f\n"%(chan,ant1,ant2,csrhdata[ant1][ant2][chan].real,csrhdata[ant1][ant2][chan].imag,amplitude))
                self.muser.skip_frames(398)
            NFrame += 1
        train.close()
        print "Done."

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help="The observational file", dest='infile',
                        default="", required=True)
    parser.add_argument('-o', '--ofile', help="The observational file", dest='outfile',
                        default="", required=True)

    args = parser.parse_args()

    infile = args.infile
    outfile = args.outfile
    # Check file existed?
    if not os.path.exists(infile):
        print "Error: cannot find the observational raw data"
        exit(0)
    filesize = os.path.getsize(infile)
    if filesize == 1920000000:
        sub_array = 1
    elif filesize == 3932160000:
        sub_array = 2
    else:
        print "Error: the file is not a proper file of MUSER."
        exit(0)

    muser = Trainset(sub_array, infile, outfile)

    muser.write_file()
