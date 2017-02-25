#! /usr/bin/env python
# encoding: utf-8
"""
createCSRHUVFITS.py
=====================
Creates a UVFITS file for CSRH-I.
"""
from __future__ import division
import time, datetime, string
import os, sys
path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserdata import *
#from pymuser.muserenv import *
from argparse import *


class FirstFrame:
    def __init__(self, sub_array, start_date):
        """
        Main function call. Process raw data: delay process and sum
        """

        self.sub_array = sub_array
        self.manualOrAuto = 0

        self.start_date = start_date
        # CSRHRawData Class
        self.muser = MuserData(self.sub_array, self.manualOrAuto)

    def get_first_frame_time(self):

        self.muser.set_data_date_time(self.start_date.date().year, self.start_date.date().month, self.start_date.date().day,
                                      self.start_date.time().hour, self.start_date.time().minute,
                                      self.start_date.time().second, 0, 0, 0)

        if self.muser.open_data_file() == False:
            print "Cannot find observational data."
            return str(False), []

        if self.muser.read_one_frame() == False:
            return str(False), []

        return str(True), self.muser.current_frame_time.get_detail_time()


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
    parser.add_argument('-s', "--startdate", help="Date format YYYY-MM-DD [HH:MM:SS] [SSSSSS]", dest='start_date',
                        nargs='+', required=True)
    # parser.add_argument('–version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()

    start_date = valid_date(args.start_date)

    uvfits = FirstFrame(args.sub_array, start_date)
    print uvfits.get_first_frame_time()
