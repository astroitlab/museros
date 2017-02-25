#! /usr/bin/env python
# encoding: utf-8
"""

"""
from __future__ import division
import os, sys, time
from shutil import copy

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.musertime import *
from pymuser.muserfile import *

from argparse import *

class Archive:
    def __init__(self, sub_array, filename):
        """
        Main function call. Process raw data: delay process and sum
        """

        self.sub_array = sub_array
        self.manualOrAuto = 0
        self.filename = filename

        # CSRHRawData Class
        self.muser = MuserData(self.sub_array, self.manualOrAuto)

    def get_first_frame_time(self):
        self.muser.set_data_date_time()
        if self.muser.open_raw_file(self.filename) == False:
            print "Cannot find observational data."
            exit(0)

        if self.muser.read_one_frame() == False:
            print "Error reading frame."
            exit(0)

        return self.muser.current_frame_time.get_detail_time()

    def archive(self):
        year,month,day,hour,minute,second,ms,ns,ps = self.get_first_frame_time()
        print "Source file: ", self.filename
        newfile = self.muser.env.data_file(self.sub_array,year,month,day,hour,minute)
        print "Archive file:" , newfile
        copy(self.filename, newfile)
        print "Done."

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help="The observational file", dest='infile',
                        default="",required=True)

    args = parser.parse_args()

    infile = args.infile
    #Check file existed?
    if not os.path.exists(infile):
        print "Error: cannot find the observational raw data"
        exit(0)
    filesize = os.path.getsize(infile)
    if filesize <= 1920000000:
        sub_array = 1
    elif filesize == 3932160000:
        sub_array = 2
    else:
        print "Error: the file is not a proper file of MUSER."
        exit(0)


    muser = Archive(sub_array, infile)

    muser.archive()
