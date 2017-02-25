#! /usr/bin/env python
# encoding: utf-8
"""
Creates a UVFITS file for CSRH-I.
"""
from __future__ import division
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserfile import *
from pymuser.muserlogger import *
from pymuser.muserenv import *
from argparse import *
import logging

thelogfile = 'muser.log'
muserlog = MuserLogger()
muserlog.setLogger(muserenv.get_log_dir()+os.path.sep+thelogfile, levelconsole=logging.DEBUG,  filelog=True, consolelog=True) #showconsole)


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
    parser.add_argument('-s', "--startdate", help="Date format YYYY-MM-DD [HH:MM:SS]", dest='start_date',
                        nargs='+', required=True)
    parser.add_argument('-d', "--datatext", help="Parameters from text file", dest='data_source', default=0,
                        choices=[0, 1], type=int)

    parser.add_argument('-i', "--integral", help="Integral number", dest='integral', type=int, default=1)
    parser.add_argument('-r', "--repeat", help="repeat number", dest='repeat', type=int, default=1)
    parser.add_argument('-c', "--calibration", help="Calibration sequence number", dest='calibration', type=int,
                        default=0)
    parser.add_argument('-n',"--nocalibration",help="Do not calibrate the data",dest='nocalibration',type=int,
                        default=1)

    parser.add_argument('-f',"--file",help="Input filename",dest='inputfile', default="")

    parser.add_argument('-g', '--genraw',help="Generate raw data", dest='genraw',default=0)
    parser.add_argument('-b', '--big', help="Generate one big  sfile",  action="store_true", dest='big',default=False)
    args = parser.parse_args()

    start_date = valid_date(args.start_date)


    uvfits = MuserFile(args.sub_array, args.data_source)
    uvfits.set_parameters(start_date, args.integral, args.repeat, args.calibration, args.nocalibration, args.inputfile, args.genraw)

    if args.integral == 1:
        if args.big == False:
            uvfits.write_single_uvfits()
        elif args.big == True and args.repeat > 1:
            uvfits.write_one_big_uvfits()
        else:
            print "Please input reasonable parameter. "
            print "bigfile=True: niter>1 "
    else:
        uvfits.write_integral_uvfits()


