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
from argparse import *


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

    parser.add_argument('-i', "--integral", help="Integral number", dest='integral', type=int, default=1, )
    parser.add_argument('-r', "--repeat", help="repeat number", dest='repeat', type=int, default=1)
    parser.add_argument('-c', "--calibration", help="Calibration sequence number", dest='calibration', type=int,
                        default=0)
    parser.add_argument('-n',"--nocalibration",help="Do not calibrate the data",dest='nocalibration',type=int,
                        default=1)

    parser.add_argument('-f',"--file",help="Input filename",dest='inputfile', default="")
    args = parser.parse_args()

    start_date = valid_date(args.start_date)

    uvfits = MuserFile(args.sub_array, args.data_source)
    uvfits.set_parameters(start_date, args.integral, args.repeat, args.calibration, args.nocalibration, args.inputfile)

    if args.integral > 1:
        uvfits.write_integral_uvfits(False)
    else:
        uvfits.write_single_uvfits(False)
