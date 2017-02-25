#! /usr/bin/env python
# encoding: utf-8
"""

"""
from __future__ import division
import os, sys, time

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.musertime import *
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
    parser.add_argument('-s', "--startdate", help="Date format YYYY-MM-DD [HH:MM:SS] [SSSSSS]", dest='start_date',
                        nargs='+', required=True)

    parser.add_argument('-e', "--enddate", help="Date format YYYY-MM-DD [HH:MM:SS] [SSSSSS]", dest='end_date',
                        nargs='+', required=True)

    parser.add_argument('-i', "--integral", help="Integral time (unit: second)", dest='integral_time', type=float,
                        required=True)

    args = parser.parse_args()

    start_date = valid_date(args.start_date)
    end_date = valid_date(args.end_date)

    start_time = MuserTime()
    start_time.set_with_date_time(start_date)

    end_time = MuserTime()
    end_time.set_with_date_time(end_date)

    integral_time = args.integral_time

    muser = MuserFile(args.sub_array)
    print muser.get_data_info(start_time, end_time, integral_time)
