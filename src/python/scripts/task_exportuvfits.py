import os
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserfile import *
from argparse import *

from taskinit import *
import traceback
import datetime


def valid_date(s):
    try:
        s = s.strip()
        if s==None or len(s)==0:
            s='1970-1-1'
        split_s = string.split(s, ' ')
        # print split_s, len(split_s)
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

def exportuvfits (
    muser=None,
    inputfile=None,
    start=None,
    integral=1,
    niter=1,
    hourangle=999,
    declination=999,
    correct=0,
    calibration=True,
    bigfile=False,
    debug=None,
    ):

    try:
        muserlog.origin('exportuvfits')

        # -----------------------------------------'
        # beginning of importmiriad implementation
        # -----------------------------------------
        uvfits = MuserFile(muser)
        uvfits.set_parameters(valid_date(start), integral, niter, hourangle, declination, correct, calibration, inputfile, debug, genraw=0)
        if integral > 1:
            uvfits.write_integral_uvfits()
        else:
            if bigfile == True:
                uvfits.write_one_big_uvfits()
            else:
                # uvfits.get_visdata()
                uvfits.write_single_uvfits()
                # uvfits.merge_one_big_uvfits()
    except Exception, e:
        print traceback.print_exc()
    	muserlog.post("Failed to export uvfits file")
        raise
    return


