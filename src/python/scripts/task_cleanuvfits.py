import os
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.cleanuvfits import *
import traceback


def cleanuvfits(
    infile = None,
    is_loop_mode = True,
    mode = None,
    weight = None,
    band = None,
    channel = None,
    polarization = 0,
    movera = 0,
    movedec = 0,
    P_ANGLE = False,
    fits = 0,
    debug=None,
    outdir=None,
    ):

    try:
        clean = Muser_CleanUVFITS()
        clean.cleanuvfits(infile, is_loop_mode, mode, weight, band, channel, polarization, movera, movedec, P_ANGLE, fits, debug, outdir)

    except Exception, e:
        print traceback.format_exc()
    return


