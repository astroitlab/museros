import os
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserdirty import *
from argparse import *

from taskinit import *
import datetime
import traceback


def dirtymap(
    subarray=None,
    polarization=None,
    frequency = None,
    visfile=None,
    uvfile=None,
    ra = None,
    dec = None,
    outdir = None,
    plot = True,
    fits = True,
    correct = False,
    debug=None,
    ):
    #subarray, polarization, frequency, vis_file, uv_file, ra, dec, outdir, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG
    try:
        #muserlog.origin('writeuvfits')
        # -----------------------------------------'
        # beginning of importmiriad implementation
        # -----------------------------------------
        #obsFileName = valid_date(start)

        dirty = MuserDirty()
        dirty.dirty_realtime(subarray, polarization, frequency, visfile, uvfile, ra, dec, outdir, plot, fits, correct, debug)

    except Exception, e:
        print traceback.format_exc()
    	muserlog.post("Failed to import muser file %s" % uvfile)
    return


