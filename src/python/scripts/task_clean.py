import os
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserclean import *
from argparse import *

from taskinit import *
import datetime
import traceback


def clean (
    inputfile=None,
    outdir = None,
    channel = None,
    weight= 'natural',
    mode = 'hogbom',
    automove = True,
    movera = 0.,
    movedec = 0.,
    plot = True,
    fits = False,
    correct = False,
    debug=None,
    ):

    try:
        #muserlog.origin('writeuvfits')
        # -----------------------------------------'
        # beginning of importmiriad implementation
        # -----------------------------------------
        #obsFileName = valid_date(start)

        clean = MuserClean()
        clean.band = None
        if mode=='hybrid':
            clean.hybrid_clean_with_fits(inputfile, outdir, channel, weight, mode, automove, movera, movedec, plot, fits, correct,  debug)
        else:
            clean.clean_with_fits(inputfile, outdir, channel, weight, mode, automove, movera, movedec, plot, fits, correct,  debug)


    except Exception, e:
        print traceback.format_exc()
    	muserlog.post("Failed to import muser file %s" % inputfile)
    return


