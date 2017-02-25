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
from pymuser import museriers

from taskinit import *
import datetime

def synciers (
    debug=None,
    ):

    try:
        # muserlog.origin('Task: Synciers')
        # -----------------------------------------'
        iers = museriers.MuserIers()
        iers.update()
        # muserlog.post('Task: Synciers finished')
    except Exception, e:
        pass
    	# muserlog.post("Failed to synchronize IERS data.")
    return


