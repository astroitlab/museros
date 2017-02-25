import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.integrationuvfits import *
from argparse import *

from taskinit import *
import datetime
import traceback

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

def integrationuvfits (
    subarray = None,
    is_loop_mode = True,
    start_time = None,
    end_time = None,
    task_type = None,
    time_average = None,
    time_interval = None,
    debug=None,
    ):

    try:
        #muserlog.origin('writeuvfits')
        # -----------------------------------------'
        # beginning of importmiriad implementation
        # -----------------------------------------
        #obsFileName = valid_date(start)

        iuvfits = Muser_IntegrationUVFITS()
        iuvfits.integrationuvfits(subarray, is_loop_mode, valid_date(start_time), valid_date(end_time), task_type, time_average, time_interval, debug)

    except Exception, e:
        print traceback.format_exc()
    return


