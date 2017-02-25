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
import datetime

class FirstFrame:
    def __init__(self, sub_array, start_date=None, inputfile=None):
        """
        Main function call. Process raw data: delay process and sum
        """
        self.start_date = None
        self.inputfile = None
        self.sub_array = sub_array
        self.manualOrAuto = 0
        if start_date is not None and len(start_date)>0:
            self.start_date = valid_date(start_date)
        if inputfile is not None and len(inputfile) > 0:
            self.inputfile = inputfile
        # CSRHRawData Class
        self.muser = MuserData(self.sub_array)

    def get_first_frame_time(self, number=1):
        if self.start_date is not None:
            self.muser.set_data_date_time(self.start_date.date().year, self.start_date.date().month, self.start_date.date().day,
                                      self.start_date.time().hour, self.start_date.time().minute,
                                      self.start_date.time().second, 0, 0, 0)
        if self.inputfile is not None:
            self.muser.input_file_name=self.inputfile

        if self.muser.open_data_file() == False:
            print "Cannot find observational data."
            return str(False), []

        print ('%-5.5s %-30.30s  %-12.12s  %-18.18s' % ('No.', 'Date/Time','Polarization','Frequency(MHz)'))
        for i in range(number):
            if self.muser.read_one_frame() == False:
                return str(False), []
            print ('%-5.5s %-30.30s  %-12.12s  %-18.18s' % (str(i+1), self.muser.current_frame_time.get_fits_date_time(), 'LL' if self.muser.polarization==0 else 'RR', str(int(self.muser.frequency//1E6))))


def valid_date(s):
    try:
        s = s.strip()
        split_s = string.split(s, ' ')
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

def headdata(
    muser=None,
    inputfile=None,
    start=None,
    frame=None,
    debug=None,
    ):

    try:

        muserlog.origin('headdata')
        # -----------------------------------------'
        # beginning of importmiriad implementation
        # -----------------------------------------
        #start_date = valid_date(start)

        uvfits = FirstFrame(muser, start, inputfile)
        uvfits.get_first_frame_time(frame)

    except Exception, e:
    	muserlog.post("Failed to display head information.")
    return


