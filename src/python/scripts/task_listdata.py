from pymuser.muserenv import muserenv
from pymuser.muserfile import *
import os
import string
import time,datetime

class FirstFrame:
    def __init__(self, sub_array):
        self.sub_array = sub_array
        # CSRHRawData Class
        self.muser = MuserData(self.sub_array)

    def get_first_frame_time(self, i, inputfile = None):

        self.muser.input_file_name=inputfile

        if self.muser.open_data_file() == False:
            print "Cannot find observational data."
            return str(False), []

        if self.muser.read_one_frame() == False:
            return str(False), []

        print ('%-5.5s %-16.16s %-30.30s %-5.5s %-10.10s' % (str(i+1), os.path.basename(self.muser.input_file_name), self.muser.current_frame_time.get_fits_date_time(), 'LL' if self.muser.polarization==0 else 'RR', str(int(self.muser.frequency//1E6))))


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
            raise
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise

def listdata\
    (
    muser=None,
    start=None,
    debug=None
    ):
    if start is not None and len(start)>0:
        start_date = valid_date(start)
    else:
        return
    startdir = muserenv.data_dir(muser, start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute)

    #listfile=os.listdir(info)
    # LIst all info
    # print ('%-5.5s %-30.30s  %-12.12s  %-18.18s' % ('No.', 'Date/Time','Polarization','Frequency(MHz)'))

    print ('%-5.5s %-16.16s %-30.30s %-5.5s %-10.10s' % ('No.','File', 'Time', 'Pol.','Freq(MHz)'))
    #os.chdir(info)
    #s=len(listfile)
    i = 0
    ff = FirstFrame(muser)
    for dirpath, dirnames, filenames in os.walk(startdir):
            for fitsfile in filenames:
                if len(fitsfile.split('-')[1]) == 4:
                    filename = os.path.join(startdir, fitsfile)
                    try:
                        ff.get_first_frame_time(i, filename)
                        i += 1
                    finally:
                        pass

