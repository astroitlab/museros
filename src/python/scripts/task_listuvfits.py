from pymuser.muserenv import muserenv
import struct
import binascii
import pyfits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
import os
import time
import sys
import re
import struct
import binascii
import time,datetime, string
import globalpy
#from matplotlib import mplDeprecation
import matplotlib as mpl


from argparse import *

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

def data_dir(self, sub_array, year, month, day, hour, minute):
        file_name = ('%04d%02d%02d-%02d%02d') % (year, month, day, hour, minute)
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/dat/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        #full_file_name = os.path.join(file_path, file_name)
        return file_path


def listuvfits (
    muser=None,
    start='',
    debug=None,
    ):
    if start is not None and len(start)>0:
        start_date = valid_date(start)
    else:
        return
    startdir = muserenv.uvfits_dir(muser, start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute)
    #listfile=os.listdir(info)

    print ('%-5.5s %-35.35s %-5.5s %-10.10s' % ('No.','File', 'Pol.','Freq(GHz)'))
    #os.chdir(info)
    #s=len(listfile)
    i = 0
    for dirpath, dirnames, filenames in os.walk(startdir):
            for fitsfile in filenames:
                if os.path.splitext(fitsfile)[1] == '.uvfits':
                    filename = os.path.join(startdir, fitsfile)
                    try:
                        hdulist = pyfits.open(filename, mode='readonly', ignore_missing_end=True)

                        # hdulist.info()
                        object = hdulist[0].header['OBJECT']
                        polarization = np.int32( hdulist[0].header['CRVAL3'])
                        basefreq = np.float32( hdulist[0].header['CRVAL4'])
                        freq = (basefreq + np.float32( hdulist[1].data["IF FREQ"][0]))/1E9

                        g= os.path.getsize(filename)
                        d= os.path.getctime(filename)
                        h=time.ctime(d)
                        time_original = h
                        time_format = datetime.datetime.strptime(time_original, '%a %b %d %H:%M:%S %Y')
                        time_format = time_format.strftime('%Y-%m-%d  %H:%M:%S')
                        print ('%-5.5s %-35.35s %-5.5s %-10.10s' % (str(i+1),fitsfile, 'LL' if polarization==-2 else 'RR', freq))
                        i += 1
                    finally:
                        hdulist.close()
