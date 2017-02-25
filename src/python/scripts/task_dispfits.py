__author__ = 'Administrator'
from matplotlib import mplDeprecation
import matplotlib as mpl
import pyfits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def dispfits (
    muser=None,
    start='',
    debug=None,
    ):
    self.muser.input_file_name=start
    for dirpath, dirnames, filenames in os.walk(start):
            for fitsfile in filenames:
                    filename = os.path.join(start, fitsfile)
                    fig = plt.figure()
                    hdulist = pyfits.open(filename, mode='readonly1', ignore_missing_end=True)
                    hdulist.info()
                    b=hdulist[0].header.items()
                    scidata=hdulist[0].data
                    ax=fig.add_subplot(111)
                    #cmap = mpl.cm.summer
                    norm = mpl.colors.Normalize(vmin=0, vmax=0)
                    im=ax.imshow(scidata)
                    plt.colorbar(im,norm=norm,ticks=[0,0,0])
                    plt.show()
