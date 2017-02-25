import os

from importmuser_cli import  importmuser_cli as importmuser
from exportuvfits_cli import exportuvfits_cli as exportuvfits
from exportphase_cli import exportphase_cli as exportphase
from listuvfits_cli import listuvfits_cli as listuvfits
from listdata_cli import listdata_cli as  listdata
if os.environ.get('MUSERGPU')=='TRUE':
    from clean_cli import clean_cli as clean
    from dirtymap_cli import dirtymap_cli as dirtymap
    from cleanuvfits_cli import cleanuvfits_cli as cleanuvfits
    from integrationcleanR_cli import integrationcleanR_cli as integrationclean
    # from clean_cl_cli import clean_cl_cli as clean_cl
from headdata_cli import headdata_cli as headdata
from synciers_cli import synciers_cli as synciers
from ephemeris_cli import ephemeris_cli as ephemeris
from integrationuvfits_cli import integrationuvfits_cli as integrationuvfits

#from listdata_cli import listdata_cli as listdata
#from tget import *
