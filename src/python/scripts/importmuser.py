#
# This file was generated using xslt from its XML file
#
# Copyright 2009, Associated Universities Inc., Washington DC
#
import sys
import os
from  casac import *
import string
from taskinit import casalog
#from taskmanager import tm
import task_importmiriad
def importmiriad(mirfile='', vis='', tsys=False, spw='all', vel='', linecal=False, wide='all', debug=0):

        """Convert a Miriad visibility file into a CASA MeasurementSet

        importmiriad(mirfile='ngc5921.uv', vis='ngc5921.ms',tsys=True)

 
        """

#
#    The following is work around to avoid a bug with current python translation
#
        mytmp = {}

        mytmp['mirfile'] = mirfile
        mytmp['vis'] = vis
        mytmp['tsys'] = tsys
        mytmp['spw'] = spw
        mytmp['vel'] = vel
        mytmp['linecal'] = linecal
        mytmp['wide'] = wide
        mytmp['debug'] = debug
	pathname='file:///'+os.environ.get('CASAPATH').split()[0]+'/'+os.environ.get('CASAPATH').split()[1]+'/xml/'
        trec = casac.utils().torecord(pathname+'importmiriad.xml')

        casalog.origin('importmiriad')
        if trec.has_key('importmiriad') and casac.utils().verify(mytmp, trec['importmiriad']) :
	    result = task_importmiriad.importmiriad(mirfile, vis, tsys, spw, vel, linecal, wide, debug)

	else :
	  result = False
        return result
