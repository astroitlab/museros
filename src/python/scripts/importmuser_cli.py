#
# This file was generated using xslt from its XML file
#
# Copyright 2014, Associated Universities Inc., Washington DC
#
import sys
import os
#from muserc import *
#import muserc
import string
import time
import inspect
import gc
import numpy
from odict import odict
from types import * 
from task_importmuser import importmuser

class importmuser_cli_:
    __name__ = "importmuser"
    rkey = None
    i_am_a_muserpy_task = None
    # The existence of the i_am_a_muserpy_task attribute allows help()
    # (and other) to treat muserpy tasks as a special case.

    def __init__(self) :
       self.__bases__ = (importmuser_cli_,)
       self.__doc__ = self.__call__.__doc__

       self.parameters={'mirfile':None, 'vis':None, 'tsys':None, 'spw':None, 'vel':None, 'linecal':None, 'wide':None, 'debug':None, }


    def result(self, key=None):
	    #### and add any that have completed...
	    return None


    def __call__(self, mirfile=None, vis=None, tsys=None, spw=None, vel=None, linecal=None, wide=None, debug=None, ):

        """Convert a Miriad visibility file into a muser MeasurementSet

	Detailed Description:

        Convert a Miriad visibility file into a muser MeasurementSet with
	optional selection of spectral windows and weighting scheme

	Arguments :
		mirfile:	Name of input Miriad visibility file
		   Default Value:

		vis:	Name of output MeasurementSet
		   Default Value:

		tsys:	Use the Tsys to set the visibility weights
		   Default Value: False

		spw:	Select spectral windows
		   Default Value: all

		vel:	Select velocity reference (TOPO,LSRK,LSRD)
		   Default Value:

		linecal:	(CARMA) Apply line calibration
		   Default Value: False

		wide:	(CARMA) Select wide window averages
		   Default Value: all

		debug:	Display increasingly verbose debug messages
		   Default Value: 0

	Returns: void

	Example :


        importmiriad(mirfile='ngc5921.uv', vis='ngc5921.ms',tsys=True)


        """
	if not hasattr(self, "__globals__") or self.__globals__ == None :
           self.__globals__=sys._getframe(len(inspect.stack())-1).f_globals
	#muserc = self.__globals__['muserc']
	muserlog = self.__globals__['muserlog']
	#muserlog = muserc.muserc.logsink()
        self.__globals__['__last_task'] = 'importmuser'
        self.__globals__['taskname'] = 'importmuser'
        ###
        self.__globals__['update_params'](func=self.__globals__['taskname'],printtext=False,ipython_globals=self.__globals__)
        ###
        ###
        #Handle globals or user over-ride of arguments
        #
        if type(self.__call__.func_defaults) is NoneType:
            function_signature_defaults={}
	else:
	    function_signature_defaults=dict(zip(self.__call__.func_code.co_varnames[1:],self.__call__.func_defaults))
	useLocalDefaults = False

        for item in function_signature_defaults.iteritems():
                key,val = item
                keyVal = eval(key)
                if (keyVal == None):
                        #user hasn't set it - use global/default
                        pass
                else:
                        #user has set it - use over-ride
			if (key != 'self') :
			   useLocalDefaults = True

	myparams = {}
	if useLocalDefaults :
	   for item in function_signature_defaults.iteritems():
	       key,val = item
	       keyVal = eval(key)
	       exec('myparams[key] = keyVal')
	       self.parameters[key] = keyVal
	       if (keyVal == None):
	           exec('myparams[key] = '+ key + ' = self.itsdefault(key)')
		   keyVal = eval(key)
		   if(type(keyVal) == dict) :
                      if len(keyVal) > 0 :
		         exec('myparams[key] = ' + key + ' = keyVal[len(keyVal)-1][\'value\']')
		      else :
		         exec('myparams[key] = ' + key + ' = {}')

        else :
            print ''

            myparams['mirfile'] = mirfile = self.parameters['mirfile']
            myparams['vis'] = vis = self.parameters['vis']
            myparams['tsys'] = tsys = self.parameters['tsys']
            myparams['spw'] = spw = self.parameters['spw']
            myparams['vel'] = vel = self.parameters['vel']
            myparams['linecal'] = linecal = self.parameters['linecal']
            myparams['wide'] = wide = self.parameters['wide']
            myparams['debug'] = debug = self.parameters['debug']


	result = None

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
	pathname='file:///'+os.environ.get('MUSERHOME')+'/resource/xml/'
        #trec = muserc.muserc.utils().torecord(pathname+'importmiriad.xml')

        muserlog.origin('importmuser')
	try :
          #if not trec.has_key('importmiriad') or not muserc.muserc.utils().verify(mytmp, trec['importmiriad']) :
	        #return False

          #muserc.muserc.utils().verify(mytmp, trec['importmiriad'], True)
          scriptstr=['']
          saveinputs = self.__globals__['saveinputs']
          if type(self.__call__.func_defaults) is NoneType:
              saveinputs=''
          else:
              saveinputs('importmuser', 'importmuser.last', myparams, self.__globals__,scriptstr=scriptstr)
          tname = 'importmiriad'
          spaces = ' '*(18-len(tname))
          muserlog.post('\n##########################################'+
                       '\n##### Begin Task: ' + tname + spaces + ' #####')
          if type(self.__call__.func_defaults) is NoneType:
              muserlog.post(scriptstr[0]+'\n', 'INFO')
          else :
              muserlog.post(scriptstr[1][1:]+'\n', 'INFO')
          result = importmuser(mirfile, vis, tsys, spw, vel, linecal, wide, debug)
          muserlog.post('##### End Task: ' + tname + '  ' + spaces + ' #####'+
                       '\n##########################################')

	except Exception, instance:
          if(self.__globals__.has_key('__rethrow_muser_exceptions') and self.__globals__['__rethrow_muser_exceptions']) :
             raise
          else :
             #print '**** Error **** ',instance
	     tname = 'importmiriad'
             muserlog.post('An error occurred running task '+tname+'.', 'ERROR')
             pass

        gc.collect()
        return result
#
#
#
    def paramgui(self, useGlobals=True, ipython_globals=None):
        """
        Opens a parameter GUI for this task.  If useGlobals is true, then any relevant global parameter settings are used.
        """
        import paramgui
	if not hasattr(self, "__globals__") or self.__globals__ == None :
           self.__globals__=sys._getframe(len(inspect.stack())-1).f_globals

        if useGlobals:
	    if ipython_globals == None:
                myf=self.__globals__
            else:
                myf=ipython_globals

            paramgui.setGlobals(myf)
        else:
            paramgui.setGlobals({})

        paramgui.runTask('importmuser', myf['_ip'])
        paramgui.setGlobals({})

#
#
#
    def defaults(self, param=None, ipython_globals=None, paramvalue=None, subparam=None):
	if not hasattr(self, "__globals__") or self.__globals__ == None :
           self.__globals__=sys._getframe(len(inspect.stack())-1).f_globals
        if ipython_globals == None:
            myf=self.__globals__
        else:
            myf=ipython_globals

        a = odict()
        a['mirfile']  = ''
        a['vis']  = ''
        a['tsys']  = False
        a['spw']  = 'all'
        a['vel']  = ''
        a['linecal']  = False
        a['wide']  = 'all'
        a['debug']  = 0


### This function sets the default values but also will return the list of
### parameters or the default value of a given parameter
        if(param == None):
                myf['__set_default_parameters'](a)
        elif(param == 'paramkeys'):
                return a.keys()
        else:
            if(paramvalue==None and subparam==None):
               if(a.has_key(param)):
                  return a[param]
               else:
                  return self.itsdefault(param)
            else:
               retval=a[param]
               if(type(a[param])==dict):
                  for k in range(len(a[param])):
                     valornotval='value'
                     if(a[param][k].has_key('notvalue')):
                        valornotval='notvalue'
                     if((a[param][k][valornotval])==paramvalue):
                        retval=a[param][k].copy()
                        retval.pop(valornotval)
                        if(subparam != None):
                           if(retval.has_key(subparam)):
                              retval=retval[subparam]
                           else:
                              retval=self.itsdefault(subparam)
		     else:
                        retval=self.itsdefault(subparam)
               return retval


#
#
    def check_params(self, param=None, value=None, ipython_globals=None):
      if ipython_globals == None:
          myf=self.__globals__
      else:
          myf=ipython_globals
#      print 'param:', param, 'value:', value
      try :
         if str(type(value)) != "<type 'instance'>" :
            value0 = value
            value = myf['cu'].expandparam(param, value)
            matchtype = False
            if(type(value) == numpy.ndarray):
               if(type(value) == type(value0)):
                  myf[param] = value.tolist()
               else:
                  #print 'value:', value, 'value0:', value0
                  #print 'type(value):', type(value), 'type(value0):', type(value0)
                  myf[param] = value0
                  if type(value0) != list :
                     matchtype = True
            else :
               myf[param] = value
            value = myf['cu'].verifyparam({param:value})
            if matchtype:
               value = False
      except Exception, instance:
         #ignore the exception and just return it unchecked
         myf[param] = value
      return value
#
#
    def description(self, key='importmuser', subkey=None):
        desc={'importmuser': 'Convert a Miriad visibility file into a muser MeasurementSet',
               'mirfile': 'Name of input Miriad visibility file',
               'vis': 'Name of output MeasurementSet',
               'tsys': 'Use the Tsys to set the visibility weights',
               'spw': 'Select spectral windows',
               'vel': 'Select velocity reference (TOPO,LSRK,LSRD)',
               'linecal': '(CARMA) Apply line calibration',
               'wide': '(CARMA) Select wide window averages',
               'debug': 'Display increasingly verbose debug messages',

              }

#
# Set subfields defaults if needed
#

        if(desc.has_key(key)) :
           return desc[key]

    def itsdefault(self, paramname) :
        a = {}
        a['mirfile']  = ''
        a['vis']  = ''
        a['tsys']  = False
        a['spw']  = 'all'
        a['vel']  = ''
        a['linecal']  = False
        a['wide']  = 'all'
        a['debug']  = 0

        #a = sys._getframe(len(inspect.stack())-1).f_globals

        if a.has_key(paramname) :
	      return a[paramname]
importmuser_cli = importmuser_cli_()
