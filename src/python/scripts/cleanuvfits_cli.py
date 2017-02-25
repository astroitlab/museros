#
# This file was generated using xslt from its XML file
#
# Copyright 2014, Associated Universities Inc., Washington DC
#
import sys
import os
import string
import time
import inspect
import gc
import numpy
from odict import odict
from types import *
from task_cleanuvfits import cleanuvfits
import globalpy

class cleanuvfits_cli_:
    __name__ = "cleanuvfits"
    rkey = None
    i_am_a_muserpy_task = None
    # The existence of the i_am_a_muserpy_task attribute allows help()
    # (and other) to treat muserpy tasks as a special case.

    def __init__(self) :
       self.__bases__ = (cleanuvfits_cli_,)
       self.__doc__ = self.__call__.__doc__

       self.parameters={'infile':None, 'is_loop_mode':None, 'mode':None, 'weight':None, 'band':None, 'channel':None, 'polarization':None, 'movera':None, 'movedec':None, 'P_ANGLE':None, 'fits':None, 'debug':None, 'outdir':None}
        #infile, is_loop_mode, weight, band, channel, polarization, movera, movedec, P_ANGLE, fits, debug

    def result(self, key=None):
        return None


    def __call__(self, infile=None, is_loop_mode=None, mode=None, weight=None, band=None, channel=None, polarization=None, movera=None, movedec=None, P_ANGLE=None, fits=None, debug=None, outdir=None):

        """Hogbom Clean Algorithm: Integration Clean

	Detailed Description:

        Hogbom Clean Algorithm From intergated FITS file(time_inteval, average, mixture)

	Arguments :

		inputfile:	Name of input Muser visibility file
		   Default Value:

		is_loop_mode: loop mode or not
		   Default Value: True

		P_ANGLE:	Correct P Angle
		   Default Value:

		debug:	Display increasingly verbose debug messages
		   Default Value: 0

	Returns: void

	Example :
        cleanuvfits(inputfile='ngc5921.uvfits' )


        """
	if not hasattr(self, "__globals__") or self.__globals__ == None :
           self.__globals__=sys._getframe(len(inspect.stack())-1).f_globals
	#muserc = self.__globals__['muserc']
	muserlog = self.__globals__['muserlog']
	#muserlog = muserc.muserc.logsink()
        self.__globals__['__last_task'] = 'cleanuvfits'
        self.__globals__['taskname'] = 'cleanuvfits'
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
            myparams['infile'] = infile = self.parameters['infile']
            myparams['is_loop_mode'] = is_loop_mode = self.parameters['is_loop_mode']
            myparams['mode'] = mode = self.parameters['mode']
            myparams['weight'] = weight = self.parameters['weight']
            myparams['band'] = band = self.parameters['band']
            myparams['channel'] = channel = self.parameters['channel']
            myparams['polarization'] = polarization = self.parameters['polarization']
            myparams['movera'] = movera = self.parameters['movera']
            myparams['movedec'] = movedec = self.parameters['movedec']
            myparams['P_ANGLE'] = P_ANGLE = self.parameters['P_ANGLE']
            myparams['fits'] = fits = self.parameters['fits']
            myparams['debug'] = debug = self.parameters['debug']
            myparams['outdir'] = outdir = self.parameters['outdir']


	result = None

#
#    The following is work around to avoid a bug with current python translation
#
        mytmp = {}
        myparams['infile'] = infile
        mytmp['is_loop_mode'] = is_loop_mode
        mytmp['mode'] = mode
        mytmp['weight'] = weight
        mytmp['band'] = band
        mytmp['channel'] = channel
        mytmp['polarization'] = polarization
        mytmp['movera'] = movera
        mytmp['movedec'] = movedec
        mytmp['P_ANGLE'] = P_ANGLE
        mytmp['fits'] = fits
        mytmp['debug'] = debug
        mytmp['outdir'] = outdir
	pathname='file:///'+os.environ.get('MUSERHOME')+'/resource/xml/'
        #trec = muserc.muserc.utils().torecord(pathname+'importmiriad.xml')

        muserlog.origin('cleanuvfits')
	try :
          #if not trec.has_key('importmiriad') or not muserc.muserc.utils().verify(mytmp, trec['importmiriad']) :
	    #return False
          #muserc.muserc.utils().verify(mytmp, trec['importmiriad'], True)
          scriptstr=['']
          saveinputs = self.__globals__['saveinputs']
          # if type(self.__call__.func_defaults) is NoneType:
          #     saveinputs=''
          # else:
          #     saveinputs('cleanuvfits', 'cleanuvfits.last', myparams, self.__globals__,scriptstr=scriptstr)
          tname = 'cleanuvfits'
          spaces = ' '*(18-len(tname))
          # muserlog.post('\n##########################################'+
          #              '\n##### Begin Task: ' + tname + spaces + ' #####')
          # if type(self.__call__.func_defaults) is NoneType:
          #     muserlog.post(scriptstr[0]+'\n', 'INFO')
          # else :
          #     muserlog.post(scriptstr[1][1:]+'\n', 'INFO')

          result = cleanuvfits(infile, is_loop_mode, mode, weight, band, channel, polarization, movera, movedec, P_ANGLE, fits, debug, outdir)

	except Exception, instance:
          if(self.__globals__.has_key('__rethrow_muser_exceptions') and self.__globals__['__rethrow_muser_exceptions']) :
             raise
          else :
             print '**** Error **** ',instance
	     tname = 'cleanuvfits'
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

        paramgui.runTask('cleanuvfits', myf['_ip'])
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
        a['infile'] = ''
        a['is_loop_mode'] = True
        a['mode'] = True
        a['band'] = None
        a['channel'] = 4
        a['weight'] = 0
        a['polarization']  = 0
        a['movera'] = 0
        a['movedec'] = 0
        a['P_ANGLE']  = False
        a['fits'] = 0
        a['debug']  = globalpy.muser_global_debug
        a['outdir'] = ''

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
    def description(self, key='cleanuvfits', subkey=None):
        desc={'cleanuvfits': ' CLEAN from Integrated FITS file',
              'infile': 'input FITS file',
              'is_loop_mode':'LOOP Mode or not',
              'mode': 'hybrid or regular',
              'weight': 'weighting mode: natural or uniform',
               'band': 'Band',
               'channel': 'Channel',
               'polarization': 'Generate results with FITS format',
               'movera': 'RA offset',
               'movedec':'DEC offset',
               'P_ANGLE': 'Correct p angle',
               'fits': 'writing FITS file of images or not',
               'debug': 'Display increasingly verbose debug messages',
               'outdir': 'output directory for images ',

              }

#
# Set subfields defaults if needed
#

        if(desc.has_key(key)) :
           return desc[key]

    def itsdefault(self, paramname) :
        a = {}
        a['infile'] = ''
        a['is_loop_mode'] = True
        a['mode'] = ''
        a['weight'] = None
        a['band'] = None
        a['channel'] = None
        a['polarization']  = 0
        a['movera'] = 0
        a['movedec'] = 0
        a['P_ANGLE']  = False
        a['fits'] = 0
        a['debug']  = globalpy.muser_global_debug
        a['outdit'] = ''

        if a.has_key(paramname):
            return a[paramname]

cleanuvfits_cli = cleanuvfits_cli_()
