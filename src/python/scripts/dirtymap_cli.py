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
from task_dirty import dirtymap
import globalpy

class dirtymap_cli_:
    __name__ = "dirtymap"
    rkey = None
    i_am_a_muserpy_task = None
    # The existence of the i_am_a_muserpy_task attribute allows help()
    # (and other) to treat muserpy tasks as a special case.

    def __init__(self) :
       self.__bases__ = (dirtymap_cli_,)
       self.__doc__ = self.__call__.__doc__

       self.parameters={'subarray':None, 'polarization':None, 'frequency':None, 'visfile':None, 'uvfile':None, 'ra':None, 'dec':None, 'outdir':None, 'fits':None, 'plot':None, 'correct':None,'debug':None }


    def result(self, key=None):
	    #### and add any that have completed...
	    return None


    def __call__(self, subarray=None, polarization=None, frequency=None, visfile=None, uvfile=None, ra=None, dec=None, outdir=None, fits=None, plot=None, correct=None, debug=None):

        """Hogbom Clean Algorithm

	Detailed Description:

        Hogbom Clean Algorithm

	Arguments :

		inputfile:	Name of input Muser visibility file
		   Default Value:

		outdir:	Output directory
		   Default Value:

		fits:	Generate results with FITS format
		   Default Value: False

		plot:	Generate results with PNG image
		   Default Value: False

		correct:	Correct P Angle
		   Default Value:

		debug:	Display increasingly verbose debug messages
		   Default Value: 0

	Returns: void

	Example :


        clean(inputfile='ngc5921.uvfits')


        """
	if not hasattr(self, "__globals__") or self.__globals__ == None :
           self.__globals__=sys._getframe(len(inspect.stack())-1).f_globals
	#muserc = self.__globals__['muserc']
	muserlog = self.__globals__['muserlog']
	#muserlog = muserc.muserc.logsink()
        self.__globals__['__last_task'] = 'dirtymap'
        self.__globals__['taskname'] = 'dirtymap'
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
            myparams['subarray'] = subarray = self.parameters['subarray']
            myparams['polarization'] = polarization = self.parameters['polarization']
            myparams['frequency'] = frequency = self.parameters['frequency']
            myparams['visfile'] = visfile = self.parameters['visfile']
            myparams['uvfile'] = uvfile = self.parameters['uvfile']
            myparams['ra'] = ra = self.parameters['ra']
            myparams['dec'] = dec = self.parameters['dec']
            myparams['outdir'] = outdir = self.parameters['outdir']
            myparams['fits'] = fits = self.parameters['fits']
            myparams['plot'] = plot = self.parameters['plot']
            myparams['correct'] = correct = self.parameters['correct']
            myparams['debug'] = debug = self.parameters['debug']


	result = None

#
#    The following is work around to avoid a bug with current python translation
#
        mytmp = {}
        mytmp['subarray']=subarray
        mytmp['polarization']= polarization
        mytmp['frequency']= frequency
        mytmp['visfile'] = visfile
        mytmp['uvfile'] = uvfile
        mytmp['ra'] = ra
        mytmp['dec'] = dec
        mytmp['outdir'] = outdir
        mytmp['fits'] = fits
        mytmp['plot'] = plot
        mytmp['correct'] = correct
        mytmp['debug'] = debug
	pathname='file:///'+os.environ.get('MUSERHOME')+'/resource/xml/'
        #trec = muserc.muserc.utils().torecord(pathname+'importmiriad.xml')

        muserlog.origin('dirtymap')
	try :
          #if not trec.has_key('importmiriad') or not muserc.muserc.utils().verify(mytmp, trec['importmiriad']) :
	    #return False
          #muserc.muserc.utils().verify(mytmp, trec['importmiriad'], True)
          scriptstr=['']
          saveinputs = self.__globals__['saveinputs']
          # if type(self.__call__.func_defaults) is NoneType:
          #     saveinputs=''
          # else:
          #     saveinputs('clean', 'clean.last', myparams, self.__globals__,scriptstr=scriptstr)
          tname = 'dirtymap'
          spaces = ' '*(18-len(tname))
          # muserlog.post('\n##########################################'+
          #              '\n##### Begin Task: ' + tname + spaces + ' #####')
          # if type(self.__call__.func_defaults) is NoneType:
          #     muserlog.post(scriptstr[0]+'\n', 'INFO')
          # else :
          #     muserlog.post(scriptstr[1][1:]+'\n', 'INFO')

          result = dirtymap(subarray, polarization, frequency, visfile, uvfile, ra, dec, outdir, plot, fits, correct, debug)
          # muserlog.post('##### End Task: ' + tname + '  ' + spaces + ' #####'+
          #              '\n##########################################')

	except Exception, instance:
          if(self.__globals__.has_key('__rethrow_muser_exceptions') and self.__globals__['__rethrow_muser_exceptions']) :
             raise
          else :
             print '**** Error **** ',instance
	     tname = 'clean'
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

        paramgui.runTask('dirtymap', myf['_ip'])
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
        a['subarray']= 1
        a['polarization']= 0
        a['frequency']= 400000000
        a['visfile'] = ''
        a['uvfile'] = ''
        a['ra'] = ''
        a['dec'] = ''
        a['outdir']  = ''
        a['fits']  = False
        a['plot']  = True
        a['correct']  = False
        a['debug']  = globalpy.muser_global_debug


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
    def description(self, key='dirtymap', subkey=None):
        desc={'dirtymap': 'Dirty Map',

               'outdir': 'Output file directory',
               'fits': 'Generate results with FITS format',
               'plot': 'Generate results with PNG format',
               'correct': 'Correct p angle',
               'debug': 'Display increasingly verbose debug messages',

              }

#
# Set subfields defaults if needed
#

        if(desc.has_key(key)) :
           return desc[key]

    def itsdefault(self, paramname) :
        a = {}
        a['subarray']= 1
        a['polarization']= 0
        a['frequency']= 400000000
        a['visfile'] = ''
        a['uvfile'] = ''
        a['ra'] = ''
        a['dec'] = ''
        a['inputfile']  = ''
        a['outdir']  = ''
        a['fits']  = False
        a['plot']  = True
        a['correct']  = False
        a['debug']  = globalpy.muser_global_debug

        #a = sys._getframe(len(inspect.stack())-1).f_globals

        if a.has_key(paramname) :
	      return a[paramname]

dirtymap_cli = dirtymap_cli_()
