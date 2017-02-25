#
# This file was generated using xslt from its XML file
#
# Copyright 2008, Associated Universities Inc., Washington DC
#
import sys
import os
from casac import *
import string
import time
import inspect
import gc
import numpy
from odict import odict
from task_importmiriad import importmiriad
from task_importmiriad import casalog

class importmiriad_pg_:
    __name__ = "importmuser"

    def __init__(self) :
       self.__bases__ = (importmiriad_pg_,)
       self.__doc__ = self.__call__.__doc__


    def __call__(self, mirfile=None,vis=None,tsys=None,spw=None,vel=None,linecal=None,wide=None,debug=None,):

        a=inspect.stack()
        stacklevel=0
        for k in range(len(a)):
          if (string.find(a[k][1], 'ipython console') > 0) or (string.find(a[k][1], '<string>') >= 0):
                stacklevel=k
                break
        myf=sys._getframe(stacklevel).f_globals
        myf['__last_task'] = 'importmiriad'
        myf['taskname'] = 'importmiriad'
        ###
        myf['update_params'](func=myf['taskname'],printtext=False)
        ###
        ###
        #Handle globals or user over-ride of arguments
        #
	function_signature_defaults=dict(zip(self.__call__.func_code.co_varnames,self.__call__.func_defaults))
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
                        #myf[key]=keyVal

	myparams = {}
	if useLocalDefaults :
	   for item in function_signature_defaults.iteritems():
	       key,val = item
	       keyVal = eval(key)
	       exec('myparams[key] = keyVal')
	       if (keyVal == None):
	           exec('myparams[key] = '+ key + ' = self.itsdefault(key)')
		   keyVal = eval(key)
		   if(type(keyVal) == dict) :
		      exec('myparams[key] = ' + key + ' = keyVal[len(keyVal)-1][\'value\']')

	else :
            uselessvariable = None 
            myparams['mirfile'] = mirfile = myf['mirfile']
            myparams['vis'] = vis = myf['vis']
            myparams['tsys'] = tsys = myf['tsys']
            myparams['spw'] = spw = myf['spw']
            myparams['vel'] = vel = myf['vel']
            myparams['linecal'] = linecal = myf['linecal']
            myparams['wide'] = wide = myf['wide']
            myparams['debug'] = debug = myf['debug']


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
	pathname='file:///'+os.environ.get('CASAPATH').split()[0]+'/'+os.environ.get('CASAPATH').split()[1]+'/xml/'
        trec = casac.utils().torecord(pathname+'importmiriad.xml')

        casalog.origin('importmiriad')
        if not trec.has_key('importmiriad') or not casac.utils().verify(mytmp, trec['importmiriad']) :
	    return False


	try :
          casalog.post('')
          casalog.post('##########################################')
          casalog.post('##### Begin Task: importmiriad           #####')
          casalog.post('')
          result = importmiriad(mirfile, vis, tsys, spw, vel, linecal, wide, debug)
          casalog.post('')
          casalog.post('##### End Task: importmiriad           #####')
          casalog.post('##########################################')


# saveinputs for individule engine has no use
# saveinputs should alos be removed from casa_in_py.py
#
#
#          saveinputs = myf['saveinputs']
#          saveinputs('importmiriad', 'importmiriad.last', myparams)
#
#
	except Exception, instance:
	  if(myf.has_key('__rethrow_casa_exceptions') and myf['__rethrow_casa_exceptions']) :
	     raise
	  else:
	     tname = 'importmiriad'
	     casalog.post('An error occurred running task %s: %s' % (tname,instance), 'ERROR')
	     pass

        gc.collect()
        return result
#
#
##
#    def paramgui(self, useGlobals=True):
#        """
#        Opens a parameter GUI for this task.  If useGlobals is true, then any relevant global parameter settings are used.
#        """
#        import paramgui
#
#        a=inspect.stack()
#        stacklevel=0
#        for k in range(len(a)):
#          if (string.find(a[k][1], 'ipython console') > 0) or (string.find(a[k][1], '<string>') >= 0):
#            stacklevel=k
#            break
#        myf = sys._getframe(stacklevel).f_globals
#
#        if useGlobals:
#            paramgui.setGlobals(myf)
#        else:
#            paramgui.setGlobals({})
#
#        paramgui.runTask('importmiriad', myf['_ip'])
#        paramgui.setGlobals({})
#
#
#
#
    def defaults(self, param=None):
        a=inspect.stack()
        stacklevel=0
        for k in range(len(a)):
          if (string.find(a[k][1], 'ipython console') > 0) or (string.find(a[k][1], '<string>') >= 0):
                stacklevel=k
                break
        myf=sys._getframe(stacklevel).f_globals
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
	        if(a.has_key(param)):
		   #if(type(a[param]) == dict) :
		   #   return a[param][len(a[param])-1]['value']
	   	   #else :
		      return a[param]


#
#
    def check_params(self, param=None, value=None):
      a=inspect.stack() 
      stacklevel=0
      for k in range(len(a)):
        if (string.find(a[k][1], 'ipython console') > 0) or (string.find(a[k][1], '<string>') >= 0):
	    stacklevel=k
	    break
      myf=sys._getframe(stacklevel).f_globals

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

        if a.has_key(paramname) :
	      return a[paramname]
importmuser_pg = importmuser_pg_()
