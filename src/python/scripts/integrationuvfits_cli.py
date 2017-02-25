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
from task_integrationuvfits import integrationuvfits
import globalpy

class integrationuvfits_cli_:
    __name__ = "integrationuvfits"
    rkey = None
    i_am_a_muserpy_task = None
    # The existence of the i_am_a_muserpy_task attribute allows help()
    # (and other) to treat muserpy tasks as a special case.

    def __init__(self) :
       self.__bases__ = (integrationuvfits_cli_,)
       self.__doc__ = self.__call__.__doc__

       self.parameters={'subarray':None, 'is_loop_mode':None, 'start_time':None, 'end_time':None, 'task_type':None, 'time_average':None, 'time_interval':None, 'debug':None}
        #sub_ARRAY, MODE, start_time, end_time, TASK_TYPE, time_Interval, BAND, CHANNEL, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG, outdir


    def result(self, key=None):
	    #### and add any that have completed...
	    return None


    def __call__(self, subarray=None, is_loop_mode=None, start_time=None, end_time=None, task_type=None, time_average=None, time_interval=None, debug=None):

        """Hogbom Clean Algorithm: Integration Clean

	Detailed Description:

        Hogbom Clean Algorithm

	Arguments :

		inputfile:	Name of input Muser visibility file
		   Default Value:

                integration: Integrate or not
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


        """
	if not hasattr(self, "__globals__") or self.__globals__ == None :
           self.__globals__=sys._getframe(len(inspect.stack())-1).f_globals
	#muserc = self.__globals__['muserc']
	muserlog = self.__globals__['muserlog']
	#muserlog = muserc.muserc.logsink()
        self.__globals__['__last_task'] = 'integrationuvfits'
        self.__globals__['taskname'] = 'integrationuvfits'
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
            print ''  #sub_ARRAY, MODE, start_time, end_time, TASK_TYPE, time_Interval, BAND, CHANNEL, PLOT_ME, WRITE_FITS, P_ANGLE, DEBUG, outdir
            myparams['subarray'] = subarray = self.parameters['subarray']
            myparams['is_loop_mode'] = is_loop_mode = self.parameters['is_loop_mode']
            myparams['start_time'] = start_time = self.parameters['start_time']
            myparams['end_time'] = end_time = self.parameters['end_time']
            myparams['task_type'] = task_type = self.parameters['task_type']
            myparams['time_average'] = time_average = self.parameters['time_average']
            myparams['time_interval'] = time_interval = self.parameters['time_interval']
            myparams['debug'] = debug = self.parameters['debug']

	result = None

#
#    The following is work around to avoid a bug with current python translation
#
        mytmp = {}
        mytmp['subarray'] = subarray
        mytmp['is_loop_mode'] = is_loop_mode
        mytmp['start_time'] = start_time
        mytmp['end_time'] = end_time
        mytmp['task_type'] = task_type
        mytmp['time_average'] = time_average
        mytmp['time_interval'] = time_interval
        mytmp['debug'] = debug
	pathname='file:///'+os.environ.get('MUSERHOME')+'/resource/xml/'
        #trec = muserc.muserc.utils().torecord(pathname+'importmiriad.xml')

        muserlog.origin('integrationuvfits')
	try :
          saveinputs = self.__globals__['saveinputs']

          tname = 'integrationuvfits'
          spaces = ' '*(18-len(tname))
          # muserlog.post('\n##########################################'+
          #              '\n##### Begin Task: ' + tname + spaces + ' #####')
          # if type(self.__call__.func_defaults) is NoneType:
          #     muserlog.post(scriptstr[0]+'\n', 'INFO')
          # else :
          #     muserlog.post(scriptstr[1][1:]+'\n', 'INFO')

          result = integrationuvfits(subarray, is_loop_mode, start_time, end_time,  task_type, time_average, time_interval, debug)
          # muserlog.post('##### End Task: ' + tname + '  ' + spaces + ' #####'+
          #              '\n##########################################')

	except Exception, instance:
          if(self.__globals__.has_key('__rethrow_muser_exceptions') and self.__globals__['__rethrow_muser_exceptions']) :
             raise
          else :
             print '**** Error **** ',instance
	     tname = 'integrationuvfits'
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

        paramgui.runTask('integrationuvfits', myf['_ip'])
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
        a['subarray'] = 1
        a['is_loop_mode'] = True
        a['start_time'] = ''
        a['end_time'] = ''
        a['task_type'] = ''
        a['time_average'] = ''
        a['time_interval'] = None
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
    def description(self, key='integrationuvfits', subkey=None):
        desc={'integrationuvfits': 'Integration FITS file for MUSER',
              'subarray':'MUSER I or II (1, 2)',
              'start_time':'start time (start_time = "2016-07-05 12:04:38 463")',
              'end_time':'end time (end_time = "2016-07-05 12:30:00") ',
              'task_type': 'task type: average, interval or mixture',
              'time_average': 'time span of average computing',
              'time_interval': 'time_interval',
              'is_loop_mode':'LOOP Mode or not(True, False)',
              'debug': 'Display increasingly verbose debug messages (0, 1)',

              }

        if(desc.has_key(key)) :
           return desc[key]

    def itsdefault(self, paramname) :
        a = {}
        a['subarray'] = None
        a['start_time'] = ''
        a['end_time'] = ''
        a['task_type'] = ''
        a['time_average'] = ''
        a['time_interval'] = None
        a['is_loop_mode'] = True
        a['debug']  = globalpy.muser_global_debug

        if a.has_key(paramname) :
	      return a[paramname]

integrationuvfits_cli = integrationuvfits_cli_()
