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
from task_exportphase import exportphase
import globalpy


class exportphase_cli_:
    __name__ = "exportphase"
    rkey = None
    i_am_a_muserpy_task = None

    # The existence of the i_am_a_muserpy_task attribute allows help()
    # (and other) to treat muserpy tasks as a special case.

    def __init__(self):
        self.__bases__ = (exportphase_cli_,)
        self.__doc__ = self.__call__.__doc__

        self.parameters = {'muser': None, 'is_loop_mode':None,'inputfile': None, 'start': None, 'frame': None, 'calibration': None,
                           'debug': None,}

    def result(self, key=None):
        #### and add any that have completed...
        return None

    def __call__(self, muser=None, is_loop_mode=None, inputfile=None, start=None, frame=None, calibration=None, debug=None, ):

        """Convert a MUSER visibility file into a calibration source file

	Detailed Description:

        Convert a MUSER visibility file into a calibration source file

	Arguments :
	    muser:  Subarray ID of muser (1-low, 2-high)
	       Deafult Value:

		inputfile:	Name of input Muser visibility file
		   Default Value:

		start:	Start date and time
		   Default Value:

		frame:	The number of frames skipped
		   Default Value: 1

		calibration:	Calibration source
		   Default Value:

		debug:	Display increasingly verbose debug messages
		   Default Value: 0

	Returns: void

	Example :


        exportphase(muser=1, inputfile='ngc5921.uv')


        """
        if not hasattr(self, "__globals__") or self.__globals__ == None:
            self.__globals__ = sys._getframe(len(inspect.stack()) - 1).f_globals
        # muserc = self.__globals__['muserc']
        muserlog = self.__globals__['muserlog']
        self.__globals__['__last_task'] = 'exportphase'
        self.__globals__['taskname'] = 'exportphase'
        ###
        self.__globals__['update_params'](func=self.__globals__['taskname'], printtext=False,
                                          ipython_globals=self.__globals__)
        ###
        ###
        # Handle globals or user over-ride of arguments
        #
        if type(self.__call__.func_defaults) is NoneType:
            function_signature_defaults = {}
        else:
            function_signature_defaults = dict(
                zip(self.__call__.func_code.co_varnames[1:], self.__call__.func_defaults))
        useLocalDefaults = False

        for item in function_signature_defaults.iteritems():
            key, val = item
            keyVal = eval(key)
            if (keyVal == None):
                # user hasn't set it - use global/default
                pass
            else:
                # user has set it - use over-ride
                if (key != 'self'):
                    useLocalDefaults = True

        myparams = {}
        if useLocalDefaults:
            for item in function_signature_defaults.iteritems():
                key, val = item
                keyVal = eval(key)
                exec ('myparams[key] = keyVal')
                self.parameters[key] = keyVal
                if (keyVal == None):
                    exec ('myparams[key] = ' + key + ' = self.itsdefault(key)')
                    keyVal = eval(key)
                    if (type(keyVal) == dict):
                        if len(keyVal) > 0:
                            exec ('myparams[key] = ' + key + ' = keyVal[len(keyVal)-1][\'value\']')
                        else:
                            exec ('myparams[key] = ' + key + ' = {}')
        else:
            print ''
            myparams['muser'] = muser = self.parameters['muser']
            myparams['is_loop_mode'] = is_loop_mode = self.parameters['is_loop_mode']
            myparams['inputfile'] = inputfile = self.parameters['inputfile']
            myparams['start'] = start = self.parameters['start']
            myparams['frame'] = frame = self.parameters['frame']
            myparams['calibration'] = calibration = self.parameters['calibration']
            myparams['debug'] = debug = self.parameters['debug']

        result = None

        #
        #    The following is work around to avoid a bug with current python translation
        #
        mytmp = {}
        mytmp['muser'] = muser
        mytmp['is_loop_mode'] = is_loop_mode
        mytmp['inputfile'] = inputfile
        mytmp['start'] = start
        mytmp['frame'] = frame
        mytmp['calibration'] = calibration
        mytmp['debug'] = debug
        pathname = 'file:///' + os.environ.get('MUSERHOME') + '/resource/xml/'
        # trec = muserc.muserc.utils().torecord(pathname+'exportphase.xml')

        muserlog.origin('exportphase')
        try:
            # if not trec.has_key('importmiriad') or not muserc.muserc.utils().verify(mytmp, trec['importmiriad']) :
            # return False
            # muserc.muserc.utils().verify(mytmp, trec['importmiriad'], True)
            scriptstr = ['']
            saveinputs = self.__globals__['saveinputs']
            if type(self.__call__.func_defaults) is NoneType:
                saveinputs = ''
            else:
                saveinputs('exportphase', 'exportphase.last', myparams, self.__globals__, scriptstr=scriptstr)
            tname = 'exportphase'
            spaces = ' ' * (18 - len(tname))
            muserlog.post('\n##########################################' +
                          '\n##### Begin Task: ' + tname + spaces + ' #####')
            if type(self.__call__.func_defaults) is NoneType:
                muserlog.post(scriptstr[0] + '\n', 'INFO')
            else:
                muserlog.post(scriptstr[1][1:] + '\n', 'INFO')

            result = exportphase(muser, is_loop_mode, inputfile, start, frame, calibration, debug)
            muserlog.post('##### End Task: ' + tname + '  ' + spaces + ' #####' +
                          '\n##########################################')

        except Exception, instance:
            if (self.__globals__.has_key('__rethrow_muser_exceptions') and self.__globals__[
                '__rethrow_muser_exceptions']):
                raise
            else:
                print '**** Error **** ', instance
                tname = 'exportphase'
                muserlog.post('An error occurred running task ' + tname + '.', 'ERROR')
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
        if not hasattr(self, "__globals__") or self.__globals__ == None:
            self.__globals__ = sys._getframe(len(inspect.stack()) - 1).f_globals

        if useGlobals:
            if ipython_globals == None:
                myf = self.__globals__
            else:
                myf = ipython_globals

            paramgui.setGlobals(myf)
        else:
            paramgui.setGlobals({})

        paramgui.runTask('exportphase', myf['_ip'])
        paramgui.setGlobals({})

    #
    #
    #
    def defaults(self, param=None, ipython_globals=None, paramvalue=None, subparam=None):
        if not hasattr(self, "__globals__") or self.__globals__ == None:
            self.__globals__ = sys._getframe(len(inspect.stack()) - 1).f_globals
        if ipython_globals == None:
            myf = self.__globals__
        else:
            myf = ipython_globals

        a = odict()
        a['muser'] = globalpy.muser_global_muser
        a['is_loop_mode'] = True
        a['inputfile'] = ''
        a['start'] = ''
        a['frame'] = 1
        a['calibration'] = 0
        a['debug'] = globalpy.muser_global_debug

        ### This function sets the default values but also will return the list of
        ### parameters or the default value of a given parameter
        if (param == None):
            myf['__set_default_parameters'](a)
        elif (param == 'paramkeys'):
            return a.keys()
        else:
            if (paramvalue == None and subparam == None):
                if (a.has_key(param)):
                    return a[param]
                else:
                    return self.itsdefault(param)
            else:
                retval = a[param]
                if (type(a[param]) == dict):
                    for k in range(len(a[param])):
                        valornotval = 'value'
                        if (a[param][k].has_key('notvalue')):
                            valornotval = 'notvalue'
                        if ((a[param][k][valornotval]) == paramvalue):
                            retval = a[param][k].copy()
                            retval.pop(valornotval)
                            if (subparam != None):
                                if (retval.has_key(subparam)):
                                    retval = retval[subparam]
                                else:
                                    retval = self.itsdefault(subparam)
                        else:
                            retval = self.itsdefault(subparam)
                return retval


            #
            #

    def check_params(self, param=None, value=None, ipython_globals=None):
        if ipython_globals == None:
            myf = self.__globals__
        else:
            myf = ipython_globals
        #      print 'param:', param, 'value:', value
        try:
            if str(type(value)) != "<type 'instance'>":
                value0 = value
                value = myf['cu'].expandparam(param, value)
                matchtype = False
                if (type(value) == numpy.ndarray):
                    if (type(value) == type(value0)):
                        myf[param] = value.tolist()
                    else:
                        # print 'value:', value, 'value0:', value0
                        # print 'type(value):', type(value), 'type(value0):', type(value0)
                        myf[param] = value0
                        if type(value0) != list:
                            matchtype = True
                else:
                    myf[param] = value
                value = myf['cu'].verifyparam({param: value})
                if matchtype:
                    value = False
        except Exception, instance:
            # ignore the exception and just return it unchecked
            myf[param] = value
        return value

    #
    #
    def description(self, key='exportphase', subkey=None):
        desc = {'exportphase': 'Convert a MUSER satellite visibility file into a calibration file',
                'muser': 'Muser subarray 1 or 2',
                'is_loop_mode': 'Loop Mode or not',
                'inputfile': 'Name of input MUSER visibility file',
                'start': 'The start date and time of the observational data',
                'frame': 'The number skipped frames(unit = 1 frame)',
                'calibration': 'Calibration ID ',
                'debug': 'Display increasingly verbose debug messages',

                }

        #
        # Set subfields defaults if needed
        #

        if (desc.has_key(key)):
            return desc[key]

    def itsdefault(self, paramname):
        a = {}
        a['muser'] = globalpy.muser_global_muser
        a['is_loop_mode'] = True
        a['inputfile'] = ''
        a['start'] = ''
        a['frame'] = 1
        a['calibration'] = 0
        a['debug'] = globalpy.muser_global_debug

        # a = sys._getframe(len(inspect.stack())-1).f_globals

        if a.has_key(paramname):
            return a[paramname]


exportphase_cli = exportphase_cli_()
