import os
if os.environ.has_key('LD_PRELOAD'):
    del os.environ['LD_PRELOAD']
import sys
import time
import signal


homedir = os.getenv('MUSERHOME')
if homedir == None :
   print "Environment variable HOME is not set, please set it"
   sys.exit(1)
home=os.environ['MUSERHOME']


try:
   import IPython
except ImportError, e:
   print 'Failed to load IPython: ', e
   exit(1)
   
   
try:
    import matplotlib
except ImportError, e:
    print "failed to load matplotlib:\n", e
    print "sys.path =", "\n\t".join(sys.path)


try:
    import muserc
except ImportError, e:
    print "failed to load muser:\n", e
    sys.exit(1)
    
import muserdef
muser = { 'build': {
             'time': muserdef.build_time,
             'version': muserdef.muser_version,
             'number': muserdef.subversion_revision
         },
         'source': {
             'url': muserdef.subversion_url,
             'revision': muserdef.subversion_revision
         },
         'helpers': {
             'logger': 'muserlogger',
             'viewer': 'muserviewer',
             'info': None,
             'dbus': None,
             'ipcontroller': None,
             'ipengine': None
         },
         'dirs': {
             'rc': homedir + '/resource',
             'data': None,
             'recipes': muserdef.python_library_directory + '/recipes',
             'root': None
         },
         'flags': { },
         'files': { 
             'logfile': os.getcwd( ) + '/muserpy-'+time.strftime("%Y%m%d-%H%M%S", time.gmtime())+'.log'
         },
         'state' : { 'startup': True }
       }


# Set up muser root
# if os.environ.has_key('MUSERPATH') :
#     __muserpath__ = os.environ['MUSERPATH'].split(' ')[0]
#     if not os.path.exists(__muserpath__ + "/data") :
#         raise RuntimeError, "Unable to find the data repository directory in your muserPATH. Please fix."
#     else :
#         muser['dirs']['root'] = __muserpath__
#         muser['dirs']['data'] = __muserpath__ + "/data"
# else :
#     __muserpath__ = muserc.__file__
#     while __muserpath__ and __muserpath__ != "/" :
#         if os.path.exists( __muserpath__ + "/data") :
#             break
#         __muserpath__ = os.path.dirname(__muserpath__)
#     if not os.path.exists(__muserpath__ + "/data") :
#         raise RuntimeError, "muser path could not be determined"
#     else :
#         muser['dirs']['root'] = __muserpath__
#         muser['dirs']['data'] = __muserpath__ + "/data"


# Setup helper paths
if os.path.exists( __muserpath__ + "/bin/muserpyinfo") :
    muser['helpers']['info'] = __muserpath__ + "/bin/muserpyinfo"
    

if os.uname()[0]=='Darwin' :
    muser_path = os.environ['MUSERHOME'].split()

    muser['helpers']['viewer'] = muser_path[0]+'/'+muser_path[1]+'/apps/muserviewer.app/Contents/MacOS/muserviewer'
    # In the distro of the app then the apps dir is not there and you find things in MacOS
    if not os.path.exists(muser['helpers']['viewer']) :
        muser['helpers']['viewer'] = muser_path[0]+'/MacOS/muserviewer'

    if muser['flags'].has_key('--maclogger') :
        muser['helpers']['logger'] = 'console'
    else:
        muser['helpers']['logger'] = muser_path[0]+'/'+muser_path[1]+'/apps/muserlogger.app/Contents/MacOS/muserlogger'

        # In the distro of the app then the apps dir is not there and you find things in MacOS
        if not os.path.exists(muser['helpers']['logger']) :
            muser['helpers']['logger'] = muser_path[0]+'/Resources/Logger.app/Contents/MacOS/muserlogger'
            

## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
## ensure default initialization occurs before this point...
##
##      prelude.py  =>  setup/modification of muser settings
##      init.py     =>  user setup (with task access)
##
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
if os.path.exists( muser['dirs']['rc'] + '/prelude.py' ) :
    try:
        execfile ( muser['dirs']['rc'] + '/prelude.py' )
    except:
        print str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1])
        print 'Could not execute initialization file: ' + muser['dirs']['rc'] + '/prelude.py'
        sys.exit(1)


# Check IPYTHONDIR is defined by user and make it if not there
ipythonenv  = muser['dirs']['rc'] + '/ipython'
ipythonpath = muser['dirs']['rc'] + '/ipython'
# try :
#    os.makedirs(ipythonpath, 0755)
# except :
#    pass

# if(not os.environ.has_key('IPYTHONDIR')):
#     os.environ['IPYTHONDIR']=ipythonpath
# if(not os.path.exists(os.environ['IPYTHONDIR'])):
#     os.makedirs(os.environ['IPYTHONDIR'], 0755)


# os.environ['__muserRCDIR__']=muser['dirs']['rc']


# Special case if the back-end is set to MacOSX reset it
# to TkAgg as our TablePlot stuff is specific for TkAgg
if matplotlib.get_backend() == "MacOSX" :
   matplotlib.use('TkAgg')


# Check if the display environment is set if not
# switch the backend to Agg only if it's TkAgg
if not os.environ.has_key('DISPLAY') and matplotlib.get_backend() == "TkAgg" :
   matplotlib.use('Agg')


showconsole = False
deploylogger = False
thelogfile = 'null'


# Task Interface
from parameter_check import *


# CAS-951: matplotlib unresponsive on some 64bit systems
import platform
import pylab as pl
if (platform.architecture()[0]=='64bit'):
    if os.environ.has_key('DISPLAY') and os.environ['DISPLAY']!="" and not muser['flags'].has_key('--nogui'):
        pl.ioff( )
        pl.clf( )
        pl.ion( )


# Provide flexibility for boolean representation in the muser shell
true  = True
T     = True
false = False
F     = False

# Import tasks but don't load task manager and dbus
# os.environ['muser_ENGINE']="YES"

from tasks import *

# Setup available tasks
from math import *
from parameter_dictionary import *
from task_help import *


## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##
##      prelude.py  =>  setup/modification of muser settings (above)
##      init.py     =>  user setup (with task access)
##
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# if os.path.exists( muser['dirs']['rc'] + '/init.py' ) :
#     try:
#         execfile ( muser['dirs']['rc'] + '/init.py' )
#     except:
#         print str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1])
#         print 'Could not execute initialization file: ' + muser['dirs']['rc'] + '/init.py'
#         sys.exit(1)
#
#
# muser['state']['startup'] = False
