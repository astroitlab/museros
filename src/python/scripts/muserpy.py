import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
if os.environ.has_key('LD_PRELOAD'):
    del os.environ['LD_PRELOAD']
if os.environ.has_key('MUSERHOME'):
    homedir = os.getenv('MUSERHOME')

pymodules_dir = homedir+'/python'
#print pymodules_dir
import sys
import time
import signal
import traceback
import IPython
from pymuser.muserenv import MuserEnv
from taskinit import muserlog
from pymuser.musersqlite import *

#sys.path = sys.path + path_addition

# i.e. /usr/lib/pymodules/python2.6, needed for matplotlib in Debian and its derivatives.
#pymodules_dir = sys.prefix + '/lib/pymodules/python' + '.'.join(map(str, sys.version_info[:2]))

if os.path.isdir(pymodules_dir) and pymodules_dir not in sys.path:
    sys.path.append(pymodules_dir)

##
## watchdog... which is *not* in the muserpy process group
##
if os.fork( ) == 0 :
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    ## close standard input to avoid terminal interrupts
    sys.stdin.close( )
    sys.stdout.close( )
    sys.stderr.close( )
    os.close(0)
    os.close(1)
    os.close(2)
    ppid = os.getppid( )
    while True :
        try:
            os.kill(ppid,0)
        except:
            break
        time.sleep(3)
    # jagonzal: Don't be gentle in a MPI environment in order not to block the mpirun command
    os.killpg(ppid, signal.SIGTERM)
    time.sleep(120)
    os.killpg(ppid, signal.SIGKILL)
    sys.exit(1)

##
## ensure that we're the process group leader
## of all processes that we fork...
##
try:
    os.setpgid(0,0)
except OSError, e:
    print "setgpid( ) failed: " + e.strerror
    print "                   processes may be left dangling..."
#

##
## no one likes a bloated watchdog...
## ...do this after setting up the watchdog
##
# try:
#     import muserc
# except ImportError, e:
#     print "failed to load muser:\n", e
#     sys.exit(1)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore");
        import matplotlib
except ImportError, e:
    print "failed to load matplotlib:\n", e
    print "sys.path =", "\n\t".join(sys.path)

homedir = os.getenv('MUSERHOME')
if homedir == None :
    print "Environment variable HOME is not set, please set it"
    sys.exit(1)

import muserdef

muserpy = { 'build': {
             'time': muserdef.build_time,
             'version': muserdef.muser_version,
             'number': muserdef.subversion_revision
         },
         'source': {
             'url': muserdef.subversion_url,
             'revision': muserdef.subversion_revision
         },
         'helpers': {
             'logger': 'muserlog',
             'viewer': None,
             'info': None,
             'dbus': None,
             'ipcontroller': None,
             'ipengine': None
         },
         'dirs': {
             'rc': homedir + '/resource',
             'data': homedir+'/data',
             'recipes': muserdef.python_library_directory + '/recipes',
             'root': None,
             'pipeline': None
         },
         'flags': { },
         'files': {
             'logfile': os.getcwd( ) + '/muserpy-'+time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.log'
         },
         'state' : {
             'startup': True,
             'unwritable': set( )
         }
       }

muser_path = os.getenv('MUSERPATH')
if muser_path == None :
   print "Environment variable PATH is not set, please set it"
   sys.exit(1)

__muserpath__ = muser_path


## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
## setup helper paths...
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
## try to set muserpyinfo path...
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
if os.path.exists( __muserpath__ + "/muserpyinfo") :
    muserpy['helpers']['info'] = __muserpath__ + "/muserpyinfo"

## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##     first try to find executables using muserpyinfo...
##            (since system area versions may be incompatible)...
##     next try likely system areas...
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##
##   note:  hosts which have dbus-daemon-1 but not dbus-daemon seem to have a broken dbus-daemon-1...
##
# for info in [ (['dbus-daemon'],'dbus'),
#               (['ipcontroller','ipcontroller-2.6'], 'ipcontroller'),
#               (['ipengine','ipengine-2.6'], 'ipengine') ]:
#     exelist = info[0]
#     entry = info[1]
#     for exe in exelist:
#         if muser['helpers']['info']:
#             muser['helpers'][entry] = (lambda fd: fd.readline().strip('\n'))(os.popen(muser['helpers']['info'] + " --exec 'which " + exe + "'"))
#         if muser['helpers'][entry] and os.path.exists(muser['helpers'][entry]):
#             break
#         else:
#             muser['helpers'][entry] = None
#
#         ### first look in known locations relative to top (of binary distros) or known muser developer areas
#         for srchdir in [ __muserpath__ + '/bin', __muserpath__ + '/lib/muser/bin', '/usr/lib64/muser/01/bin', '/opt/muser/01/bin' ] :
#             dd = srchdir + os.sep + exe
#             if os.path.exists(dd) and os.access(dd,os.X_OK) :
#                 muser['helpers'][entry] = dd
#                 break
#         if muser['helpers'][entry] is not None:
#             break
#
#     ## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#     ##     next search through $PATH for executables
#     ## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#     if muser['helpers'][entry] is None:
#         for exe in exelist:
#             for srchdir in os.getenv('PATH').split(':') :
#                 dd = srchdir + os.sep + exe
#                 if os.path.exists(dd) and os.access(dd,os.X_OK) :
#                     muser['helpers'][entry] = dd
#                     break
#             if muser['helpers'][entry] is not None:
#                 break
print\
"""
    __  _____  _______ __________
   /  |/  / / / / ___// ____/ __ \\
  / /|_/ / / / /\__ \/ __/ / /_/ /
 / /  / / /_/ /___/ / /___/ _, _/
/_/  /_/\____//____/_____/_/ |_|
"""

print "Version " + muserpy['build']['version'] + "-REL (r" + muserpy['source']['revision'] + ")\nCompiled on: " + muserpy['build']['time']

a = [] + sys.argv             ## get a copy from goofy python
a.reverse( )
##
## A session configuration 'muser.conf' now included in muser tree...
##
# dbus_conf = __muserpath__ + "/etc/dbus/muser.conf"
# if not os.path.exists(dbus_conf):
#     dbus_conf = __muserpath__ + "/Resources/dbus/muser.conf"

__ipython_colors = 'LightBG'
while len(a) > 0:
    c = a.pop()
    if c == '--colors':
        ##
        ## strip out 2 element ipython flags (which we recognize) here...
        ##
        if len(a) == 0 :
            print "A option must be specified with " + c + "..."
            sys.exit(1)
        else:
            c = a.pop( )
            if c != 'NoColor' and c != 'Linux' and c != 'LightBG':
                print "unrecognized option for '--color': " + c
                sys.exit(1)
            else:
                __ipython_colors = c

    elif c.startswith('--colors='):
        ##
        ## strip out single element ipython flags (which we recognize) here...
        ##
        c = c.split('=')[1]
        if c != 'NoColor' and c != 'Linux' and c != 'LightBG':
            print "unrecognized option for '--color': " + c
            sys.exit(1)
        else:
            __ipython_colors = c

    elif c == '--logfile' or c == '-c' or c == '--rcdir':
        ##
        ## we join multi-arg parameters here
        ##
        if len(a) == 0 :
            print "A file must be specified with " + c + "..."
            sys.exit(1)
        else :
            muserpy['flags'][c] = a.pop( )
            if c == '--rcdir':
                muserpy['dirs']['rc'] = muserpy['flags'][c]

    elif c.find('=') > 0 :
        muserpy['flags'][c[0:c.find('=')]] = c[c.find('=')+1:]

    else :
        muserpy['flags'][c] = ''



if muserpy['flags'].has_key('--logfile') :
    muserpy['files']['logfile'] = muserpy['flags']['--logfile']	## user specifies a log file
if muserpy['flags'].has_key('--nologfile') :
    muserpy['files'].pop('logfile')				## user indicates no log file

if muserpy['flags'].has_key('--help') :
	print "Options are: "
	print "   --rcdir directory"
	print "   --logfile logfilename"
	print "   --maclogger"
	print "   --log2term"
	print "   --nologger"
	print "   --nologfile"
	print "   --nogui"
        print "   --colors=[NoColor|Linux|LightBG]"
	print "   --noipython"
        print "   --pipeline"
	print "   -c filename-or-expression"
	print "   --help, print this text and exit"
	print
	sys.exit(0)

if os.uname()[0]=='Darwin' :
    muser_path = os.environ['MUSERPATH']

    # muser['helpers']['viewer'] = muser_path[0]+'/'+muser_path[1]+'/apps/muserviewer.app/Contents/MacOS/muserviewer'
    # # In the distro of the app then the apps dir is not there and you find things in MacOS
    # if not os.path.exists(muser['helpers']['viewer']) :
    #     muser['helpers']['viewer'] = muser_path[0]+'/MacOS/muserviewer'
    #
    # if muser['flags'].has_key('--maclogger') :
    muserpy['helpers']['logger'] = 'console'
    # else:
    #     muser['helpers']['logger'] = muser_path[0]+'/'+muser_path[1]+'/apps/muserlogger.app/Contents/MacOS/muserlogger'
    #
    #     # In the distro of the app then the apps dir is not there and you find things in MacOS
    #     if not os.path.exists(muser['helpers']['logger']) :
    #         muser['helpers']['logger'] = muser_path[0]+'/Resources/Logger.app/Contents/MacOS/muserlogger'


## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
## ensure default initialization occurs before this point...
##
##      prelude.py  =>  setup/modification of muser settings
##      init.py     =>  user setup (with task access)
##
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
if os.path.exists( muserpy['dirs']['rc'] + '/prelude.py' ) :
    try:
        execfile ( muserpy['dirs']['rc'] + '/prelude.py' )
    except:
        print str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1])
        print 'Could not execute initialization file: ' + muserpy['dirs']['rc'] + '/prelude.py'
        sys.exit(1)

## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
## on linux set up a dbus-daemon for muser because each
## x-server (e.g. Xvfb) gets its own dbus session...
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# if muserpy['helpers']['dbus'] is not None :
#
#     argv_0_path = os.path.dirname(os.path.abspath(sys.argv[0]))
#     dbus_path = os.path.dirname(os.path.abspath(muser['helpers']['dbus']))
#
#     (r,w) = os.pipe( )
#
#     if os.fork( ) == 0 :
#         os.close(r)
#         signal.signal(signal.SIGINT, signal.SIG_IGN)
#         signal.signal(signal.SIGHUP, signal.SIG_IGN)
#         ## close standard input to avoid terminal interrupts
#         sys.stdin.close( )
#         os.close(0)
#         args = [ 'muser-dbus-daemon' ]
#         args = args + ['--print-address', str(w)]
#         if dbus_conf is not None and os.path.exists(dbus_conf) :
#             args = args + ['--config-file',dbus_conf]
#         else:
#             args = args + ['--session']
#         os.execvp(muser['helpers']['dbus'],args)
#         sys.exit
#
#     os.close(w)
#     dbus_address = os.read(r,200)
#     dbus_address = dbus_address.strip( )
#     os.close(r)
#     if len(dbus_address) > 0 :
#         os.putenv('DBUS_SESSION_BUS_ADDRESS',dbus_address)
#         os.environ['DBUS_SESSION_BUS_ADDRESS'] = dbus_address


ipythonenv  = muserpy['dirs']['rc'] + '/ipython'
ipythonpath = muserpy['dirs']['rc'] + '/ipython'
# try :
#    os.makedirs(ipythonpath, 0755)
# except :
#    pass
###check IPYTHONDIR is defined by user and make it if not there
# if(not os.environ.has_key('IPYTHONDIR')):
#     os.environ['IPYTHONDIR']=ipythonpath
# if(not os.path.exists(os.environ['IPYTHONDIR'])):
#     os.makedirs(os.environ['IPYTHONDIR'], 0755)

# os.environ['__muserRCDIR__']=muser['dirs']['rc']

#import string

#
# Special case if the backend is set to MacOSX reset it to TkAgg as our TablePlot
# stuff is specific for TkAgg
#
if matplotlib.get_backend() == "MacOSX" :
   matplotlib.use('Agg')
   #matplotlib.use('TkAgg')

#
# Check if the display environment is set if not
# switch the backend to Agg only if it's TkAgg
#
if not os.environ.has_key('DISPLAY') and matplotlib.get_backend() == "TkAgg" :
   matplotlib.use('Agg')

#
# If the user has requested pipeline through a command line option set
# to use AGG
if muserpy['flags'].has_key('--pipeline'):
    matplotlib.use('Agg')

#
# We put in all the task declarations here...
#
from taskinit import *

logpid=[]

muserenv = MuserEnv()

showconsole = False

thelogfile = ''

showconsole = muserpy['flags'].has_key('--log2term')

if muserpy['files'].has_key('logfile') :
    thelogfile = muserpy['files']['logfile']
if muserpy['flags'].has_key('--nologfile') :
    thelogfile = 'null'

thelogfile = 'muser-'+time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.log'

muserlog.setLogger(muserenv.get_log_dir()+os.path.sep+thelogfile,  filelog=True, consolelog=True) #showconsole)

if not os.access('.', os.W_OK) :
    print
    print "********************************************************************************"
    print "Warning: no write permission in current directory, no log files will be written."
    print "********************************************************************************"
    deploylogger = False
    thelogfile = 'null'

if muserpy['flags'].has_key('--nologger') :
    deploylogger = False

if muserpy['flags'].has_key('--nogui') :
    deploylogger = False


from parameter_check import *
####################
def go(taskname=None):
    """ Execute taskname: """
    myf = sys._getframe(len(inspect.stack())-1).f_globals
    if taskname==None: taskname=myf['taskname']
    oldtaskname=taskname
    if(myf.has_key('taskname')):
        oldtaskname=myf['taskname']
    #myf['taskname']=taskname
    if type(taskname)!=str:
        taskname=taskname.__name__
        myf['taskname']=taskname
    try:
        parameter_checktype(['taskname'],[taskname],str)
    except TypeError, e:
        print "go -- TypeError: ",e
        return
    fulltaskname=taskname+'()'
    exec(fulltaskname)
    myf['taskname']=oldtaskname


def inp(taskname=None, page=False):
    """
    Function to browse input parameters of a given task
    taskname: name of task of interest
    page: use paging if True, useful if list of parameters is longer than terminal height
    """
    try:
        ####paging contributed by user Ramiro Hernandez
        if(page):
            #########################
            class TemporaryRedirect(object):
                def __init__(self, stdout=None, stderr=None):
                    self._stdout = stdout or sys.stdout
                    self._stderr = stderr or sys.stderr
                def __enter__(self):
                    self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
                    self.old_stdout.flush(); self.old_stderr.flush()
                    sys.stdout, sys.stderr = self._stdout, self._stderr
                def __exit__(self, exc_type, exc_value, traceback):
                    self._stdout.flush(); self._stderr.flush()
                    sys.stdout = self.old_stdout
                    sys.stderr = self.old_stderr
            #######################end class
            tempfile="__temp_input.muser"
            temporal = open(tempfile, 'w')

            with TemporaryRedirect(stdout=temporal):
                inp(taskname, False)
            temporal.close()
            os.system('more '+tempfile)
            os.system('rm '+tempfile)
            return
        ####
        myf=sys._getframe(len(inspect.stack())-1).f_globals
        if((taskname==None) and (not myf.has_key('taskname'))):
            print 'No task name defined for inputs display'
            return
        if taskname==None: taskname=myf['taskname']
        myf['taskname']=taskname
        if type(taskname)!=str:
            taskname=taskname.__name__
            myf['taskname']=taskname

        try:
            parameter_checktype(['taskname'],taskname,str)
        except TypeError, e:
            print "inp -- TypeError: ", e
            return
        except ValueError, e:
            print "inp -- OptionError: ", e
            return

        ###Check if task exists by checking if task_defaults is defined
        if ( not myf.has_key(taskname) and
             str(type(myf[taskname])) != "<type 'instance'>" and
             not hasattr(myf[taskname],"defaults") ):
            raise TypeError, "task %s is not defined " %taskname
        if(myf.has_key('__last_taskname')):
            myf['__last_taskname']=taskname
        else:
            myf.update({'__last_taskname':taskname})

        print '# ',myf['taskname']+' :: '+(eval(myf['taskname']+'.description()'))
        update_params(myf['taskname'], myf)
    except TypeError, e:
        print "inp --error: ", e
    except Exception, e:
        print "---",e

def update_params(func, printtext=True, ipython_globals=None):
    from odict import odict

    if ipython_globals == None:
        myf=sys._getframe(len(inspect.stack())-1).f_globals
    else:
        myf=ipython_globals

    ### set task to the one being called
    myf['taskname']=func
    obj=myf[func]

    if ( str(type(obj)) == "<type 'instance'>" and
         hasattr(obj,"check_params") ):
        hascheck = True
    else:
        hascheck = False

    noerror=True
    ###check if task has defined a task_check_params function

    if (hascheck):
        has_othertasks = myf.has_key('task_location')
        if(has_othertasks) :
           has_task = myf['task_location'].has_key(myf['taskname'])
           if (has_task) :
               pathname=myf['task_location'][myf['taskname']]
           else :
               pathname = os.environ.get('MUSERHOME')+'/resource/xml'

        else :
           pathname = os.environ.get('MUSERHOME')+'/resource/xml'
        xmlfile=pathname+'/'+myf['taskname']+'.xml'

    a=myf[myf['taskname']].defaults("paramkeys",myf)
    itsdef=myf[myf['taskname']].defaults
    itsparams=myf[myf['taskname']].parameters
    params=a
    #print 'itsparams:', itsparams
    for k in range(len(params)):
        paramval = obj.defaults(params[k], myf)

        notdict=True
        ###if a dictionary with key 0, 1 etc then need to peel-open
        ###parameters
        if(type(paramval)==dict):
            if(paramval.has_key(0)):
                notdict=False
        if(myf.has_key(params[k])):
            itsparams.update({params[k]:myf[params[k]]})
        else:
            itsparams.update({params[k]:obj.itsdefault(params[k])})
        if (notdict ):
            if(not myf.has_key(params[k])):
                myf.update({params[k]:paramval})
                itsparams.update({params[k]:paramval})
            if(printtext):
                #print 'params:', params[k], '; myf[params]:', myf[params[k]]
                if(hascheck):
                    noerror = obj.check_params(params[k],myf[params[k]],myf)
                # RI this doesn't work with numpy arrays anymore.  Noone seems
                # interested, so I'll be the red hen and try to fix it.

                #print 'params:', params[k], '; noerror:', noerror, '; myf[params]:', myf[params[k]]
                myfparamsk=myf[params[k]]
                if(type(myf[params[k]])==pl.ndarray):
                    myfparamsk=myfparamsk.tolist()
                #if(myf[params[k]]==paramval):
                if(myfparamsk==paramval):
                    print_params_col(params[k],myf[params[k]],obj.description(params[k]), 'ndpdef', 'black',noerror)
                else:
                    print_params_col(params[k],myf[params[k]],obj.description(params[k]), 'ndpnondef', 'black', noerror)
                itsparams[params[k]] = myf[params[k]]
        else:
            subdict=odict(paramval)
            ##printtext is False....called most probably to set
            ##undefined params..no harm in doing it anyways
            if(not printtext):
                ##locate which dictionary is user selected
                userdict={}
                subkeyupdated={}
                for somekey in paramval:
                    somedict=dict(paramval[somekey])
                    subkeyupdated.update(dict.fromkeys(somedict, False))
                    if(somedict.has_key('value') and myf.has_key(params[k])):
                        if(somedict['value']==myf[params[k]]):
                            userdict=somedict
                    elif(somedict.has_key('notvalue') and myf.has_key(params[k])):
                        if(somedict['notvalue']!=myf[params[k]]):
                            userdict=somedict
                ###The behaviour is to use the task.defaults
                ### for all non set parameters and parameters that
                ### have no meaning for this selection
                for j in range(len(subdict)):
                    subkey=subdict[j].keys()

                    for kk in range(len(subkey)):

                        if( (subkey[kk] != 'value') & (subkey[kk] != 'notvalue') ):
                            #if user selecteddict
                            #does not have the key
                            ##put default
                            if(userdict.has_key(subkey[kk])):
                                if(myf.has_key(subkey[kk])):
                                    itsparams.update({subkey[kk]:myf[subkey[kk]]})
                                else:
                                    itsparams.update({subkey[kk]:userdict[subkey[kk]]})
                                subkeyupdated[subkey[kk]]=True
                            elif((not subkeyupdated[subkey[kk]])):
                                itsparams.update({subkey[kk]:itsdef(params[k], None, itsparams[params[k]], subkey[kk])})
                                subkeyupdated[subkey[kk]]=True
            ### need to do default when user has not set val
            if(not myf.has_key(params[k])):
                if(paramval[0].has_key('notvalue')):
                    itsparams.update({params[k]:paramval[0]['notvalue']})
                    myf.update({params[k]:paramval[0]['notvalue']})
                else:
                    itsparams.update({params[k]:paramval[0]['value']})
                    myf.update({params[k]:paramval[0]['value']})
            userval=myf[params[k]]
            choice=0
            notchoice=-1
            valuekey='value'
            for j in range(len(subdict)):
                if(subdict[j].has_key('notvalue')):
                    valuekey='notvalue'
                    if(subdict[j]['notvalue'] != userval):
                        notchoice=j
                        break
                else:
                    if(subdict[j]['value']==userval):
                        choice=j
                        notchoice=j
                        break
            subkey=subdict[choice].keys()
            if(hascheck):
                noerror=obj.check_params(params[k],userval,myf)
            if(printtext):
                if(myf[params[k]]==paramval[0][valuekey]):
                    print_params_col(params[k],myf[params[k]],obj.description(params[k]),'dpdef','black', noerror)
                else:
                    print_params_col(params[k],myf[params[k]],obj.description(params[k]),'dpnondef','black', noerror)
                itsparams[params[k]] = myf[params[k]]
            for j in range(len(subkey)):
                if((subkey[j] != valuekey) & (notchoice > -1)):
                    ###put default if not there
                    if(not myf.has_key(subkey[j])):
                        myf.update({subkey[j]:subdict[choice][subkey[j]]})
                    paramval=subdict[choice][subkey[j]]
                    if (j==(len(subkey)-1)):
                        # last subparameter - need to add an extra line to allow cut/pasting
                        comment='last'
                    else:
                        comment='blue'
                    if(hascheck):
                        noerror = obj.check_params(subkey[j],myf[subkey[j]],myf)
                    if(printtext):
                        if(myf[subkey[j]]==paramval):
                            print_params_col(subkey[j],myf[subkey[j]],obj.description(subkey[j],userval),'spdef',comment, noerror)
                        else:
                            print_params_col(subkey[j],myf[subkey[j]],obj.description(subkey[j],userval),'spnondef',comment, noerror)
                        itsparams[params[k]] = myf[params[k]]
    #
    # Verify the complete record, with errors being reported to the user
    #
    #cu.verify(itsparams, cu.torecord('file://'+xmlfile)[myf['taskname']]);

####function to print inputs with coloring
####colorparam 'blue'=> non-default, colorcomment 'green'=> can have sub params
#### 'blue' => is a sub-parameter 
# blue = \x1B[94m
# bold = \x1B[1m
# red  = \x1B[91m
# cyan = \x1B[96m
# green= \x1B[92m
# normal   = \x1B[0m
# underline= \x1B[04m
# reverse = \x1B[7m
# highlight with black = \x1B[40s

def print_params_col(param=None, value=None, comment='', colorparam=None,
                     colorcomment=None, noerrorval=True):
    try:
        from TerminalController import TerminalController
        term = TerminalController()
        cols = term.COLS
        del term
    except:
        cols = 80
    #
    #print 'colorparam is: ', colorparam
    #
    if type(value) == str:
        printval = "'" + value + "'"
    else:
        printval = value

    if colorparam == 'ndpnondef':
        firstcol = '\x1B[0m'
        valcol   = '\x1B[94m'
    elif colorparam == 'dpdef':
        firstcol = '\x1B[1m' + '\x1B[47m'
        valcol   = '\x1B[1m' + '\x1B[0m'
    elif colorparam == 'dpnondef':
        firstcol = '\x1B[1m' + '\x1B[47m'
        valcol   = '\x1B[1m' + '\x1B[94m'
    elif colorparam == 'spdef':
        firstcol = '\x1B[32m'
        valcol   = '\x1B[0m'
    elif colorparam == 'spnondef':
        firstcol = '\x1B[32m'
        valcol   = '\x1B[94m'
    else:
        firstcol = '\x1B[0m'
        valcol   = '\x1B[0m'

    if not noerrorval:
        valcol = '\x1B[1m' + '\x1B[91m'

    if colorcomment == 'green':
        secondcol = '\x1B[102m'
    elif colorcomment == 'blue':
        #secondcol='\x1B[104m'
        secondcol = '\x1B[0m'
    else:
        secondcol = '\x1B[0m'

    # RR: I think colorcomment should really be called submenu.
    #     Since these are left justified, I've absorbed the right space into
    #     the %s's, in order to handle as long a parameter name as possible.
    #     (The uvfilterb* params were busting out of %-10s.)
    if colorcomment in ('last', 'blue'):
        parampart = firstcol + '     %-14s ='
    else:
        parampart = firstcol + '%-19s ='
    parampart %= param

    valpart = valcol + ' %10s \x1B[0m' % printval + secondcol
    # Well behaved (short) parameters and values tally up to 33 characters
    # so far.  Pad them up to 40, assuming the param is short enough.
    pad = 7
    paramlen = len(str(param))
    if colorcomment in ('last', 'blue') and paramlen > 14:
        pad -= paramlen - 14
    elif paramlen > 19:
        pad -= paramlen - 19
    valuelen = len(str(printval))
    if valuelen > 10:
        pad -= valuelen - 10
    if pad > 0:
        valpart += ' ' * pad

    try:
        from textwrap import fill
        if pad < 0:
            firstskip = 40 - pad
            firstfiller = ' ' * firstskip + '#  '
            afterfiller = ' ' * 40 + '#   '
        else:
            firstskip = 40
            firstfiller = ' ' * 40 + '#  '
            afterfiller = firstfiller + ' '
        commentpart = fill(comment, cols, initial_indent=firstfiller,
                           subsequent_indent=afterfiller)[firstskip:]
    except:
        if comment:
            commentpart = '#  ' + comment
        else:
            commentpart = ''
    commentpart += '\x1B[0m'          # RR: I think this might be redundant.
    if colorcomment == 'last':        #     (Is colorcomment ever green?)
        commentpart += "\n"

    print parampart + valpart + commentpart

def __set_default_parameters(b):
    myf=sys._getframe(len(inspect.stack())-1).f_globals
    a=b
    elkey=a.keys()
    for k in range(len(a)):
        if (type(a[elkey[k]]) != dict):
            myf[elkey[k]]=a[elkey[k]]
        elif (type(a[elkey[k]]) == dict and len(a[elkey[k]])==0):
            myf[elkey[k]]=a[elkey[k]]
        else:
            subdict=a[elkey[k]]
            ##clear out variables of other options if they exist
            for j in range(1,len(subdict)):
                subkey=subdict[j].keys()
                for kk in range(len(subkey)):
                    if((subkey[kk] != 'value') & (subkey[kk] != 'notvalue') ):
                        if(myf.has_key(subkey[kk])):
                            del myf[subkey[kk]]
            ###
            if(subdict[0].has_key('notvalue')):
                myf[elkey[k]]=subdict[0]['notvalue']
            else:
                myf[elkey[k]]=subdict[0]['value']
            subkey=subdict[0].keys()
            for j in range(0, len(subkey)):
                if((subkey[j] != 'value') & (subkey[j] != 'notvalue')):
                    myf[subkey[j]]=subdict[0][subkey[j]]

def tput(taskname=None, outfile=''):
    myf = sys._getframe(len(inspect.stack())-1).f_globals
    if taskname == None: taskname = myf['taskname']
    if type(taskname) != str:
        taskname=taskname.__name__
    myf['taskname'] = taskname
    outfile = myf['taskname']+'.last'
    saveinputs(taskname, outfile)

def saveinputs(taskname=None, outfile='', myparams=None, ipython_globals=None, scriptstr=['']):
    #parameter_printvalues(arg_names,arg_values,arg_types)
    """ Save current input values to file on disk for a specified task:

    taskname -- Name of task
        default: <unset>; example: taskname='bandpass'
        <Options: type tasklist() for the complete list>
    outfile -- Output file for the task inputs
        default: taskname.saved; example: outfile=taskname.orion

    """
    try:

        if ipython_globals == None:
            myf = sys._getframe(len(inspect.stack())-1).f_globals
        else:
            myf=ipython_globals

        if taskname==None: taskname=myf['taskname']
        myf['taskname']=taskname
        if type(taskname)!=str:
            taskname=taskname.__name__
            myf['taskname']=taskname
        parameter_checktype(['taskname','outfile'],[taskname,outfile],[str,str])

        ###Check if task exists by checking if task_defaults is defined
        obj = False
        if ( not myf.has_key(taskname) and
             str(type(myf[taskname])) != "<type 'instance'>" and
             not hasattr(myf[taskname],"defaults") ):
            raise TypeError, "task %s is not defined " %taskname
        else:
            obj = myf[taskname]

        if taskname==None: taskname=myf['taskname']
        myf['taskname']=taskname
        if outfile=='': outfile=taskname+'.saved'
        ##make sure unfolded parameters get their default values
        myf['update_params'](func=myf['taskname'], printtext=False, ipython_globals=myf)
        ###

        do_save_inputs = False
        outpathdir = os.path.realpath(os.path.dirname(outfile))
        outpathfile = outpathdir + os.path.sep + os.path.basename(outfile)
        #if outpathfile not in muser['state']['unwritable'] and outpathdir not in muser['state']['unwritable']:
        try:
            taskparameterfile=open(outfile,'w')
            print >>taskparameterfile, '%-15s    = "%s"'%('taskname', taskname)
            do_save_inputs = True
        except:
            print "********************************************************************************"
            print "Warning: no write permission for %s, cannot save task" % outfile
            if os.path.isfile(outfile):
                print "         inputs in %s..." % outfile
                muserpy['state']['unwritable'].add(outpathfile)
            elif not os.path.isdir(outfile):
                print "         inputs in dir %s..." % outpathdir
                muserpy['state']['unwritable'].add(outpathdir)
            else:
                print "         inputs because given file (%s) is a dir..." % outpathfile
            print "********************************************************************************"

        #retrieve the parameters list (arguments list)
        f=zip(myf[taskname].__call__.func_code.co_varnames[1:],myf[taskname].__call__.func_defaults)

        scriptstring='#'+str(taskname)+'('
        if myparams == None :
            myparams = {}
        l=0
        for j in range(len(f)):
            k=f[j][0]
            if not myparams.has_key(k) and k != 'self' :
                myparams[k] = myf[taskname].parameters[k]
            if(k != 'self' and type(myparams[k])==str):
                if ( myparams[k].count( '"' ) < 1 ):
                    # if the string doesn't contain double quotes then
                    # use double quotes around it in the parameter file.
                    if do_save_inputs:
                        print >>taskparameterfile, '%-15s    =  "%s"'%(k, myparams[k])
                    scriptstring=scriptstring+k+'="'+myparams[k]+'",'
                else:
                    # use single quotes.
                    if do_save_inputs:
                        print >>taskparameterfile, "%-15s    =  '%s'"%(k, myparams[k])
                    scriptstring=scriptstring+k+"='"+myparams[k]+"',"
            else :
                if ( j != 0 or k != "self" or
                     str(type(myf[taskname])) != "<type 'instance'>" ) :
                    if do_save_inputs:
                        print >>taskparameterfile, '%-15s    =  %s'%(k, myparams[k])
                    scriptstring=scriptstring+k+'='+str(myparams[k])+','

            ###Now delete varianle from global user space because
            ###the following applies: "It would be nice if one
            ### could tell the system to NOT recall
            ### previous non-default settings sometimes."
            if(not myf['muserglobals'] and myf.has_key(k)):
                del myf[k]
            l=l+1
            # if l%5==0:
            #     scriptstring=scriptstring+'\n        '
        scriptstring=scriptstring.rstrip()
        scriptstring=scriptstring.rstrip('\n')
        scriptstring=scriptstring.rstrip(',')
        scriptstring=scriptstring+')'
        scriptstr.append(scriptstring)
        scriptstring=scriptstring.replace('        ', '')
        scriptstring=scriptstring.replace('\n', '')
        if do_save_inputs:
            print >>taskparameterfile,scriptstring
            taskparameterfile.close()
    except TypeError, e:
        #print traceback.print_exc()
        print "saveinputs --error: ", e

def default(taskname=None):
    """ reset given task to its default values :

    taskname -- Name of task


    """

    try:
        myf = sys._getframe(len(inspect.stack())-1).f_globals
        if taskname==None: taskname=myf['taskname']
        myf['taskname']=taskname
        if type(taskname)!=str:
            taskname=taskname.__name__
            myf['taskname']=taskname

        ###Check if task exists by checking if task_defaults is defined
        if ( not myf.has_key(taskname) and
             str(type(myf[taskname])) != "<type 'instance'>" and
             not hasattr(myf[taskname],"defaults") ):
            raise TypeError, "task %s is not defined " %taskname
        eval(myf['taskname']+'.defaults()')

        muserlog.origin('default')
        taskstring=str(taskname).split()[0]
        muserlog.post(' #######  Setting values to default for task: '+taskstring+'  #######')


    except TypeError, e:
        print "default --error: ", e

def taskparamgui(useGlobals=True):
    """
        Show a parameter-setting GUI for all available tasks.
    """
    import paramgui

    if useGlobals:
        paramgui.setGlobals(sys._getframe(len(inspect.stack())-1).f_globals)
    else:
        paramgui.setGlobals({})

    paramgui.runAll(_ip)
    paramgui.setGlobals({})

####################

def exit():
    #__IPYTHON__.exit_now=True
    sys.exit()
    #print 'Use CNTRL-D to exit'
    #return

import pylab as pl

#
# 
import platform
##
## CAS-951: matplotlib unresponsive on some 64bit systems
##

# if (platform.architecture()[0]=='64bit'):
#     if os.environ.has_key('DISPLAY') and os.environ['DISPLAY']!="" and not muser['flags'].has_key('--nogui'):
#         pl.ioff( )
#         pl.clf( )
#         pl.ion( )
# ##
##

# Provide flexibility for boolean representation in the muser shell
true  = True
T     = True
false = False
F     = False

import IPython
ipython = True
# Case where muserpy is run non-interactively
# ipython = not muser['flags'].has_key('--noipython')
# try:
#    import IPython
# except ImportError, e:
#    print 'Failed to load IPython: ', e
#    exit(1)
#

# setup available tasks
#
from math import *
from tasks import *
from parameter_dictionary import *
from task_help import *

#
# import testing environment
#
# import publish_summary
# import runUnitTest
# #
# home=os.environ['HOME']
#
# #
# # If the pipeline is there and the user requested it, load the pipeline tasks
# #
# if muserpy['flags'].has_key('--pipeline'):
#     if muserpy['dirs']['pipeline'] is not None:
#         sys.path.insert(0,muser['dirs']['pipeline'])
#         import pipeline
#         pipeline.initcli()
#     else:
#         print "Unable to locate pipeline installation, exiting"
#         sys.exit(1)

## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##
##      prelude.py  =>  setup/modification of muser settings (above)
##      init.py     =>  user setup (with task access)
##
## ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# if os.path.exists( muserpy['dirs']['rc'] + '/init.py' ) :
#     try:
#         execfile ( muserpy['dirs']['rc'] + '/init.py' )
#     except:
#         print str(sys.exc_info()[0]) + ": " + str(sys.exc_info()[1])
#         print 'Could not execute initialization file: ' + muserpy['dirs']['rc'] + '/init.py'
#         sys.exit(1)

if ipython:
    startup()

# assignment protection
#
#pathname=os.environ.get('muserPATH').split()[0]
#uname=os.uname()
#unameminusa=str.lower(uname[0])
fullpath = muserdef.python_library_directory + 'assignmentFilter.py'
#muserlog.origin('muser')

#
# Use something else than python's builtin help() for
# documenting muserpy tasks
#
import pydoc

import os

if IPython.version_info[0]>=5:
    from IPython.terminal.prompts import Prompts, Token
    class MyPrompt(Prompts):

        def in_prompt_tokens(self, cli=None):   # default
            return [
                (Token.Prompt, 'Muser<'),
                (Token.PromptNum, str(self.shell.execution_count)),
                (Token.Prompt, '>: '),
            ]

        def continuation_prompt_tokens(self, cli=None, width=None):
            # if width is None:
            #     width = self._width()
            # return [
            #     (Token.Prompt, (' ' * (width - 2)) + '>'),
            # ]
            return [(Token.Prompt,''),]

        def out_prompt_tokens(self, cli=None):   # default
            return [
                (Token.Prompt, 'Muser-Out['),
                (Token.PromptNum, str(self.shell.execution_count)),
                (Token.Prompt, ']: '),
            ]


class muserDocHelper(pydoc.Helper):
    def help(self, request):
        if hasattr(request, 'i_am_a_muserpy_task'):
            pydoc.pager('Help on ' + pydoc.text.bold(request.__name__) + ' task:\n\n' + request.__doc__)
        else:
            return pydoc.Helper.help(self, request)

pydoc.help = muserDocHelper(sys.stdin, sys.stdout)

fullpath= muserdef.python_library_directory + '/assignmentFilter.py'

if os.environ.has_key('MUSERHOME'):
    fullpath=os.environ['MUSERHOME'] + '/python/assignmentFilter.py'
    showbanner = os.environ['MUSERHOME'] + '/python/showbanner.py'



#print fullpath
if ipython:
    try:
        print 'Current IPython Version:', IPython.version_info[0]
        if IPython.version_info[0]<5:
            c = IPython.Config()
            # c = get_config()
            c.TerminalIPythonApp.display_banner = False
            c.InteractiveShell.colors = __ipython_colors
            c.InteractiveShell.automagic = False
            c.InteractiveShell.autocall = 2
            c.PromptManager.in_template  = 'Muser <\#>: '
            c.PromptManager.in2_template = '   .\D.: '
            c.PromptManager.out_template = 'Muser-Out[\#]: '

            c.PromptManager.justify = True#
            c.InteractiveShellApp.log_level = 0
            c.InteractiveShellApp.exec_lines = [
                'execfile("'+showbanner+'")',
             ]

            IPython.start_ipython(config=c, user_ns=globals())
        else:
            from traitlets.config.loader import Config

            c = Config()
            c.TerminalIPythonApp.display_banner = False
            c.InteractiveShell.colors = 'Neutral'
            c.InteractiveShell.automagic = False
            c.InteractiveShell.autocall = 2
            c.InteractiveShellApp.log_level = 0
            c.TerminalInteractiveShell.prompts_class = MyPrompt
            c.InteractiveShellApp.exec_lines = [
                'execfile("'+showbanner+'")',
             ]
            #c.TerminalInteractiveShell.banner2 = '*** Welcome! ***'


            IPython.start_ipython(config=c, user_ns=globals())
            ip=get_ipython()
            ip.prompts= MyPrompt(ip)

        #ipshell = IPython.start_ipython(argv=['-prompt_in1','muser <\#>: ','-autocall','2','-colors',__ipython_colors, '-nomessages', '-nobanner','-logfile',ipythonlog,'-upgrade','-ipythondir',muser['dirs']['rc']+'/ipython'])
        try:
            if muserdb is not None:
                muserdb.close()
        except:
            pass
        print "Thanks for using MUSER Data Processing System."
    except:
        print "ERROR: falied to create an instance of IPython.Shell.IPShell -d"


import shutil

###
### append user's originial path...
###
# if os.environ.has_key('_PYTHONPATH'):
#     sys.path.extend(os.getenv('_PYTHONPATH').split(':'))

# if ipython:
#     #ipshell.mainloop( )
#     if(os.uname()[0] == 'Darwin') and type(muser) == "<type 'dict'>" and muser['flags'].has_key('--maclogger') :
#            os.system("osascript -e 'tell application \"Console\" to quit'")
#     for pid in logpid:
#         #print 'pid: ',pid
#         os.kill(pid,9)
#
#     for x in os.listdir('.'):
#        if x.lower().startswith('muserpy.scratch-'):
#           if os.path.isdir(x):
#              #shutil.rmtree(x, ignore_errors=True)
#              os.system("rm -rf %s" % x)
#              #print "Removed: ", x, "\n"
#
#     ## leave killing off children to the watchdog...
#     ## so everyone has a chance to die naturally...
#     print "leaving muserpy..."
