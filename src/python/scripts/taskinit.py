#import pmuser
#import muserc
import inspect
import string
import sys
import os

from pymuser.muserlogger import MuserLogger
import globalpy

muserlog = MuserLogger()
#
##allow globals for taskby default
muserglobals=True

# Log initialization ###################################################################################################

# IMPORTANT: The following steps must be follow the described order, 
#            otherwise a seg fault occurs when setting the log file.
# 1st Create muserlog object, it will be used by tasks when importing taskinit

# Log initialization ###################################################################################################

def setlevel(level):
    """
    set level of output information

    level - The level of output information

    Example:
        setlevel('DEBUG')
    """
    muserlog.setLevel(level)

def debugon():
    """
    Switch DEBUG on.

    Example:
        debugon
    """
    globalpy.muser_global_debug = 1

def debugoff():
    """
    Switch DEBUG off.

    Example:
        debugon
    """
    globalpy.muser_global_debug = 0

def setmuser(array):
    """
    Set current Muser Subarray number (1-low, 2-high)

    array - the ID of subarray

    Example:
        setmuser(1)
    """
    if array in [1,2]:
        globalpy.muser_global_muser = array

def gentools(tools=None):
	"""
	Generate a fresh set of tools; only the ones who have
	states..so globally sharing the same one can be unpredicatable 
	im,cb,ms,tb,me,ia,po,sm,cl,cs,rg,sl,dc,vp,msmd,fi,fn,imd,sdms,lm=gentools() 
	or if you want specific set of tools
	im, ia, cb=gentools(['im', 'ia', 'cb'])

	"""
	tooldic={}
	reqtools=[]
        if (not tools) or not hasattr(tools, '__iter__'):
		reqtools=[]
	else:
		reqtools=tools
	return tuple([eval(tooldic[reqtool]) for reqtool in reqtools])

#mc=gentools()

def write_history(myms, vis, tname, param_names, param_vals, myclog=None, debug=False):
        """
        Update vis with the parameters that task tname was called with.

        myms - an ms tool instance
        vis  - the MS to write to.
        tname - name of the calling task.
        param_names - list of parameter names.
        param_vals - list of parameter values (in the same order as param_names).
        myclog - a muserlog instance (optional)
        debug - Turns on debugging print statements on errors if True.

        Example:
        The end of split does
        param_names = split.func_code.co_varnames[:split.func_code.co_argcount]
        param_vals = [eval(p) for p in param_names]  # Must be done in the task.
        write_history(myms, outputvis, 'split', param_names, param_vals,
                      muserlog),
        which appends, e.g.,
        
        vis = 'TWHydra_CO3_2.ms'
        outputvis   = 'scan9.ms'
        datacolumn  = 'data'
        field       = ''
        spw         = ''
        width       = 1
        antenna     = ''
        timebin     = '0s'
        timerange   = ''
        scan        = '9'
        intent      = ''
        array       = ''
        uvrange     = ''
        correlation = ''
        keepflags   = True
        async       = False

        to the HISTORY of outputvis.
        """
        if not hasattr(myms, 'writehistory'):
                if debug:
                        print "write_history(myms, %s, %s): myms is not an ms tool" % (vis, tname)
                return False
        retval = True
        isopen = False
        try:
                if not myclog and hasattr(muserlog, 'post'):
                        myclog = muserlog
        except Exception, instance:
                # There's no logger to complain to, and I don't want to exit
                # just because of that.
                pass
        try:
                myms.open(vis, nomodify=False)
                isopen = True
                myms.writehistory(message='taskname=%s' % tname, origin=tname)
                vestr = 'version: '
                try:
                        # Don't use myclog.version(); it also prints to the
                        # logger, which is confusing.
                        vestr += muser['build']['version'] + ' '
                        vestr += muser['source']['url'].split('/')[-2]
                        vestr += ' rev. ' + muser['source']['revision']
                        vestr += ' ' + muser['build']['time']
                except Exception, instance:
                        if hasattr(myclog, 'version'):
                                # Now give it a try.
                                vestr += myclog.version()
                        else:
                                vestr += ' could not be determined' # We tried.
                myms.writehistory(message=vestr, origin=tname)

                # Write the arguments.
                for argnum in xrange(len(param_names)):
                        msg = "%-11s = " % param_names[argnum]
                        val = param_vals[argnum]
                        if type(val) == str:
                                msg += '"'
                        msg += str(val)
                        if type(val) == str:
                                msg += '"'
                        myms.writehistory(message=msg, origin=tname)
        except Exception, instance:
                if hasattr(myclog, 'post'):
                        myclog.post("*** Error \"%s\" updating HISTORY of %s" % (instance, vis),
                                    'SEVERE')
                retval = False
        finally:
                if isopen:
                        myms.close()
        return retval        

###done with common tools

# setup viewer tool
# jagonzal (CAS-4322): Don't load viewer at the engine level
# if not os.environ.has_key('muser_ENGINE'):
# 	try :
# 		if muser.has_key('state') and muser['state'].has_key('startup') :
# 			ving = viewertool.viewertool( False, pre_launch=muser['state']['startup'] )
# 			if muser['flags'].has_key('--nogui') :
# 				vi = ving
# 			else:
# 				vi = viewertool.viewertool( True, pre_launch=muser['state']['startup'] )
# 	except :
# 		print "Unable to start viewer, maybe no dbus available?"

defaultsdir = {}

# def selectfield(vis,minstring):
#         """Derive the fieldid from  minimum matched string(s): """
#
#         # tb.open(vis+'/FIELD')
#         # fields=list(tb.getcol('NAME'))#get fieldname list
#         # tb.close()              #close table
#         indexlist=list()        #initialize list
# 	stringlist=list()
#
#         fldlist=minstring.split()#split string into elements
# 	print 'fldlist is ',fldlist
#         for fld in fldlist:     #loop over fields
#                 _iter=fields.__iter__() #create iterator for fieldnames
#                 while 1:
#                         try:
#                                 x=_iter.next() # has first value of field name
#                         except StopIteration:
#                                 break
#                         #
#                         if (x.find(fld)!=-1):
# 				indexlist.append(fields.index(x))
# 				stringlist.append(x)
#
# 	print 'Selected fields are: ',stringlist
#         return indexlist

# def selectantenna(vis,minstring):
#         """Derive the antennaid from matched string(s): """
#
#         tb.open(vis+'/ANTENNA')
#         ants=list(tb.getcol('NAME'))#get fieldname list
#         tb.close()              #close table
#         indexlist=list()        #initialize list
# 	stringlist=list()
#
#         antlist=minstring.split()#split string into elements
#         for ant in antlist:     #loop over fields
#         	try:
# 			ind=ants.index(ant)
# 			indexlist.append(ind)
# 			stringlist.append(ant)
#                 except ValueError:
#                         pass
#
# 	print 'Selected reference antenna: ',stringlist
# 	print 'indexlist: ',indexlist
#         return indexlist[0]

def readboxfile(boxfile):
	""" Read a file containing clean boxes (compliant with AIPS BOXFILE)

	Format is:
	#FIELDID BLC-X BLC-Y TRC-X TRC-Y
	0       110   110   150   150 
	or
	0       hh:mm:ss.s dd.mm.ss.s hh:mm:ss.s dd.mm.ss.s

	Note all lines beginning with '#' are ignored.

	"""
	union=[]
	f=open(boxfile)
	while 1:
		try: 
			line=f.readline()
			if (line.find('#')!=0): 
				splitline=line.split('\n')
				splitline2=splitline[0].split()
				if (len(splitline2[1])<6): 
					boxlist=[int(splitline2[1]),int(splitline2[2]),
					int(splitline2[3]),int(splitline2[4])]
				else:
					boxlist=[splitline2[1],splitline2[2],splitline2[3],
 					splitline2[4]]
	
				union.append(boxlist)
	
		except:
			break

	f.close()
	return union


def array2string( array ):
	returnValue=""
	for i in range( len(array) ):
		if ( i > 1 ):
			returnValue+=","
		if ( isinstance( array[i], str ) ):
			returnValue+=array[i]
		else:
			returnValue+=str(array[i])
	return returnValue

def recursivermdir( top='' ):
	# Delete everything from the directory named in 'top',
	# assuming there are no symbolic links.
	for root, dirs, files in os.walk( top, topdown=False ):
		for name in files:
			os.remove( os.path.join( root, name ) )
		for name in dirs:
			os.rmdir( os.path.join( root, name ) )
	os.rmdir(top)
