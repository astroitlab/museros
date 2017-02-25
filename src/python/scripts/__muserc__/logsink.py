
from sys import version_info


class logsink():
    """Proxy of C++ casac::logsink class"""
    # __setattr__ = lambda self, name, value: self.name = value
    # __getattr__ = lambda self, name: return self.name
    __repr__ = "logsink"
    def __init__(self): 
        """__init__(self) -> logsink"""
        self.a = 1

    def origin(self, *args, **kwargs):
        """
        origin(self, fromwhere) -> bool

        Summary
        	Set the origin of the message

        Description
        	Sets the origin of messages to be displayed

        Input Parameters:
        	fromwhere	 The origin of a log messages 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def processorOrigin(self, *args, **kwargs):
        """
        processorOrigin(self, fromwhere) -> bool

        Summary
        	Set the CASA processor origin

        Description
        	Sets the CASA processor origin which is shown at the end of each log origin

        Input Parameters:
        	fromwhere	 Input CASA processor origin name 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def filter(self, *args, **kwargs):
        """
        filter(self, level=string("DEBUG2")) -> bool

        Summary
        	Set the filter level

        Description
        	Set the filter level of logging messages to be displayed.
        	This will determine what log messages go into the log file. The logger itself
                can adjust what gets displayed so you could set INFO5 and then filter in the
                logger everything above INFO1.

        Input Parameters:
        	level		 Level of messages to display to the console/log file ERROR WARN INFO INFO1 INFO2 INFO3 INFO4 INFO5 DEBUG DEBUG1 DEBUG2 INFO 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def filterMsg(self, *args, **kwargs):
        """
        filterMsg(self, msgList=std::vector< string >(1, ""))

        Summary
        	Add messages to the filter out list

        Description
        	Add messages to the filter out list

        Input Parameters:
        	msgList		 Array of strings identifying messages to filter out 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def clearFilterMsgList(self):
        """
        clearFilterMsgList(self)

        Summary
        	Clear list of messages to be filter out

        Description
        	Clear list of messages to be filter out
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def post(self, *args, **kwargs):
        """
        post(self, message, priority=string("INFO"), origin=string("")) -> bool

        Summary
        	Post a message

        Description
        	If the message passes the filter, write it (same as postLocally)

        Input Parameters:
        	message		 Message to be posted 
        	priority	 Priority of message to be posted INFO 
        	origin		 Origin of message to be posted 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def postLocally(self, *args, **kwargs):
        """
        postLocally(self, message, priority=string("INFO"), origin=string("")) -> bool

        Summary
        	Post locally

        Description
        	If the message passes the filter, write it

        Input Parameters:
        	message		 Message to be posted 
        	priority	 Priority of message to be posted INFO 
        	origin		 Origin of message to be posted 
        	
        Example:
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def localId(self):
        """
        localId(self) -> string

        Summary
        	Get local ID

        Description
        	Returns the id for this class
        Example:
        	
        --------------------------------------------------------------------------------
        	      
        """
        return 1 #_logsink.logsink_localId(self)

    def version(self):
        """
        version(self) -> string

        Summary
        	version of CASA

        Description
        	Returns the version of CASA as well as sending it to the log
        Example:
        	casalog.version()
        --------------------------------------------------------------------------------
        	      
        """
        return 1.0

    def id(self):
        """
        id(self) -> string

        Summary
        	Get ID

        Description
        	Returns the ID of the LogSink in use
        Example:
        	
        --------------------------------------------------------------------------------
        	      
        """
        return 1

    def setglobal(self, isglobal=True):
        """
        setglobal(self, isglobal=True) -> bool

        Summary
        	Set this logger to be the global logger

        Input Parameters:
        	isglobal	 Use as global logger true 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def setlogfile(self, *args, **kwargs):
        """
        setlogfile(self, filename=string("casapy.log")) -> bool

        Summary
        	Set the name of file for logger output

        Input Parameters:
        	filename	 filename for logger casapy.log 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True

    def showconsole(self, onconsole=False):
        """
        showconsole(self, onconsole=False) -> bool

        Summary
        	Choose to send messages to the console/terminal

        Input Parameters:
        	onconsole	 All messages to the console as well as log file false 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return True #_logsink.logsink_showconsole(self, onconsole)

    def logfile(self):
        """
        logfile(self) -> string

        Summary
        	Returns the full path of the log file

        Description
        	Returns the full path of the log file
        Example:
        	logfile = casalog.logfile()
        --------------------------------------------------------------------------------
        	      
        """
        return True #_logsink.logsink_logfile(self)


# This file is compatible with both classic and new-style classes.


