"""
Definition of the various exceptions that are used in CSRH.

CSRH - Chinese Spectral RadioHeliograph.  Copyright by cnlab.net.
"""
import sys, zlib, logging
import traceback, linecache
try:
    import copyreg
except ImportError:
    import copy_reg as copyreg
__all__ = ["CRSHError","CommunicationError","ConnectionClosedError","TimeoutError","NamingError","DaemonError","SecurityError","formatTraceback"]

class CRSHError(Exception):
    """Generic base of all CSRH-specific errors."""
    pass


class CommunicationError(CRSHError):
    """Base class for the errors related to network communication problems."""
    pass


class ConnectionClosedError(CommunicationError):
    """The connection was unexpectedly closed."""
    pass


class TimeoutError(CommunicationError):
    """
    A call could not be completed within the set timeout period,
    or the network caused a timeout.
    """
    pass


class ProtocolError(CommunicationError):
    """received a message that didn't match the active network protocol, or there was a protocol related error."""
    pass


class NamingError(CRSHError):
    """There was a problem related to the name server or object names."""
    pass


class DaemonError(CRSHError):
    """The Daemon encountered a problem."""
    pass


class SecurityError(CRSHError):
    """A security related error occurred."""
    pass

def formatTraceback(ex_type=None, ex_value=None, ex_tb=None, detailed=False):
    """Formats an exception traceback. If you ask for detailed formatting,
    the result will contain info on the variables in each stack frame.
    You don't have to provide the exception info objects, if you omit them,
    this function will obtain them itself using ``sys.exc_info()``."""
    if ex_type is not None and ex_value is None and ex_tb is None:
        # possible old (3.x) call syntax where caller is only providing exception object
        if type(ex_type) is not type:
            raise TypeError("invalid argument: ex_type should be an exception type, or just supply no arguments at all")
    if ex_type is None and ex_tb is None:
        ex_type, ex_value, ex_tb=sys.exc_info()
    if detailed and sys.platform!="cli":    # detailed tracebacks don't work in ironpython (most of the local vars are omitted)
        def makeStrValue(value):
            try:
                return repr(value)
            except:
                try:
                    return str(value)
                except:
                    return "<ERROR>"
        try:
            result=["-"*52+"\n"]
            result.append(" EXCEPTION %s: %s\n" % (ex_type, ex_value))
            result.append(" Extended stacktrace follows (most recent call last)\n")
            skipLocals=True  # don't print the locals of the very first stackframe
            while ex_tb:
                frame=ex_tb.tb_frame
                sourceFileName=frame.f_code.co_filename
                if "self" in frame.f_locals:
                    location="%s.%s" % (frame.f_locals["self"].__class__.__name__, frame.f_code.co_name)
                else:
                    location=frame.f_code.co_name
                result.append("-"*52+"\n")
                result.append("File \"%s\", line %d, in %s\n" % (sourceFileName, ex_tb.tb_lineno, location))
                result.append("Source code:\n")
                result.append("    "+linecache.getline(sourceFileName, ex_tb.tb_lineno).strip()+"\n")
                if not skipLocals:
                    names=set()
                    names.update(getattr(frame.f_code, "co_varnames", ()))
                    names.update(getattr(frame.f_code, "co_names", ()))
                    names.update(getattr(frame.f_code, "co_cellvars", ()))
                    names.update(getattr(frame.f_code, "co_freevars", ()))
                    result.append("Local values:\n")
                    for name in sorted(names):
                        if name in frame.f_locals:
                            value=frame.f_locals[name]
                            result.append("    %s = %s\n" % (name, makeStrValue(value)))
                            if name=="self":
                                # print the local variables of the class instance
                                for name, value in vars(value).items():
                                    result.append("        self.%s = %s\n" % (name, makeStrValue(value)))
                skipLocals=False
                ex_tb=ex_tb.tb_next
            result.append("-"*52+"\n")
            result.append(" EXCEPTION %s: %s\n" % (ex_type, ex_value))
            result.append("-"*52+"\n")
            return result
        except Exception:
            return ["-"*52+"\nError building extended traceback!!! :\n",
                    "".join(traceback.format_exception(*sys.exc_info())) + '-'*52 + '\n',
                    "Original Exception follows:\n",
                    "".join(traceback.format_exception(ex_type, ex_value, ex_tb))]
    else:
        # default traceback format.
        return traceback.format_exception(ex_type, ex_value, ex_tb)
