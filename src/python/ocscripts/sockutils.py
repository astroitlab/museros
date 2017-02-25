import socket, os, errno, time, sys
import select
from errors import ConnectionClosedError, TimeoutError, CommunicationError

if sys.platform == "win32":
    USE_MSG_WAITALL = False   # it doesn't work reliably on Windows even though it's defined
else:
    USE_MSG_WAITALL = hasattr(socket, "MSG_WAITALL")

PREFER_IP_VERSION = 4 # 4, 6 or 0 (let OS choose according to RFC 3484)
# Note: other interesting errnos are EPERM, ENOBUFS, EMFILE
# but it seems to me that all these signify an unrecoverable situation.
# So I didn't include them in de list of retryable errors.
ERRNO_RETRIES=[errno.EINTR, errno.EAGAIN, errno.EWOULDBLOCK, errno.EINPROGRESS]
if hasattr(errno, "WSAEINTR"):
    ERRNO_RETRIES.append(errno.WSAEINTR)
if hasattr(errno, "WSAEWOULDBLOCK"):
    ERRNO_RETRIES.append(errno.WSAEWOULDBLOCK)
if hasattr(errno, "WSAEINPROGRESS"):
    ERRNO_RETRIES.append(errno.WSAEINPROGRESS)

ERRNO_BADF=[errno.EBADF]
if hasattr(errno, "WSAEBADF"):
    ERRNO_BADF.append(errno.WSAEBADF)

ERRNO_ENOTSOCK=[errno.ENOTSOCK]
if hasattr(errno, "WSAENOTSOCK"):
    ERRNO_ENOTSOCK.append(errno.WSAENOTSOCK)
if not hasattr(socket, "SOL_TCP"):
    socket.SOL_TCP=socket.IPPROTO_TCP

ERRNO_EADDRNOTAVAIL=[errno.EADDRNOTAVAIL]
if hasattr(errno, "WSAEADDRNOTAVAIL"):
    ERRNO_EADDRNOTAVAIL.append(errno.WSAEADDRNOTAVAIL)

ERRNO_EADDRINUSE=[errno.EADDRINUSE]
if hasattr(errno, "WSAEADDRINUSE"):
    ERRNO_EADDRINUSE.append(errno.WSAEADDRINUSE)

def getIpVersion(hostnameOrAddress):
    """
    Determine what the IP version is of the given hostname or ip address (4 or 6).
    First, it resolves the hostname or address to get an IP address.
    Then, if the resolved IP contains a ':' it is considered to be an ipv6 address,
    and if it contains a '.', it is ipv4.
    """
    address = getIpAddress(hostnameOrAddress)
    if "." in address:
        return 4
    elif ":" in address:
        return 6
    else:
        raise CommunicationError("Unknown IP address format" + address)


def getIpAddress(hostname, workaround127=False, ipVersion=None):
    """
    Returns the IP address for the given host. If you enable the workaround,
    it will use a little hack if the ip address is found to be the loopback address.
    The hack tries to discover an externally visible ip address instead (this only works for ipv4 addresses).
    Set ipVersion=6 to return ipv6 addresses, 4 to return ipv4, 0 to let OS choose the best one or None to use PREFER_IP_VERSION.
    """
    def getaddr(ipVersion):
        if ipVersion == 6:
            family=socket.AF_INET6
        elif ipVersion == 4:
            family=socket.AF_INET
        elif ipVersion == 0:
            family=socket.AF_UNSPEC
        else:
            raise ValueError("unknown value for argument ipVersion.")
        ip=socket.getaddrinfo(hostname or socket.gethostname(), 80, family, socket.SOCK_STREAM, socket.SOL_TCP)[0][4][0]
        if workaround127 and (ip.startswith("127.") or ip=="0.0.0.0"):
            ip=getInterfaceAddress("4.2.2.2")
        return ip
    try:
        if hostname and ':' in hostname and ipVersion is None:
            ipVersion = 0
        return getaddr(PREFER_IP_VERSION) if ipVersion is None else getaddr(ipVersion)
    except socket.gaierror:
        if ipVersion == 6 or (ipVersion is None and PREFER_IP_VERSION == 6):
            # try a (inefficient, but hey) workaround to obtain the ipv6 address:
            # attempt to connect to one of a few ipv6-servers (google's public dns servers),
            # and obtain the connected socket's address. (This will only work with an active internet connection)
            # The Google Public DNS IP addresses are as follows: 8.8.8.8, 8.8.4.4
            # The Google Public DNS IPv6 addresses are as follows:  2001:4860:4860::8888, 2001:4860:4860::8844
            for address in ["2001:4860:4860::8888", "2001:4860:4860::8844"]:
                try:
                    return getInterfaceAddress(address)
                except socket.error:
                    pass
            raise socket.error("unable to determine IPV6 address")
        return getaddr(0)


def getInterfaceAddress(ip_address):
    """tries to find the ip address of the interface that connects to the given host's address"""
    family = socket.AF_INET if getIpVersion(ip_address)==4 else socket.AF_INET6
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        sock.connect((ip_address, 53))   # 53=dns
        return sock.getsockname()[0]
    finally:
        sock.close()


def __nextRetrydelay(delay):
    # first try a few very short delays,
    # if that doesn't work, increase by 0.1 sec every time
    if delay==0.0:
        return 0.001
    if delay==0.001:
        return 0.01
    return delay+0.1


def receiveData(sock, size):
    """Retrieve a given number of bytes from a socket.
    It is expected the socket is able to supply that number of bytes.
    If it isn't, an exception is raised (you will not get a zero length result
    or a result that is smaller than what you asked for). The partial data that
    has been received however is stored in the 'partialData' attribute of
    the exception object."""
    try:
        retrydelay=0.0
        msglen=0
        chunks=[]
        EMPTY_BYTES = b""
        if sys.platform=="cli":
            EMPTY_BYTES = ""
        if USE_MSG_WAITALL:
            # waitall is very convenient and if a socket error occurs,
            # we can assume the receive has failed. No need for a loop,
            # unless it is a retryable error.
            # Some systems have an erratic MSG_WAITALL and sometimes still return
            # less bytes than asked. In that case, we drop down into the normal
            # receive loop to finish the task.
            while True:
                try:
                    data=sock.recv(size, socket.MSG_WAITALL)
                    if len(data)==size:
                        return data
                    # less data than asked, drop down into normal receive loop to finish
                    msglen=len(data)
                    chunks=[data]
                    break
                except socket.timeout:
                    raise TimeoutError("receiving: timeout")
                except socket.error:
                    x=sys.exc_info()[1]
                    err=getattr(x, "errno", x.args[0])
                    if err not in ERRNO_RETRIES:
                        raise ConnectionClosedError("receiving: connection lost: "+str(x))
                    time.sleep(0.00001+retrydelay)  # a slight delay to wait before retrying
                    retrydelay=__nextRetrydelay(retrydelay)
        # old fashioned recv loop, we gather chunks until the message is complete
        while True:
            try:
                while msglen<size:
                    # 60k buffer limit avoids problems on certain OSes like VMS, Windows
                    chunk=sock.recv(min(60000, size-msglen))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    msglen+=len(chunk)
                data = EMPTY_BYTES.join(chunks)
                del chunks
                if len(data)!=size:
                    err=ConnectionClosedError("receiving: not enough data")
                    err.partialData=data  # store the message that was received until now
                    raise err
                return data  # yay, complete
            except socket.timeout:
                raise TimeoutError("receiving: timeout")
            except socket.error:
                x=sys.exc_info()[1]
                err=getattr(x, "errno", x.args[0])
                if err not in ERRNO_RETRIES:
                    raise ConnectionClosedError("receiving: connection lost: "+str(x))
                time.sleep(0.00001+retrydelay)  # a slight delay to wait before retrying
                retrydelay=__nextRetrydelay(retrydelay)
    except socket.timeout:
        raise TimeoutError("receiving: timeout")


def sendData(sock, data):
    """
    Send some data over a socket.
    Some systems have problems with ``sendall()`` when the socket is in non-blocking mode.
    For instance, Mac OS X seems to be happy to throw EAGAIN errors too often.
    This function falls back to using a regular send loop if needed.
    """
    if sock.gettimeout() is None:
        # socket is in blocking mode, we can use sendall normally.
        try:
            sock.sendall(data)
            return
        except socket.timeout:
            raise TimeoutError("sending: timeout")
        except socket.error:
            x=sys.exc_info()[1]
            raise ConnectionClosedError("sending: connection lost: "+str(x))
    else:
        # Socket is in non-blocking mode, use regular send loop.
        retrydelay=0.0
        while data:
            try:
                sent = sock.send(data)
                data = data[sent:]
            except socket.timeout:
                raise TimeoutError("sending: timeout")
            except socket.error:
                x=sys.exc_info()[1]
                err=getattr(x, "errno", x.args[0])
                if err not in ERRNO_RETRIES:
                    raise ConnectionClosedError("sending: connection lost: "+str(x))
                time.sleep(0.00001+retrydelay)  # a slight delay to wait before retrying
                retrydelay=__nextRetrydelay(retrydelay)


_GLOBAL_DEFAULT_TIMEOUT = object()


class SocketConnection(object):
    """A wrapper class for plain sockets, containing various methods such as :meth:`send` and :meth:`recv`"""
    __slots__=["sock", "objectId"]

    def __init__(self, sock, objectId=None):
        self.sock=sock
        self.objectId=objectId

    def __del__(self):
        self.close()

    def send(self, data):
        sendData(self.sock, data)

    def recv(self, size):
        return receiveData(self.sock, size)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except (OSError, socket.error):
            pass
        try:
            self.sock.close()
        except AttributeError:
            pass

    def fileno(self):
        return self.sock.fileno()

    def setTimeout(self, timeout):
        self.sock.settimeout(timeout)

    def getTimeout(self):
        return self.sock.gettimeout()
    timeout=property(getTimeout, setTimeout)