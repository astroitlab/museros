import socket
import select
import sys
import time
import sockutils
import logging
from streamframe import StreamFrame
from errors import *

logger = logging.getLogger(__name__)

class MultiplexedSocketServerBase(object):
    def __init__(self, host, port, daemon):
        self.host = host
        self.port = port
        self.daemon = daemon
        self.sock = None
        self.clients=set()

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
        except Exception,e:
            self.sock = None
            logger.error("create socket listen error :%s",e)

    def events(self, eventSockets):
        for s in eventSockets:
            if s is self.sock:
                conn=self._handleConnection(self.sock)
                if conn:
                    self.clients.add(conn)
            else:
                # must be client socket, means remote call
                active = self.handleRequest(s)
                if not active:
                    s.close()
                    self.clients.discard(s)

    def _handleConnection(self, sock):
            try:
                if sock is None:
                    return
                csock, caddr = sock.accept()
                #csock.settimeout(2000)
            except socket.error:
                x=sys.exc_info()[1]
                err=getattr(x, "errno", x.args[0])
                if err in sockutils.ERRNO_RETRIES:
                    # just ignore this error for now and continue
                    logger.warning("accept() failed errno=%d, shouldn't happen", err)
                    return None
                if err in sockutils.ERRNO_BADF or err in sockutils.ERRNO_ENOTSOCK:
                    # our server socket got destroyed
                    raise ConnectionClosedError("server socket closed")
                raise

            try:
                conn = sockutils.SocketConnection(csock)
                return conn
            except:     # catch all errors, otherwise the event loop could terminate
                ex_t, ex_v, ex_tb = sys.exc_info()
                logger.warning("error during connect/handshake: %s", ex_v)
                try:
                    csock.shutdown(socket.SHUT_RDWR)
                except (OSError, socket.error):
                    pass
                csock.close()
            return None

    def handleRequest(self, conn):
        """Handles a single connection request event and returns if the connection is still active"""
        try:
            return self.daemon.handleRequest(conn)
        except socket.error:
            # client went away or caused a security error.
            # close the connection silently.
            return False
        except:
            # other error occurred, close the connection, but also log a warning
            ex_t, ex_v, ex_tb = sys.exc_info()
            logger.warning("error during handleRequest: %s", ex_v)
            return False

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock=None
        for c in self.clients:
            try:
                c.close()
            except Exception:
                pass
        self.clients=set()

    def setReuseAddr(self):
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
    def setNoDelay(self, sock):
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def setKeepAlive(self, sock):
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass

    def __del__(self):
        if self.sock is not None:
            self.sock.close()
            self.sock=None


class SocketServerSelect(MultiplexedSocketServerBase):
    def __init__(self, host, port, daemon):
        super(SocketServerSelect,self).__init__(host, port, daemon)

    def loop(self, loopCondition=lambda: True):
        logger.debug("entering select-based requestloop")
        while loopCondition():
            try:
                rlist=list(self.clients)
                rlist.append(self.sock)
                try:
                    rlist, _, _ = select.select(rlist, [], [], 2)# 2 seconds
                except select.error:
                    if loopCondition():
                        raise
                    else:
                        # swallow the select error if the loopcondition is no longer true, and exit loop
                        # this can occur if we are shutting down and the socket is no longer valid
                        break
                if self.sock in rlist:
                    try:
                        rlist.remove(self.sock)
                    except ValueError:
                        pass  # this can occur when closing down, even when we just tested for presence in the list
                    conn=self._handleConnection(self.sock)
                    if conn:
                        self.clients.add(conn)
                for conn in rlist:
                    # no need to remove conn from rlist, because no more processing is done after this
                    if conn in self.clients:
                        active = self.handleRequest(conn)
                        if not active:
                            conn.close()
                            self.clients.discard(conn)
            except socket.timeout:
                pass   # just continue the loop on a timeout
            except KeyboardInterrupt:
                logger.debug("stopping on break signal")
                break
        logger.debug("exit select-based requestloop")

if __name__ == "__main__" :
    server = SocketServerSelect("localhost", 8101)
    server.loop(lambda:server.sock is not None)

