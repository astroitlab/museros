import sys
import os
import time
import socket
import logging
import random

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.configuration import Conf
from streamsocketserver import SocketServerSelect

from streamframe import StreamFrame
from errors import *


logger = logging.getLogger("streamserver")


class StreamHandler(object) :
    def __init__(self,lowOrHigh):
        self.lowOrHigh = lowOrHigh
    # def __del__(self):
    #     if self.pool is not None:
    #         self.pool.close()
    def handleRequest(self,conn):
        try:
            frame = StreamFrame.recv(conn,self.lowOrHigh)


            return True
        except socket.error:
            # client went away or caused a security error.
            # close the connection silently.
            return True
        except Exception:
            xt, xv = sys.exc_info()[0:2]
            if xt is not ConnectionClosedError:
                logger.error("Exception occurred while handling request: %r,\n%s", xv,formatTraceback())
            return True

if __name__ == "__main__" :
    if len(sys.argv) != 3 :
        print "Usage : %s [port] [low|high]" % sys.argv[0]
        sys.exit(1)
    streamHandler = StreamHandler(sys.argv[2])
    server = SocketServerSelect("", int(sys.argv[1]), streamHandler)
    server.setReuseAddr()
    logger.info("server is prepared for running......")
    server.loop(lambda:server.sock is not None)