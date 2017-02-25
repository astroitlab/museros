import socket
import select
import sys
import os
import time
import sockutils
import logging
import struct

from errors import *
sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.configuration import setLogger

setLogger("SendFrame",0)
logger = logging.getLogger("SendFrame")

class SendFrame(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.fp = open("E:\\astrodata\\CSRH_20140514-132449_1283156","rb")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host,self.port))
            self.conn = sockutils.SocketConnection(self.sock)
        except Exception,e:
            self.sock = None
            logger.error("create socket connect error :%s"%e)

    def start(self,seconds = 1):
        while True :
            try:
                logger.info("send frame......")
                data = self.fp.read(100000)
                size = struct.pack("i",100000)
                self.conn.send(size)
                self.conn.send(data)
            except Exception,e:
                logger.error(e)

            time.sleep(seconds)
    def close(self):
        self.fp.close()
        self.conn.close()
        self.sock.close()
        self.sock = None

if __name__ == "__main__" :
    try :
        sendFrame = SendFrame("localhost", 8101)
        sendFrame.start()
    except socket.timeout:
        pass   # just continue the loop on a timeout
    except KeyboardInterrupt:
        sendFrame.close()
        logger.debug("stopping on break signal")


