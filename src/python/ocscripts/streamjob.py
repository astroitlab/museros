import sys
import os
import time
import socket
import random
import logging

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.item import Task, ITEM_READY
from opencluster.manager import Manager
from ocscripts.streamsocketserver import SocketServerSelect
from opencluster.threadpool import *
from ocscripts.streamframe import StreamFrame
from ocscripts.errors import *
from ocscripts.gpujob import GPUJob
from ocscripts.rtdclient import RealTimeSender

logger = logging.getLogger(__name__)
frameSize = 100000

class StreamManager(Manager) :
    def __init__(self,inHouse):
        super(StreamManager,self).__init__()
        self.inHouse = inHouse
        self.rtClient = RealTimeSender()

    def __call__(self):
        bt = time.time()
        logger.info(str(id(self))+ " started....")
        self.giveChainTask(self.inHouse)
        used = time.time()-bt
        logger.info("%.3f s used"% used )

    def __del__(self):
        del self.rtClient

    def giveTask(self, inHouse):
        workers = self.getWaitingWorkers("workerUVFITS",True)
        if len(workers) == 0 :
            logger.error("no workerUVFITS")
            return

        bigFrame = bytearray(inHouse.getObj("bigFrame"))
        totalLen = len(bigFrame)

        futureResults = []
        results = []

        i = 0
        j = random.randint(0, len(workers))
        while (i+1)*frameSize <= totalLen :
            frame = bigFrame[i*frameSize:(i+1)*frameSize]
            frameHouse = Task()
            frameHouse.setObj("frame",frame)
            if j == len(workers) :
                j = 0
            w = workers[j].doTask(frameHouse)

            futureResults.append(w.getObj("result"))

            i += 1
            j += 1

        while i > 0 :
            for fr in futureResults :
                if fr.wait(0.5) :
                    w = fr.value
                    if w.status == ITEM_READY :
                        self.rtClient.sendData("vis_%s%s"%(str(w.getObj("polarization")),str(w.getObj("freq"))),w.getObj("csrhData"))
                        self.rtClient.sendData("acv_%s%s"%(str(w.getObj("polarization")),str(w.getObj("freq"))),w.getObj("acv"))
                        w.remove("csrhData")
                        w.remove("acv")
                        results.append(w)
                        i -= 1

        jobHouse = Task()
        jobHouse.setObj("streamJobResults", results)
        return jobHouse


class StreamHandler(object) :
    def __init__(self):
        self.pool = ThreadPool()
    def __del__(self):
        if self.pool is not None:
            self.pool.close()
    def handleRequest(self,conn):
        try:
            frame = StreamFrame.recv(conn)
            wh = Task()
            wh.setObj("bigFrame",frame.data)

            streamManager = StreamManager(wh)
            streamManager.toNext(GPUJob("GPUJob"))
            self.pool.process(streamManager)

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
    streamHandler = StreamHandler()
    server = SocketServerSelect("", 8101, streamHandler)
    server.loop(lambda:server.sock is not None)