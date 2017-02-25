import sys
import os
import time
import datetime
import logging

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.worker import Worker
from opencluster.errors import *
from opencluster.item import ObjValue
from pymuser.muserfile import MuserFile

logger = logging.getLogger("WorkerBigIntegration")
class BigUvfitsCompWorker(Worker) :
    def __init__(self):
        super(BigUvfitsCompWorker,self).__init__(workerType="bigUvfitsCompWorker", level = logging.DEBUG)
        self.uvfits =[MuserFile(1),MuserFile(2)]

    def doTask(self, task_data):# task_data is typeof ObjValue

        logger.info("rawFile:%s"%(task_data.getObj("rawFile")))

        self.uvfits[int(task_data.getObj("freq"))-1].set_parameters(datetime.datetime.strptime(task_data.getObj("timeStr"),"%Y%m%d%H%M%S.%f"),
                                                                    int(task_data.getObj("integralNumber")),int(task_data.getObj("repeat")),0,1,'',self.level==logging.DEBUG,0)

        startTime = time.time()
        fileU,fileV = self.uvfits[int(task_data.getObj("freq"))-1].write_integral_uvfits()
        logger.info("%s's used,frametime:%s"%((time.time()-startTime),datetime.datetime.strptime(task_data.getObj("timeStr"),"%Y%m%d%H%M%S.%f")))
        return (fileU,fileV)

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python bigUvfitsCompWorker.py localIP"
            sys.exit(1)

        wk = BigUvfitsCompWorker()
        wk.waitWorking(sys.argv[1])


    except Exception,e :
        logger.error(e)
