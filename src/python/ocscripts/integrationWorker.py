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

logger = logging.getLogger("WorkerIntegration")
class IntegrationWorker(Worker) :
    def __init__(self):
        super(IntegrationWorker,self).__init__(workerType="integrationWorker", level = logging.DEBUG)
        self.uvfits =[MuserFile(1),MuserFile(2)]

    def doTask(self, task_data):# task_data is typeof ObjValue

        logger.info("time:%s,integralNumber:%s,freq:%s,repeat:%s"%(task_data.getObj("timeStr"),\
                                                                   task_data.getObj("integralNumber"),\
                                                                   task_data.getObj("freq"),task_data.getObj("repeat")))
        #"python genUVFITS.py <manual/auto:0/1> <low/high:0/1> <yyyymmddhhmmssssssss> <integralnumber:0/1-10> <repeat:0/n> <calpriority:0-4")
        self.uvfits[int(task_data.getObj("freq"))-1].set_parameters(datetime.datetime.strptime(task_data.getObj("timeStr"),"%Y%m%d%H%M%S.%f"),
                                                                    int(task_data.getObj("integralNumber")),int(task_data.getObj("repeat")),0,1,'',self.level==logging.DEBUG,0)

        startTime = time.time()
        strResult,fileList = self.uvfits[int(task_data.getObj("freq"))-1].write_integral_uvfits()
        logger.info("%s's used,frametime:%s"%((time.time()-startTime),datetime.datetime.strptime(task_data.getObj("timeStr"),"%Y%m%d%H%M%S.%f")))
        return fileList

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python integrationWorker.py localIP"
            sys.exit(1)

        wk = IntegrationWorker()
        wk.waitWorking(sys.argv[1])
        # task_data = ObjValue()
        # task_data.setObj("timeStr","20151101120852.007259")
        # task_data.setObj("freq",1)
        # task_data.setObj("integralNumber",40)
        # task_data.setObj("repeat",1)
        # wk.doTask(task_data)

    except Exception,e :
        logger.error(e)
