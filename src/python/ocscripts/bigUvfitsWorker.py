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

logger = logging.getLogger("bigUvfitsWorker")
class BigUvfitsWorker(Worker) :
    def __init__(self):
        super(BigUvfitsWorker,self).__init__(workerType="bigUvfitsWorker", level = logging.DEBUG)
        self.uvfits =[MuserFile(1),MuserFile(2)]

    def doTask(self, task_data):# task_data is typeof ObjValue

        logger.info("rawFile:%s,offset:%s,freq:%s,repeat:%s"%(task_data.getObj("filename"),\
                                                                   task_data.getObj("offset"),\
                                                                   task_data.getObj("freq"),
                                                                   task_data.getObj("repeat")))

        startTime = time.time()

        self.uvfits[int(task_data.getObj("freq"))-1].set_parameters(s_time=task_data.getObj("firstTime"),
                                                                    int_time=0,repeat_number=int(task_data.getObj("repeat")),priority=0,nocalibration=1,debug=self.level==logging.DEBUG)

        list = self.uvfits[int(task_data.getObj("freq"))-1].get_visdata(filename=task_data.getObj("filename"),offset=task_data.getObj("offset"),repeat_number=task_data.getObj("repeat"))
        logger.info("%s's used"%((time.time()-startTime)))
        return list

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python bigUvfitsWorker.py localIP"
            sys.exit(1)

        wk = BigUvfitsWorker()
        wk.waitWorking(sys.argv[1])
        # task_data = ObjValue()
        # task_data.setObj("timeStr","20151101120852.007259")
        # task_data.setObj("freq",1)
        # task_data.setObj("integralNumber",40)
        # task_data.setObj("repeat",1)
        # wk.doTask(task_data)

    except Exception,e :
        logger.error(e)
