import sys
import os
import logging
import time

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.configuration import Conf
from opencluster.worker import Worker
from pymuser.muserfile import MuserFile

logger = logging.getLogger(__file__)
class RealTimeWorker(Worker):
    def __init__(self):
        super(RealTimeWorker,self).__init__(workerType="realTimeWorker")
        self.uvfits =[MuserFile(1),MuserFile(2)]

    def doTask(self, task_data):

        #"python genUVFITS.py <manual/auto:0/1> <low/high:0/1> <yyyymmddhhmmssssssss> <integralnumber:0/1-10> <repeat:0/n> <calpriority:0-4")
        logger.info("receive workload-----firstFrameTime:%s,freq:%s"%(task_data.getObj("firstFrameTime"),\
                                                                      task_data.getObj("freq")))

        self.uvfits[int(task_data.getObj("freq"))-1].set_parameters(s_time=task_data.getObj("firstFrameTime"),\
                                                                    int_time=1,\
                                                                    repeat_number=8,\
                                                                    priority=0, nocalibration=1,inputfile='',debug=1, genraw=1)


        sTime = time.time()
        strResult,fileList = self.uvfits[int(task_data.getObj("freq"))-1].write_single_uvfits()

        logger.debug("%s's used,frametime:%s"%((time.time()-sTime),task_data.getObj("firstFrameTime")))

        return strResult,fileList

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python realTimeWorker.py localIP"
            sys.exit(1)
        wk = RealTimeWorker()
        wk.waitWorking(sys.argv[1])

    except Exception,e :
        print e