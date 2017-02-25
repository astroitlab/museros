import sys
import os
import logging
import time
import datetime

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.worker import Worker
from pymuser.muserfile import MuserFile

logger = logging.getLogger(__file__)
class FitsWorker(Worker):
    def __init__(self):
        super(FitsWorker,self).__init__(workerType="fitsWorker")
        self.uvfits =[MuserFile(1, 0),MuserFile(2, 0)]


    def doTask(self, inhouse):

        #"python genUVFITS.py <manual/auto:0/1> <low/high:0/1> <yyyymmddhhmmssssssss> <integralnumber:0/1-10> <repeat:0/n> <calpriority:0-4")

        logger.info("receive workload-----time:%s,freq:%s"%(inhouse.getObj("timeStr"),inhouse.getObj("freq")))

        self.uvfits[int(inhouse.getObj("freq"))-1].set_parameters(datetime.datetime.strptime(inhouse.getObj("timeStr"),"%Y%m%d%H%M%S.%f"),1,1,0,1,'',1)

        sTime = time.time()
        strResult,fileList = self.uvfits[int(inhouse.getObj("freq"))-1].write_single_uvfits(False)
        logger.debug("%s's used,frametime:%s"%((time.time()-sTime),datetime.datetime.strptime(inhouse.getObj("timeStr"),"%Y%m%d%H%M%S.%f")))

        return fileList

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python fitsWorker.py localIP"
            sys.exit(1)

        wk = FitsWorker()
        wk.waitWorking(sys.argv[1])

    except Exception,e :
        print e
        logger.error(e)