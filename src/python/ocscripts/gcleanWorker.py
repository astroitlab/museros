import sys
import os
import logging
import time

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.worker import Worker
from opencluster.errors import *
from pymuser.muserclean import MuserClean
from conf import MUSERConf


logger = logging.getLogger(__name__)
class GCleanWorker(Worker) :
    def __init__(self):
        super(GCleanWorker,self).__init__()

    def doTask(self, inhouse):

        uvfitsFileName = inhouse.getObj("fileName")

        if uvfitsFileName is None or not os.path.exists(uvfitsFileName) :
            raise Exception("filename is null or file not exist")

        basename = os.path.basename(uvfitsFileName)
        fileDate = basename.split("-")[0]

        frameDir = os.path.join(MUSERConf.getGeneratePhotoRoot(), fileDate[0:4], fileDate[4:6], fileDate[6:8])
        if not os.path.exists(frameDir) :
            os.makedirs(frameDir)

        clean = MuserClean()
        flag_ant = [8, 9,10, 11, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39] #20151101

        v = clean.clean_with_fits(infile=uvfitsFileName,outdir=frameDir,ISIZE=512,PLOT_ME=True,WRITE_FITS=True,Flag_Ant=flag_ant)

        return v
if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python gcleanWorker.py localIP"
            sys.exit(1)

        wk = GCleanWorker()
        wk.waitWorking(sys.argv[1])
    except Exception,e :
        logger.error(e)
