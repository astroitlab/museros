import sys
import os
import traceback
import time
import logging

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.worker import Worker
from pymuser.muserdirty import MuserDirty
from conf import MUSERConf

logger = logging.getLogger(__file__)
class DirtyWorker(Worker) :
    def __init__(self):
        super(DirtyWorker,self).__init__(workerType="dirtyWorker")

    def doTask(self, inhouse):

        uvfitsFileName = inhouse.getObj("filename")

        if uvfitsFileName is None :
            raise Exception("filename is null or file not exist")

        fileDate = uvfitsFileName.split("/")[-1]

        outdir = os.path.join(MUSERConf.getGeneratePhotoRoot(), fileDate[0:4], fileDate[4:6], fileDate[6:8])
        if not os.path.exists(outdir) :
            os.makedirs(outdir)

        clean = MuserDirty()

        sTime = time.time()
        v = clean.dirty_realtime(subarray=inhouse.getObj("freq"),
                                           polarization=inhouse.getObj("polar"),
                                           frequency=inhouse.getObj("frequency"),
                                           vis_file="%s_cross.dat"%(uvfitsFileName),
                                           uv_file="%s_uv.dat"%(uvfitsFileName),
                                           ra=inhouse.getObj("ra"),
                                           dec=inhouse.getObj("dec"),
                                           outdir=outdir,
                                           PLOT_ME=True,WRITE_FITS=False,P_ANGLE=False, DEBUG=1)
        logger.info("dirty worker %s's used,one frame"%((time.time()-sTime)))
        return v

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python dirtyWorker.py localIP"
            sys.exit(1)

        wk = DirtyWorker()
        wk.waitWorking(sys.argv[1])
    except Exception,e :
        print '>>> traceback <<<'
        traceback.print_exc()
        print '>>> end of traceback <<<'
