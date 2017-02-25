import sys
import os
import logging
import zipfile
import MySQLdb

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.worker import Worker
from conf import MUSERConf

mysqlStr = MUSERConf.getMySQL()

mysqlUrls = mysqlStr.split(",")
mysqlIpAndPort = mysqlUrls[0].split(":")

logger = logging.getLogger(__file__)
class ZipWorker(Worker):
    def __init__(self):
        super(ZipWorker,self).__init__(workerType="zipWorker")
        self.db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))

    def __del__(self):
        self.db.close()

    def doTask(self, task_data):
        job_id = task_data.getObj("job_id")
        zipFileName = task_data.getObj("zipFileName")
        zf = zipfile.ZipFile(zipFileName, "w", zipfile.zlib.DEFLATED)

        cur = self.db.cursor()
        try:
            cur.execute("select results from t_integration_task where job_id=%s",[job_id])
            for data in cur.fetchall():
                if data[0]:
                    files = str(data[0]).split(",")
                    for f in files:
                        if f :
                            arcname = os.path.basename(f)
                            zf.write(f,arcname)
        except Exception,e:
            import traceback
            msg = traceback.format_exc()
            logger.error("zipfile failed: %s", msg)
            return ""
        finally:
            cur.close()
            zf.close()

        return zipFileName

    def zipFiles(self, fileNames,zipFileName):
        filelist = []

        if isinstance(fileNames, list):
            filelist.extend(fileNames)
        elif os.path.isfile(fileNames):
            filelist.append(fileNames)
        else :
            for root, dirs, files in os.walk(fileNames):
                for name in files:
                    filelist.append(os.path.join(root, name))

        zf = zipfile.ZipFile(zipFileName, "w", zipfile.zlib.DEFLATED)

        for f in filelist:
            arcname = os.path.basename(f)
            zf.write(f,arcname)
        zf.close()

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python zipWorker.py localIP"
            sys.exit(1)
        wk = ZipWorker()
        wk.waitWorking(sys.argv[1])

    except Exception,e :
        print e