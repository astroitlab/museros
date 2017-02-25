import sys
import os
import logging
import time
import redis
import MySQLdb

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.service import Service
from conf import MUSERConf
from opencluster.errors import *


logger = logging.getLogger(__name__)

redisStr = MUSERConf.getRedis()
mysqlStr = MUSERConf.getMySQL()

mysqlUrls = mysqlStr.split(",")
mysqlIpAndPort = mysqlUrls[0].split(":")

redisIpAndPort = redisStr.split(":")
pool = redis.ConnectionPool(host=redisIpAndPort[0], port=int(redisIpAndPort[1]), db=0)

class CalibrationService(Service) :
    def __init__(self):
        super(CalibrationService,self).__init__("calibrationService")
    def doTask(self, inhouse):
        freq = inhouse.getObj("freq")
        timeStr = inhouse.getObj("timeStr")

        db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))
        theValue = ""

        with db:
            cur = db.cursor()
            cur.execute("select theValue from t_calibration  where freq=" + freq + " and ctime='"+timeStr+"' and priority=0")
            #for row in cur.fetchall():
            data = cur.fetchone()
            if data :
                theValue = str(data[0])
            cur.close()
            logger.debug(self.name + " get frame :" + str(freq) + ":" + freq + ",timeStr:" + timeStr + ",return:" + theValue)

        return theValue

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python %s localIP"%(sys.argv[0])
            sys.exit(1)

        wk = CalibrationService()
        wk.waitWorking(sys.argv[1])

    except Exception,e :
        logger.error(e)
