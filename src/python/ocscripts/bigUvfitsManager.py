import sys
import os
import time
import datetime
import logging
import random
import MySQLdb
import cPickle

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.item import Task,Success,ManagerOption,ObjValue
from opencluster.manager import Manager
from opencluster.util import decompress,spawn

from conf import MUSERConf
from errors import *
from pymuser.musertime import MuserTime
from pymuser.muserfile import MuserFile
from oneRawFileUvfitsManager import OneRawFileUvfitsManager
from opencluster.threadpool import ThreadPool

logger = logging.getLogger(__name__)
frameSize = 100000

redisStr = MUSERConf.getRedis()
mysqlStr = MUSERConf.getMySQL()

redisIpAndPort = redisStr.split(":")
mysqlUrls = mysqlStr.split(",")
mysqlIpAndPort = mysqlUrls[0].split(":")
loopCondition = True

class BigUvfitsManager(object) :
    def __init__(self,mode,task):
        self.pool = ThreadPool()
        self.task = task
        self.mode = mode
        self.name = "Integration-" + str(task.getObj("id"))
        self.db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))

    def __del__(self):
        if self.pool is not None:
            self.pool.close()
        logger.debug("BigUvfitsManager was deleted")
        self.db.close()

    def __call__(self):
        bt = time.time()
        cur = self.db.cursor()
        submitStatus,message = self.splitTask()
        if not submitStatus:
            try :
                cur.execute("update t_integration set status=4,results=%s where id=%s",(message,str(self.task.getObj("id"))))
                self.db.commit()
            except Exception,e :
                logger.error("update t_integration error:%s"%e)
            finally:
                cur.close()
        used = time.time()-bt
        logger.info("%.3f s used"% used )

    def splitTask(self):

        freq = int(self.task.getObj("freq"))

        v_beginTime = datetime.datetime.strptime(self.task.getObj("beginTime"),"%Y-%m-%d %H:%M:%S")
        v_endTime = datetime.datetime.strptime(self.task.getObj("endTime"),"%Y-%m-%d %H:%M:%S")

        start_time = MuserTime()
        start_time.set_with_date_time(v_beginTime)

        end_time = MuserTime()
        end_time.set_with_date_time(v_endTime)

        muser = MuserFile(freq)
        values = []
        rawfiles = muser.get_file_info(start_time,end_time)
        if not rawfiles:
            logger.info("cannot find observational data.")
            return False,"cannot find observational data."

        #192*25ms/8


        #low freq - frame size:100000,high freq - frame size:204800

        for [firstFrameTime,filename] in rawfiles:
            logger.error("rawfile:%s,first frame time:%s"%(filename,firstFrameTime))

            values.append(("%s-%s"%(self.name,firstFrameTime.strftime("%Y%m%d%H%M%S%f")),firstFrameTime.strftime("%Y%m%d%H%M%S%f"),"1",self.name,self.task.getObj("id"),freq,0,192))
            obj = ObjValue()

            obj.setObj("firstFrameTime",firstFrameTime)
            obj.setObj("filename",filename)
            obj.setObj("freq",freq)
            obj.setObj("id","%s-%s"%(self.name,firstFrameTime.strftime("%Y%m%d%H%M%S%f")))
            manager = OneRawFileUvfitsManager(self.mode,obj)
            manager()
            #self.pool.process(manager)
        try :
            cur = self.db.cursor()
            n = cur.executemany("insert into t_integration_task (task_id,timeStr,status,job_id,last_time,int_id,freq,int_number,repeat_num) values(%s,%s,%s,%s,now(),%s,%s,%s,%s)",values)
            cur.execute("update t_integration set task_num="+str(len(rawfiles)))
            self.db.commit()
        except Exception,e :
            logger.error("insert t_integration_task error:%s"%e)
        return True,None



