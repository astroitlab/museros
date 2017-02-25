import sys
import os
import time
import datetime
import logging
import random
import traceback
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

logger = logging.getLogger(__name__)
frameSize = 100000

redisStr = MUSERConf.getRedis()
mysqlStr = MUSERConf.getMySQL()

redisIpAndPort = redisStr.split(":")
mysqlUrls = mysqlStr.split(",")
mysqlIpAndPort = mysqlUrls[0].split(":")
loopCondition = True

class OneRawFileUvfitsManager(Manager) :
    def __init__(self,mode,task):
        super(OneRawFileUvfitsManager,self).__init__(mode)
        self.task = task
        self.splitNumber = 100
        self.repeatPerTask = 192
        self.name = "RawFileUVFITS-" + str(task.getObj("id"))
        self.db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))
        option = ManagerOption(cpus=0.5,mem=200,gpus=0,parallel=16,workertype="bigUvfitsWorker",warehouse=task.getObj("warehouse"),name=self.name)
        self.setOption(option)
        self.initialize()

    def __del__(self):
        logger.debug("OneRawFileUvfitsManager was deleted")
        self.db.close()

    def __call__(self):
        bt = time.time()
        cur = self.db.cursor()
        self.splitTask()

        if self.mode != "factory" :
            partFailed = False

            dataDict = {}

            for e in self.completionEvents() :
                cur = self.db.cursor()
                if isinstance(e.reason,Success) :
                    [vis_file, uvw_file, date_file] = cPickle.loads(decompress(e.result))
                    dataDict[e.task.data["offset"]] = (vis_file,uvw_file,date_file)
                    logger.debug("result return: task_id:%s,results:%s"%(e.task.id,cPickle.loads(decompress(e.result))))
                else:
                    partFailed = True
                    logger.error("Task:%s,Failed:%s"%(e.task.id,str(e.reason.message)))
                cur.close()
                self.db.commit()
                for (k,v) in dataDict.items():
                    print k
                print "---------------------------------------------------------------------------"
            try :
                if not partFailed:
                    sorted(dataDict.items())
                    vis_file, uvw_file, date_file = [],[],[]
                    for (k,v) in dataDict.items():
                        print k
                        vis_file.append(v[0])
                        uvw_file.append(v[1])
                        date_file.append(v[2])

                    print vis_file
                    print uvw_file
                    print date_file

                    muserfile = MuserFile(self.task.getObj("freq"))
                    muserfile.set_parameters(s_time=self.task.getObj("firstFrameTime"),
                                              int_time=0,repeat_number=1,priority=0,nocalibration=1,debug=1)
                    bigFitsFileName = muserfile.merge_one_big_uvfits(self.task.getObj("filename"),vis_file,uvw_file,date_file,self.repeatPerTask)


                cur = self.db.cursor()
                if partFailed :
                    cur.execute("update t_integration_task set status=3,results=null,last_time=now() where task_id='"+self.task.getObj("id")+"'")
                else:
                    cur.execute("update t_integration_task set status=2,results=%s,last_time=now() where task_id=%s",(bigFitsFileName,str(self.task.getObj("id"))))
                self.db.commit()
            except Exception,e :
                logger.error("update t_integration_task error:%s"%e)
                traceback.print_exc()
            finally:
                cur.close()
                self.db.close()

        used = time.time()-bt
        logger.info("%.3f s used"% used )
        self.shutdown()

    def splitTask(self):

        freq = int(self.task.getObj("freq"))
        firstFrameTime = self.task.getObj("firstFrameTime")

        i_beginTime = time.mktime(firstFrameTime.timetuple()) + firstFrameTime.microsecond/1e6

        bigFrameTime = 25000  #3125*8

        if freq==2:
            bigFrameTime = 206250 #3125*66

        # ------------------------------------begin split -------------------------------
        tasks = []
        offsetSpan = 19200*100000/self.splitNumber
        offset = 0
        if freq==2:
            offsetSpan = 19200*204800/self.splitNumber

        for i in range(0,self.splitNumber):

            task_data = ObjValue()
            task_data.setObj("filename",self.task.getObj("filename"))
            task_data.setObj("firstTime",datetime.datetime.fromtimestamp(i_beginTime))
            task_data.setObj("freq",freq)
            task_data.setObj("offset",offset)
            task_data.setObj("repeat",self.repeatPerTask)

            task = Task(id="%s-%s"%(self.name,offset),data=task_data,\
                        workerClass="bigUvfitsWorker.BigUvfitsWorker",workDir = os.path.dirname(os.path.abspath(__file__)),priority=3,\
                        resources={"cpus":self.options.cpus,"mem":self.options.mem,"gpus":self.options.gpus},\
                        warehouse=self.options.warehouse,\
                        jobName=self.name)

            tasks.append(task)

            i_beginTime = i_beginTime + self.repeatPerTask*3125/1e6
            offset = offset + offsetSpan

        self.schedule(tasks)
        return True,None



