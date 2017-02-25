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

logger = logging.getLogger(__name__)
frameSize = 100000

redisStr = MUSERConf.getRedis()
mysqlStr = MUSERConf.getMySQL()

redisIpAndPort = redisStr.split(":")
mysqlUrls = mysqlStr.split(",")
mysqlIpAndPort = mysqlUrls[0].split(":")
loopCondition = True

class IntegrationManager(Manager) :
    def __init__(self,mode,task):
        super(IntegrationManager,self).__init__(mode)
        self.task = task
        self.name = "Integration-" + str(task.getObj("id"))
        self.db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))
        option = ManagerOption(cpus=1,mem=500,gpus=0,parallel=8,workertype="integrationWorker",warehouse=task.getObj("warehouse"),name=self.name)
        self.setOption(option)
        self.initialize()

    def __del__(self):
        logger.debug("IntegrationManager was deleted")
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
                self.db.close()
            return
        if self.mode != "factory" :
            partFailed = False
            for e in self.completionEvents() :
                cur = self.db.cursor()
                if isinstance(e.reason,Success) :
                    list = cPickle.loads(decompress(e.result))
                    #logger.debug("result return: task_id:%s,results:%s"%(e.task.id,cPickle.loads(decompress(e.result))))
                    cur.execute("update t_integration_task set status=2,results=%s,last_time=now() where task_id=%s",(",".join(list),str(e.task.id)))
                else:
                    cur.execute("update t_integration_task set status=3,results=%s,last_time=now() where task_id=%s",(str(e.reason.message),str(e.task.id)))
                    partFailed = True
                    logger.error("Task:%s,Failed:%s"%(e.task.id,str(e.reason.message)))
                cur.close()
                self.db.commit()

            try :
                cur = self.db.cursor()
                if partFailed :
                    cur.execute("update t_integration set status=4 where id=" + str(self.task.getObj("id")))
                else:
                    cur.execute("update t_integration set status=3 where id=" + str(self.task.getObj("id")))
                self.db.commit()
            except Exception,e :
                logger.error("update t_integration error:%s"%e)
            finally:
                cur.close()
                self.db.close()

        used = time.time()-bt
        logger.info("%.3f s used"% used )
        self.shutdown()

    def splitTask(self):

        freq = int(self.task.getObj("freq"))
        integrationSecond = self.task.getObj("seconds")

        # v_beginTime = datetime.datetime.strptime(inHouse.getObj("beginTime"),"%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")
        # v_endTime = datetime.datetime.strptime(inHouse.getObj("endTime"),"%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")

        v_beginTime = datetime.datetime.strptime(self.task.getObj("beginTime"),"%Y-%m-%d %H:%M:%S")
        v_endTime = datetime.datetime.strptime(self.task.getObj("endTime"),"%Y-%m-%d %H:%M:%S")

        start_time = MuserTime()
        start_time.set_with_date_time(v_beginTime)

        end_time = MuserTime()
        end_time.set_with_date_time(v_endTime)

        muser = MuserFile(freq)

        # 3125 microsecond small frame

        firstFrameTime,integralNumber, totalIntNumber, loopMode = muser.get_data_info(start_time, end_time, integrationSecond)
        if not firstFrameTime :
            logger.info("cannot find observational data.")
            return False,"cannot find observational data."

        logger.info("%s,%s,%s,%s"%(firstFrameTime, integralNumber, totalIntNumber, loopMode))

        i_beginTime = time.mktime(firstFrameTime.timetuple()) + firstFrameTime.microsecond/1e6
        i_endTime = time.mktime(datetime.datetime.strptime(self.task.getObj("endTime"),"%Y-%m-%d %H:%M:%S").timetuple())

        totalTimeSpan = (i_endTime - i_beginTime)*(10**6) #how many microsecond is in total time span.
        smallFrameNum = totalTimeSpan//3125 or 1   # calculate how much 3125 in timeSpan or how many small frame


        bigFrameTime = 3125
        if not loopMode :
            bigFrameTime = 3125
        else:
            if freq==1:
                bigFrameTime = 25000  #3125*8
            else:
                bigFrameTime = 206250 #3125*66

        bigFrameNum = totalTimeSpan//bigFrameTime or 1

        logger.info("total microsecond:%s,small frame size:%s,big frame size:%s"%(totalTimeSpan, smallFrameNum,bigFrameNum))


        #taskNum = bigFrameNum//integralNumber  #may be equal to totalIntNumber
        taskNum = totalIntNumber
        repeatPerTask = bigFrameNum//integralNumber//50 or 1 # split 50 parts
        indexNumber = 0
        # ------------------------------------begin split -------------------------------
        tasks = []
        while taskNum > 0:
            task_data = ObjValue()
            task_data.setObj("timeStr",datetime.datetime.fromtimestamp(i_beginTime).strftime("%Y%m%d%H%M%S.%f"))
            task_data.setObj("freq",freq)
            task_data.setObj("integralNumber",integralNumber)
            task_data.setObj("repeat",repeatPerTask)

            task = Task(id="%s-%s"%(self.name,datetime.datetime.fromtimestamp(i_beginTime).strftime("%Y%m%d%H%M%S%f")),data=task_data,\
                        workerClass="integrationWorker.IntegrationWorker",workDir = os.path.dirname(os.path.abspath(__file__)),priority=3,\
                        resources={"cpus":self.options.cpus,"mem":self.options.mem,"gpus":self.options.gpus},\
                        warehouse=self.options.warehouse,\
                        jobName=self.name)

            tasks.append(task)

            i_beginTime = i_beginTime + repeatPerTask*bigFrameTime*integralNumber/1e6
            taskNum = taskNum - repeatPerTask

        values = []
        for t in tasks:
            values.append((t.id,t.data.getObj("timeStr"),"1",self.name,self.task.getObj("id"),freq,str(t.data.getObj("integralNumber")),str(t.data.getObj("repeat"))))
        try :
            cur = self.db.cursor()
            n = cur.executemany("insert into t_integration_task (task_id,timeStr,status,job_id,last_time,int_id,freq,int_number,repeat_num) values(%s,%s,%s,%s,now(),%s,%s,%s,%s)",values)
            cur.execute("update t_integration set task_num="+str(len(tasks)))
            self.db.commit()
        except Exception,e :
            logger.error("insert t_integration_task error:%s"%e)
        self.schedule(tasks)
        return True,None



