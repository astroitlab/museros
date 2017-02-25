import sys
import os
import logging
import time
import cPickle

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.item import ObjValue,Task,ManagerOption,Success
from opencluster.manager import Manager
from opencluster.util import decompress,spawn

from rtdclient import RealTimeSender

logger = logging.getLogger(__name__)

loopCondition = True
class RealTimeGPUManager(Manager):
    def __init__(self,mode,freq):
        super(RealTimeGPUManager,self).__init__(mode=mode,name="MUSERRealTimeGPUManager")
        self.freq = freq
        self.rtClient = RealTimeSender()
        option = ManagerOption(cpus=3,mem=1000,gpus=1,parallel=1,workertype="dirtyWorker",warehouse="OpenCluster1",name="MUSERRealTimeGPUManager")
        self.setOption(option)
        self.initialize()

        def check_events():
            global loopCondition
            while loopCondition :
                for e in self.completionEvents() :
                    if isinstance(e.reason,Success) :
                        filePrefix = cPickle.loads(decompress(e.result))
                        logger.debug(filePrefix)
                        self.rtClient.sendData("rtimage",filePrefix)
                    else:
                        logger.error("Failed:" + str(e.reason.message))
                        self.rtClient.sendString("error",str(e.reason.message))
                time.sleep(2)
        if self.mode != "factory":
            spawn(check_events)

    def __del__(self):
        global loopCondition
        loopCondition = False
        time.sleep(1)
        del self.rtClient

    def doTask(self, completionEvents):
        tasks = []

        for e in completionEvents:
            status,fileList = cPickle.loads(decompress(e.result))
            for file in fileList:
                task_data = ObjValue()
                task_data.setObj("filename",file[0])
                task_data.setObj("freq",self.freq)
                task_data.setObj("frequency",file[1])
                task_data.setObj("polar",file[2])
                task_data.setObj("ra",file[3])
                task_data.setObj("dec",file[4])

                task = Task(id="dirty-%s-%s-%s"%(e.task.id,file[1],file[2]),\
                        workerClass="dirtyWorker.DirtyWorker",workDir = os.path.dirname(__file__),data = task_data,priority=1)
                tasks.append(task)
        self.schedule(tasks)
