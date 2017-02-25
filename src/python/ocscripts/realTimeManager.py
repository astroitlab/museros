import sys
import os
import logging
import time
import datetime
import cPickle
import traceback
import random
import optparse

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.configuration import setLogger,Conf
from opencluster.item import ObjValue,Task,ManagerOption,Success
from opencluster.manager import Manager
from opencluster.util import decompress,spawn

from rtdclient import RealTimeSender

from pymuser.muserfile import MuserFile
from pymuser.musertime import MuserTime
from realTimeGPUManager import RealTimeGPUManager

logger = logging.getLogger(__file__)

loopCondition = True
class RealTimeManager(Manager):

    def __init__(self,mode,freq):
        super(RealTimeManager,self).__init__(mode=mode,name="MUSERRealTimeManager")
        self.freq = freq
        self.rtClient = RealTimeSender()
        option = ManagerOption(cpus=1,mem=500,gpus=0,parallel=1,workertype="realTimeWorker",warehouse="OpenCluster1",name="MUSERRealTimeManager")
        self.setOption(option)
        self.initialize()
        self.gcleanManager = RealTimeGPUManager(mode,freq)

    def __del__(self):
        del self.rtClient

    def doTask(self, sec):
        logger.info("realtime task running......")

        def check_events():
            global loopCondition
            while loopCondition :
                for e in self.completionEvents() :
                    if isinstance(e.reason,Success) :
                        status,fileList = cPickle.loads(decompress(e.result))
                        self.rtClient.sendData("rtfile",fileList)
                        self.gcleanManager.doTask([e])
                    else:
                        logger.error("Failed:" + str(e.reason.message))
                        self.rtClient.sendString("error",str(e.reason.message))
                time.sleep(4)
        def check_kafka_events():
            global loopCondition
            from kafka import KafkaConsumer, KafkaClient, SimpleProducer
            warehouse_addr = Conf.getWareHouseAddr()
            consumer = KafkaConsumer("%sResult"%self.options.warehouse,
                                   bootstrap_servers=[warehouse_addr],
                                   group_id="cnlab",
                                   auto_commit_enable=True,
                                   auto_commit_interval_ms=30 * 1000,
                                   auto_offset_reset='smallest')

            while loopCondition:
                for message in consumer.fetch_messages():
                    logger.error("topic=%s, partition=%s, offset=%s, key=%s " % (message.topic, message.partition,
                                                 message.offset, message.key))
                    task = cPickle.loads(message.value)

                    if task.state == Task.TASK_FINISHED:
                        logger.error("taskId:%s,success!!!:%s"%(task.id,task.result[1]))
                        self.rtClient.sendData("rtfile",task.result[1])
                    else:
                        logger.error("taskId:%s,failed!!!"%task.id);
                        self.rtClient.sendString("error",str(task.result))

                    consumer.task_done(message)
                    last_data_time = time.time()
                    if not loopCondition:
                        break

        if self.mode != "factory":
            spawn(check_events)
        else :
            warehouse_addrs = Conf.getWareHouseAddr().split(",")
            if len(warehouse_addrs) < 2:
                spawn(check_kafka_events)

        try :
            beginTime = datetime.datetime.strptime("20151101120645.00000","%Y%m%d%H%M%S.%f")
            while True:
                try:
                    #----------------------Production------begin----------------
                    #now = datetime.datetime.now()
                    #theTimeStamp = time.mktime(now.timetuple()) - 120
                    #timeStr = time.strftime("%Y%m%d%H%M%S",time.localtime(theNow))
                    #beginTime = datetime.datetime.fromtimestamp(theTimeStamp)
                     #----------------------Production------end----------------

                    #----------------------Test------begin----------------
                    #testTime = ["20151101120849.354161","20151101120854.354161","20151101120849.376036"]
                    #timeStr = testTime[random.randint(0, 2)]
                    #timeStr = "20151101120849.376036"

                    #beginTime = datetime.datetime.strptime(timeStr,"%Y%m%d%H%M%S.%f")
                    # beginTime = datetime.datetime.strptime("20150202191300.000000","%Y%m%d%H%M%S.%f")
                    #----------------------Test------end----------------

                    muserfile = MuserFile(self.freq)
                    mt_begin = MuserTime()
                    mt_begin.set_with_date_time(beginTime)

                    i_endTime = time.mktime(beginTime.timetuple()) + 100/1e6 #add 100 millseconds
                    endTime = datetime.datetime.fromtimestamp(i_endTime)

                    mt_end = MuserTime()
                    mt_end.set_with_date_time(endTime)

                    firstFrameTime,d,v,loopMode = muserfile.get_data_info(mt_begin, mt_end, 1)

                    if not firstFrameTime :
                        logger.info("cannot find observational data.%s"%beginTime.strftime("%Y%m%d%H%M%S.%f"))
                        i_endTime = time.mktime(beginTime.timetuple()) + 5 #add 5 seconds
                        beginTime = datetime.datetime.fromtimestamp(i_endTime)
                        time.sleep(2)
                        continue

                    task_data = ObjValue()
                    task_data.setObj("firstFrameTime",firstFrameTime)
                    task_data.setObj("freq",self.freq)

                    task = Task(id="%s-%s"%(beginTime.strftime("%Y%m%d%H%M%S.%f"),time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))),\
                            workerClass="realTimeWorker.RealTimeWorker",workDir = os.path.dirname(os.path.abspath(__file__)),\
                            data = task_data,priority=1,\
                            resources={"cpus":self.options.cpus,"mem":self.options.mem,"gpus":self.options.gpus},\
                            warehouse=self.options.warehouse)

                    logger.info(task)
                    self.schedule([task])

                    i_endTime = time.mktime(beginTime.timetuple()) + sec #add 5 seconds
                    beginTime = datetime.datetime.fromtimestamp(i_endTime)

                except Exception,e:
                    print '>>> traceback <<<'
                    traceback.print_exc()
                    print '>>> end of traceback <<<'
                    logger.error(e)

                time.sleep(sec)

        except KeyboardInterrupt,ke:
            loopCondition = False
            self.shutdown()
            self.gcleanManager.shutdown()
            logger.info("Interrupt.....")


if __name__ == "__main__" :

    parser = optparse.OptionParser(usage="Usage: python %prog [options]")

    parser.disable_interspersed_args()
    parser.add_option("-m", "--mode", type="string", default="mesos", help="local, process, standalone, mesos, factory")
    parser.add_option("-f", "--freq", type="int", default="1", help="1 for low frequency ,2 for high frequency")
    parser.add_option("-l", "--loop", type="int", default="5", help="time for loop (seconds)")
    parser.add_option("-q", "--quiet", action="store_true", help="be quiet", )
    parser.add_option("-v", "--verbose", action="store_true", help="show more useful log", )

    (options,args) = parser.parse_args()
    if not options:
        parser.print_help()
        sys.exit(2)

    options.logLevel = (options.quiet and logging.ERROR or options.verbose and logging.DEBUG or logging.INFO)
    setLogger("MUSERRealTimeManager",time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())),options.logLevel)

    task = RealTimeManager(options.mode,options.freq)
    task.doTask(options.loop)