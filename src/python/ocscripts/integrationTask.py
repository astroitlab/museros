import sys
import os
import time
import logging
import optparse
import cPickle
import traceback
import MySQLdb

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.configuration import setLogger,Conf,logger
from opencluster.threadpool import ThreadPool
from opencluster.item import Task,ObjValue
from opencluster.util import spawn
from ocscripts.kafkaUtils import KafkaUtils

from integrationManager import IntegrationManager
from bigUvfitsManager import BigUvfitsManager
from conf import MUSERConf

mysqlStr = MUSERConf.getMySQL()
mysqlUrls = mysqlStr.split(",")
mysqlIpAndPort = mysqlUrls[0].split(":")

loopCondition = True

class IntegrationTask(object) :
    def __init__(self,mode,warehouse):
        self.pool = ThreadPool()
        self.mode = mode
        self.warehouse = warehouse
        self.kafkaUtil = KafkaUtils(MUSERConf.getKafka())

    def __del__(self):
        if self.pool is not None:
            self.pool.close()
        self.kafkaUtil.close()

    def doTask(self, sec):
        logger.info("working in mode [%s], task dispather running......"%self.mode)

        def check_kafka_events(warehouse):
            global loopCondition
            from kafka import KafkaConsumer, KafkaClient, SimpleProducer
            warehouse_addr = Conf.getWareHouseAddr()
            consumer = KafkaConsumer("%sResult"%warehouse,
                                   bootstrap_servers=[warehouse_addr],
                                   group_id="cnlab",
                                   auto_commit_enable=True,
                                   auto_commit_interval_ms=30 * 1000,
                                   auto_offset_reset='smallest')

            db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))

            try :
                while loopCondition:
                    for message in consumer.fetch_messages():
                        cur = db.cursor()
                        try :
                            logger.info("topic=%s, partition=%s, offset=%s, key=%s " % (message.topic, message.partition,
                                                         message.offset, message.key))
                            task = cPickle.loads(message.value)

                            if task.state == Task.TASK_FINISHED:
                                logger.info("taskId:%s,success!!!"%task.id);
                                cur.execute("update t_integration_task set status=2,results=%s,last_time=now() where task_id=%s",(",".join(task.result),str(task.id)))
                            else:
                                logger.info("taskId:%s,failed!!!"%task.id);
                                cur.execute("update t_integration_task set status=3,results=%s,last_time=now() where task_id=%s",(str(e.result),str(task.id)))
                            db.commit()

                            consumer.task_done(message)
                            if not loopCondition:
                                break
                        except Exception,e:
                            print '>>> traceback <<<'
                            traceback.print_exc()
                            print '>>> end of traceback <<<'
                            logger.error(e)
                        finally:
                            cur.close()
            finally:
                db.close()

        def check_kafka_zip_events(warehouse):
            global loopCondition
            from kafka import KafkaConsumer, KafkaClient, SimpleProducer
            warehouse_addr = Conf.getWareHouseAddr()
            consumer = KafkaConsumer("%sResult"%warehouse,
                                   bootstrap_servers=[warehouse_addr],
                                   group_id="cnlab",
                                   auto_commit_enable=True,
                                   auto_commit_interval_ms=30 * 1000,
                                   auto_offset_reset='smallest')

            db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))

            try :
                while loopCondition:
                    for message in consumer.fetch_messages():
                        cur = db.cursor()
                        try :
                            logger.info("topic=%s, partition=%s, offset=%s, key=%s " % (message.topic, message.partition,
                                                         message.offset, message.key))
                            task = cPickle.loads(message.value)

                            if task.state == Task.TASK_FINISHED:
                                logger.error("taskId:%s,success!!!"%task.id);
                                cur.execute("update t_integration set results=%s where job_id=%s",(str(task.result),str(task.id)))
                            else:
                                logger.error("taskId:%s,failed!!!"%task.id);
                                cur.execute("update t_integration set results=%s where job_id=%s",(str(e.result),str(task.id)))
                            db.commit()

                            consumer.task_done(message)
                            if not loopCondition:
                                break
                        except Exception,e:
                            print '>>> traceback <<<'
                            traceback.print_exc()
                            print '>>> end of traceback <<<'
                            logger.error(e)
                        finally:
                            cur.close()
            finally:
                db.close()

        if self.mode == "factory":
            warehouse_addrs = Conf.getWareHouseAddr().split(",")
            if len(warehouse_addrs) < 2:
                spawn(check_kafka_events,self.warehouse)
            spawn(check_kafka_zip_events,"OpenCluster4")

        try :
            last_time = time.time()
            while True:
                logger.info("dispather task......")
                db = None
                try :
                    db = MySQLdb.connect(db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))
                    cur = db.cursor()
                    cur.execute("select date_format(beginTime,'%Y-%m-%d %H:%i:%s'),date_format(endTime,'%Y-%m-%d %H:%i:%s'),seconds,freq,id,\
                    format,is_specified_file,specified_file from t_integration where status=1 order by beginTime ASC limit 1")
                    #for row in cur.fetchall():
                    data = cur.fetchone()
                    if data :
                        n = cur.execute("update t_integration set status=2,job_id='Integration-"+str(data[4])+"' where id=" + str(data[4]))
                        db.commit()
                        logger.debug("prepared task :%s~%s,%s,%s,%s"%(data[0],data[1],data[2],data[3],str(data[4])))

                        obj = ObjValue()

                        obj.setObj("beginTime",data[0])
                        obj.setObj("endTime",data[1])
                        obj.setObj("seconds",data[2])
                        obj.setObj("freq",data[3])
                        obj.setObj("id",data[4])

                        obj.setObj("format",data[5])
                        obj.setObj("is_specified_file",data[6])
                        obj.setObj("specified_file",data[7])
                        obj.setObj("warehouse",self.warehouse)

                        if int(data[6])==1:
                            manager = BigUvfitsManager(self.mode,obj)
                        else:
                            manager = IntegrationManager(self.mode,obj)

                        manager()
                        # self.pool.process(manager)
                    if time.time() - last_time > 90 :
                        cur.execute("SELECT f_integration_check()")
                        db.commit()
                        items = cur.fetchone()
                        if items :
                            jobIdStr = items[0]
                            jobIds = jobIdStr.split(",")

                            tasks = []
                            for job_id in jobIds :
                                if job_id:
                                    task_data = ObjValue()
                                    task_data.setObj("job_id",job_id)
                                    task_data.setObj("zipFileName",os.path.join(MUSERConf.getZipRoot(),"%s.zip"%job_id))

                                    task = Task(id=job_id,data=task_data,\
                                                workerClass="zipWorker.ZipWorker",workDir = os.path.dirname(os.path.abspath(__file__)),priority=4,\
                                                resources={"cpus":2,"mem":1000,"gpus":0},\
                                                warehouse="OpenCluster4",\
                                                jobName="IntegrationZip")
                                    tasks.append(task)
                            self.kafkaUtil.produceTasks(tasks)

                        logger.info("call f_integration_check...")
                        last_time = time.time()

                except Exception,e :
                    print '>>> traceback <<<'
                    traceback.print_exc()
                    print '>>> end of traceback <<<'
                    logger.error(e)
                finally:
                    if db:
                        db.close()
                time.sleep(sec)

        except KeyboardInterrupt ,e :
            loopCondition = False
            logger.info("Interrupt.....exiting gracefully...")


if __name__ == "__main__" :
    try:
        parser = optparse.OptionParser(usage="Usage: python %prog [options]")

        parser.disable_interspersed_args()
        parser.add_option("-m", "--mode", type="string", default="standalone", help="local, process, standalone, mesos, factory")
        parser.add_option("-l", "--loop", type="int", default="5", help="time for loop (seconds)")
        parser.add_option("-q", "--quiet", action="store_true", help="be quiet", )
        parser.add_option("-v", "--verbose", action="store_true", help="show more useful log", )

        (options,args) = parser.parse_args()
        if not options:
            parser.print_help()
            sys.exit(2)

        options.logLevel = (options.quiet and logging.ERROR or options.verbose and logging.DEBUG or logging.INFO)
        setLogger("integrationTask",time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())),options.logLevel)

        task = IntegrationTask(options.mode,"OpenCluster3")
        task.doTask(options.loop)

    except Exception,e :
        logger.error(e)
