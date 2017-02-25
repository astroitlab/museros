
import logging
import cPickle

from opencluster.servicecontext import ServiceContext
from opencluster.item import Task,OtherFailure
from opencluster.util import compress, decompress

logger = logging.getLogger(__name__)

class WorkerLocal(object) :

    def __init__(self,host,port,workerType,asynchronous=False) :
        self.host = host
        self.port = port
        self.workerType = workerType
        self.asynchronous = asynchronous
        if asynchronous :
            self.workerService = ServiceContext.getAsynchronousService(host, port, workerType)
        else:
            self.workerService = ServiceContext.getService(host, port, workerType)

    def setWorker(self, worker):
        self.workerService.setWorker(worker)

    def receive(self, task):
        received = False
        try :
            received = self.workerService.receiveMaterials(task)
        except Exception, e:
            logger.error(e)
        return received

    def doTask(self, v_task):
        try :
            if self.asynchronous :
                return self.__doTaskAsyn(v_task)
            task =  compress(cPickle.dumps(v_task,-1))

            decompResult = decompress(self.workerService.doTask(task))
            return v_task.id, Task.TASK_FINISHED, cPickle.loads(decompResult)
        except Exception, e:
            logger.error(e)
            return v_task.id, Task.TASK_ERROR,OtherFailure(e.message)


    def __doTaskAsyn(self, v_task):
        try :
            task =  compress(cPickle.dumps(v_task,-1))
            return v_task.id, Task.TASK_FINISHED, self.workerService.doTask(task)
        except Exception, e:
            logger.error(e)
            return v_task.id, Task.TASK_ERROR,OtherFailure(e.message)

    def interrupt(self):
        try :
            self.workerService.stopTask()
        except Exception, e:
            logger.error(e)

    def __str__(self):
        return "%s:%s-%s"%(self.host,self.port,self.workerType)

#end
