import sys
import os
import logging
import time
import redis

sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
from opencluster.service import Service
from opencluster.errors import *
from conf import MUSERConf

logger = logging.getLogger(__name__)

redisStr = MUSERConf.getRedis()
redisIpAndPort = redisStr.split(":")

pool = redis.ConnectionPool(host=redisIpAndPort[0], port=int(redisIpAndPort[1]), db=0)

class AntennaService(Service) :

    def __init__(self):
        super(AntennaService,self).__init__("antennaService")

    def doTask(self, task_data): #task_data is ObjValue

        key = task_data.getObj("key")
        timeStr = task_data.getObj("timeStr")
        logger.info(self.name + " get frame :" + str(key) + ":" + key + ",timeStr:" + timeStr)

        v_time= time.mktime(time.strptime(timeStr,"%Y-%m-%d %H:%M:%S"))

        r = redis.Redis(connection_pool=pool)

        if not key:
            raise Exception("lack key!!!")

        # valscore = r.zrange(key,0,-1,False,True)
        valscore_big = r.zrangebyscore(key,v_time,"+inf",0,1,True)
        valscore_sml = r.zrangebyscore(key,"-inf",v_time,-1,0,True)
        items = []
        for vscore in [valscore_sml,valscore_big] :
            vo = object()
            if vscore :
                vo.key = key
                vo.time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(vscore[1]))
                vo.score = vscore[1]
                vo.value = vscore[0]
                items.append(vo)
        return items

if __name__ == "__main__" :
    try:
        if len(sys.argv) != 2 :
            print "Usage: python %s localIP"%(sys.argv[0])
            sys.exit(1)

        wk = AntennaService()
        wk.waitWorking(sys.argv[1])

    except Exception,e :
        logger.error(e)
