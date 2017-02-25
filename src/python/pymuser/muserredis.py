import time
import redis
from muserenv import *
#from conf import MUSERConf
#redisStr = 'localhost:6379'
redisIpAndPort = muserenv.get_redis().split(":")

def operator_status(func):
    '''''get operatoration status
    '''
    def gen_status(*args, **kwargs):
        error, result = None, None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)
        return {'result': result, 'error':  error}
    return gen_status

class MuserRedis(object):
    def __init__(self):
        if not hasattr(MuserRedis, 'pool'):
            MuserRedis.create_pool()
        self._connection = redis.Redis(connection_pool = MuserRedis.pool)

    @staticmethod
    def create_pool():
        MuserRedis.pool = redis.ConnectionPool(
                host = redisIpAndPort[0],
                port = redisIpAndPort[1],
                db   = 0)

    @operator_status
    def set_data(self, key, value):
        '''''set data with (key, value)
        '''
        return self._connection.set(key, value)

    @operator_status
    def get_data(self, key):
        '''''get data by key
        '''
        return self._connection.get(key)

    @operator_status
    def del_data(self, key):
        '''''delete cache by key
        '''
        return self._connection.delete(key)

    @operator_status
    def muser_set_global(self, key, refTime, theValue):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        self._connection.zremrangebyscore("Muser%s"%(key),score,score)
        return self._connection.zadd("Muser%s"%(key), "%s||||%s"%(score,theValue), score)

    @operator_status
    def muser_remove_global(self, key, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        return self._connection.zremrangebyscore("Muser%s"%(key),score,score)

    @operator_status
    def muser_set_position(self, freq, refTime, theValue):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        self._connection.zremrangebyscore("MuserPosition%s"%(freq),score,score)
        return self._connection.zadd("MuserPosition%s"%(freq), "%s||||%s"%(score,theValue), score)
    @operator_status
    def muser_remove_position(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        return self._connection.zremrangebyscore("MuserPosition%s"%(freq),score,score)

    @operator_status
    def muser_set_flag(self, freq, refTime, theValue):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        self._connection.zremrangebyscore("MuserFlag%s"%(freq),score,score)
        return self._connection.zadd("MuserFlag%s"%(freq), "%s||||%s"%(score,theValue), score)

    @operator_status
    def muser_remove_flag(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        return self._connection.zremrangebyscore("MuserFlag%s"%(freq),score,score)

    @operator_status
    def muser_set_delay(self, freq, refTime, theValue):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        self._connection.zremrangebyscore("MuserDelay%s"%(freq),score,score)
        return self._connection.zadd("MuserDelay%s"%(freq), "%s||||%s"%(score,theValue), score)

    @operator_status
    def muser_remove_delay(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        return self._connection.zremrangebyscore("MuserDelay%s"%(freq),score,score)

    @operator_status
    def muser_set_status(self, freq, refTime, theValue):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        self._connection.zremrangebyscore("MuserStatus%s"%(freq),score,score)
        return self._connection.zadd("MuserStatus%s"%(freq), "%s||||%s"%(score,theValue), score)
    @operator_status
    def muser_remove_status(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        return self._connection.zremrangebyscore("MuserStatus%s"%(freq),score,score)

    @operator_status
    def muser_set_weather(self, refTime, theValue):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        self._connection.zremrangebyscore("MuserWeather",score,score)
        return self._connection.zadd("MuserWeather", "%s||||%s"%(score,theValue), score)
    @operator_status
    def muser_remove_weather(self, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        return self._connection.zremrangebyscore("MuserWeather",score,score)

    @operator_status
    def muser_get_position(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        lastIndex = self._connection.zcount("MuserPosition%s"%(freq),"-INF",score)-1
        smlscores = self._connection.zrangebyscore("MuserPosition%s"%(freq),"-INF",score,lastIndex,1,False)
        bigscores = self._connection.zrangebyscore("MuserPosition%s"%(freq),score,"+INF",0,1,False)
        return self.splitValue(smlscores,bigscores)

    @operator_status
    def muser_get_flag(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        lastIndex = self._connection.zcount("MuserFlag%s"%(freq),"-INF",score)-1
        smlscores = self._connection.zrangebyscore("MuserFlag%s"%(freq),"-INF",score,lastIndex,1,False)
        bigscores = self._connection.zrangebyscore("MuserFlag%s"%(freq),score,"+INF",0,1,False)
        return self.splitValue(smlscores,bigscores)

    @operator_status
    def muser_get_delay(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        lastIndex = self._connection.zcount("MuserDelay%s"%(freq),"-INF",score)-1
        smlscores = self._connection.zrangebyscore("MuserDelay%s"%(freq),"-INF",score,lastIndex,1,False)
        bigscores = self._connection.zrangebyscore("MuserDelay%s"%(freq),score,"+INF",0,1,False)
        return self.splitValue(smlscores,bigscores)

    @operator_status
    def muser_get_status(self, freq, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        lastIndex = self._connection.zcount("MuserStatus%s"%(freq),"-INF",score)-1
        smlscores = self._connection.zrangebyscore("MuserStatus%s"%(freq),"-INF",score,lastIndex,1,False)
        bigscores = self._connection.zrangebyscore("MuserStatus%s"%(freq),score,"+INF",0,1,False)
        return self.splitValue(smlscores,bigscores)

    @operator_status
    def muser_get_weather(self, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        lastIndex = self._connection.zcount("MuserWeather","-INF",score)-1
        smlscores = self._connection.zrangebyscore("MuserWeather","-INF",score,lastIndex,1,False)
        bigscores = self._connection.zrangebyscore("MuserWeather",score,"+INF",0,1,False)
        return self.splitValue(smlscores,bigscores)

    @operator_status
    def muser_get_global(self, key, refTime):
        score= time.mktime(time.strptime(refTime,"%Y-%m-%d %H:%M:%S"))
        lastIndex = self._connection.zcount("Muser%s"%(key),"-INF",score)-1
        smlscores = self._connection.zrangebyscore("Muser%s"%(key),"-INF",score,lastIndex,1,False)
        bigscores = self._connection.zrangebyscore("Muser%s"%(key),score,"+INF",0,1,False)
        return self.splitValue(smlscores,bigscores)

    def muser_remove(self,key,freq, refTime):
        if key == "Position" :
            return self.muser_remove_position(freq, refTime)
        elif key == "Flag" :
            return self.muser_remove_flag(freq, refTime)
        elif key == "Delay" :
            return self.muser_remove_delay(freq, refTime)


    def splitValue(self,smlscores,bigscores):
        smlscore,bigscore = "||||","||||"
        if smlscores:
            smlscore = smlscores[0]
        if bigscores:
            bigscore = bigscores[0]
        return (smlscore.split("||||")[1],bigscore.split("||||")[1])

    @operator_status
    def redis_version(self):
        return self._connection.info()['redis_version']


if __name__ == '__main__':
    print MuserRedis().muser_get_position("1","2015-12-01 00:00:00")
    print MuserRedis().muser_get_global("observatory","2015-12-01 00:00:00")['result']
