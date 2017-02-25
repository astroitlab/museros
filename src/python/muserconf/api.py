import web
import redis
from jsonutil import JsonUtil
from ocscripts.conf import MUSERConf
from ocscripts.kafkaUtils import KafkaUtils
from ocscripts.redisClient import RedisCache

kafkaUtil = KafkaUtils(MUSERConf.getKafka())
apiUrls = [
    "/api/hello", "Hello",
    "/api/health", "Health",
    "/api/config/list", "ConfigListApi",
]
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

class RedisPool(object):
    _instance = None
    def __init__(self):
        redisStr = MUSERConf.getRedis()
        redisIpAndPort = redisStr.split(":")
        self.pool = redis.ConnectionPool(host=redisIpAndPort[0], port=int(redisIpAndPort[1]), db=0)
    @staticmethod
    def get():
        if RedisPool._instance is None:
            RedisPool._instance = RedisPool().pool
        return RedisPool._instance

class MySQLConn(object):
    _instance = None
    def __init__(self):
        mysqlStr = MUSERConf.getMySQL()
        mysqlUrls = mysqlStr.split(",")
        mysqlIpAndPort = mysqlUrls[0].split(":")
        self.db = web.database(dbn='mysql', db=mysqlUrls[1], user=mysqlUrls[2],passwd=mysqlUrls[3],host=mysqlIpAndPort[0],port=int(mysqlIpAndPort[1]))
    @staticmethod
    def get():
        if MySQLConn._instance is None:
            MySQLConn._instance = MySQLConn()
        return MySQLConn._instance

web.config.debug = True

class Hello(object):
    def GET(self):
        return "hello "
    def PUT(self):
        return "hello "

class Health(object):
    def GET(self):
        return "ok"
    def PUT(self):
        return "ok"

class ConfigListApi(object):
    def GET(self):
        try :
            req = web.input()
            key = ""
            if req.has_key("key"):
                key = req.key
            datasets = MySQLConn.get().db.query("select * from t_config where keyName like '%" + key +  "%'")
            items = []
            for it in datasets:
                item = {}
                item["id"] = it.id
                item["keyName"] = str(it.keyName)
                item["createTime"] = str(it.createTime)
                item["theValue"] = str(it.theValue)
                items.append(item)

            return JsonUtil.successObjJson(items)
        except Exception, e:
            return JsonUtil.errorMsgJson(str(e.message))

    def PUT(self):
        pass

    def DELETE(self):
        pass