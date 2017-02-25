import sys,os
sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])
import json
import cPickle
import traceback

import wsgiref
import datetime
import mimetypes
import time

from api import *
from opencluster.item import Task,ObjValue

PWD = os.path.abspath(os.path.dirname(__file__))
PageSize = 20

urls = [
            "/res/(.*)", "StaticRes",
            "/muserres/(.*)", "MuserStaticRes",
            "/", "Index",
            "/antenna", "AntennaList",
            "/antennaOp", "AntennaOperation",
            "/calibration", "CalibrationList",
            "/calibrationOp","CalibrationOperation",
            "/config", "ConfigList",
            "/configInfo","ConfigInfo",
            "/configOp","ConfigOperation",
            "/dump2sqlite","Dump2Sqlite",
            "/dump2redis","Dump2Redis",
            "/integration", "IntegrationList",
            "/integrationOp","IntegrationOperation",
            "/integrationTasks","IntegrationTasks",
            "/integrationTaskOp","IntegrationTaskOperation",
            "/integrationResults","IntegrationResults",
            "/imaging", "ImagingList",
            "/download", "Download",
            "/imagingOp","ImagingOperation",
            "/weather", "WeatherList",
            "/weatherOp","WeatherOperation",
            "/rawfile","RawFile",
            "/about/(about)", "About",
            "/about/(contact)", "About",
]
urls.extend(apiUrls)
urls.extend(["*","WebHandler"])

servers = MUSERConf.getMuserconfServer()

if sys.platform == "win32":#for developer
    WebStaticFullPath = "E:\\work\\python\\opencluster\\opencluster\\ui\\res" #for developer
    WebMuserStaticFullPath = "E:\\work\\python\\opencluster\\muserconf\\res" #for developer
    SatelliteFileRoot = "E:\\astrodata"
    SatelliteFileOutputRoot = "E:\\astrodata"
else:
    WebStaticFullPath = MUSERConf.getWebStaticFullPath()
    WebMuserStaticFullPath = MUSERConf.getWebMuserStaticFullPath()
    SatelliteFileRoot = MUSERConf.getCalibrationFileRoot()
    SatelliteFileOutputRoot = MUSERConf.getCalibrationOutputRoot()


templates_path = "/templates/"
app = web.application(tuple(urls),globals())



folder_templates_full_path = PWD + templates_path

def render(params={},partial=True):
    global_vars = dict({'title':'MUSER Configuration'}.items() + params.items())
    if not partial:
        return web.template.render(folder_templates_full_path, globals=global_vars)
    else:
        return web.template.render(folder_templates_full_path, base='layout', globals=global_vars)

def titled_render(subtitle=''):
    subtitle = subtitle + ' - ' if subtitle else ''
    return render({'title': subtitle + ' MUSER Configuration'})

class WebServer(object) :
    def __init__(self,__server):
        server = __server.split(":")
        self.server = (server[0],int(server[1]))
        self.setup_session_folder_full_path()
        # web.config.static_path = PWD + Conf.getWebStaticPath()

    def start(self) :
        try :
            app.run(self.server)
        finally:
            kafkaUtil.close()


    def setup_session_folder_full_path(self):
        # global session
        #
        # if not web.config.get("_session"):
        #     folder_sessions_full_path = PWD + Conf.getWebSessionsPath()
        #     session = web.session.Session(app, web.session.DiskStore(folder_sessions_full_path), initializer = {"username": None})
        #     web.config._session = session
        # else:
        #     session = web.config._session
        pass



def server_static(dir, filename, mime_type=None):
    ''''' Serves a file statically '''
    if mime_type is None:
        mime_type = mimetypes.guess_type(filename)[0]
    web.header('Content-Type', bytes('%s' % mime_type))
    filename = os.path.join(dir, filename)
    if not os.path.exists(filename):
        raise web.NotFound()

    stat = os.stat(filename)
    web.header('Content-Length', '%s' % stat.st_size)
    web.header('Last-Modified', '%s' %
    web.http.lastmodified(datetime.datetime.fromtimestamp(stat.st_mtime)))
    return wsgiref.util.FileWrapper(open(filename, 'rb'), 16384)

class StaticRes(object):
    def GET(self, name):
        return server_static(WebStaticFullPath,name)

class MuserStaticRes(object):
    def GET(self, name):
        return server_static(WebMuserStaticFullPath,name)

class About(object):
    def GET(self, name):
        if name=="about" :
            return titled_render("About").about(about = "About")
        elif name == "contact" :
            return titled_render("Contact").about(about = "Contact")
        else:
            return web.NotFound()

class WebHandler(object):
    def GET(self):
        return "OpenCluster"
class ValueObj(object):
    def __init__(self):
        self.key = ""
        self.value = ""
        self.score = ""
class Index(object):
    def GET(self):
        try :
            items = {}
            mysqlState = ""

            try:
                mysqlState = MySQLConn.get().db.query("select version() as v")[0]
            except Exception, e:
                mysqlState = {"v":""}

            redisState = RedisCache().redis_version()
            redis_dump_ver = RedisCache().get_data("dumpver")

            rawfileCount = MySQLConn.get().db.query("select count(*) as d from t_raw_file")[0]
            rawfileLastTime = MySQLConn.get().db.query("select max(startTime) as d from t_raw_file")[0]

            integrations = MySQLConn.get().db.query("select count(*) as d,status from t_integration group by status")
            imagings = MySQLConn.get().db.query("select count(*) as d,status from t_imaging group by status")

            weather = MySQLConn.get().db.query("select theValue as d,refTime from p_weather order by  refTime DESC limit 1")[0]
            position1 = MySQLConn.get().db.query("select theValue as d from p_antenna_position where freq=1 order by  refTime DESC limit 1")[0]
            position2 = MySQLConn.get().db.query("select theValue as d from p_antenna_position where freq=2 order by  refTime DESC limit 1")[0]

            flag1 = MySQLConn.get().db.query("select theValue as d from p_antenna_flag where freq=1 order by  refTime DESC limit 1")[0]
            flag2 = MySQLConn.get().db.query("select theValue as d from p_antenna_flag where freq=2 order by  refTime DESC limit 1")[0]

            delay1 = MySQLConn.get().db.query("select theValue as d from p_antenna_delay where freq=1 order by  refTime DESC limit 1")[0]
            delay2 = MySQLConn.get().db.query("select theValue as d from p_antenna_delay where freq=2 order by  refTime DESC limit 1")[0]

            items["rawfileCount"] = rawfileCount.d
            items["rawfileLastTime"] = rawfileLastTime.d

            items["position"] = (position1.d,position2.d)
            items["delay"] = (delay1.d,delay2.d)
            items["flag"] = (flag1.d,flag2.d)
            items["weather"] = weather.d
            items["weatherTime"] = str(weather.refTime)

            items["integration"] = {"tasks":0,"running":0,"error":0,"finished":0}
            items["imaging"] = {"tasks":0,"running":0,"error":0,"finished":0}

            for item in integrations:
                items["integration"]["tasks"] += int(item.d)
                if item.status == 2:
                    items["integration"]["running"] = int(item.d)
                elif item.status == 3:
                    items["integration"]["finished"] = int(item.d)
                elif item.status == 4:
                    items["integration"]["error"] = int(item.d)

            for item in imagings:
                items["imaging"]["tasks"] += int(item.d)
                if item.status == 2:
                    items["imaging"]["running"] = int(item.d)
                elif item.status == 3:
                    items["imaging"]["finished"] = int(item.d)
                elif item.status == 4:
                    items["imaging"]["error"] = int(item.d)

            return titled_render().index(items = items,redisState=redisState["result"],mysqlState=mysqlState["v"],redis_dump_ver=redis_dump_ver,sqlite_dump_ver=get_sqlite_version())
        except Exception, e:
            return titled_render().error(error=e.message)

class AntennaList(object):
    def GET(self):
        try :
            req = web.input()
            key = str(req.key)
            tableName = ""
            if key == "Position" :
                tableName = "p_antenna_position"
            elif key == "Flag" :
                tableName = "p_antenna_flag"
            elif key == "Delay" :
                tableName = "p_antenna_delay"
            elif key == "Status" :
                tableName = "p_instrument_status"
            else:
                raise Exception("invalid key")

            if req.has_key("action") :
                id = req.id
                items = MySQLConn.get().db.query("select * from "+tableName+" where id=" + id)
                obj = items[0]
                item = {}
                item["refTime"] = str(obj.refTime)
                item["freq"] = str(obj.freq)
                item["theValue"] = str(obj.theValue)
                item["id"] = str(obj.id)
                return JsonUtil.successObjJson(item)

            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""

            if req.has_key("freq") :
                freq = str(req.freq)
            else :
                freq = "0"

            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d 00:00:00",time.localtime(time.time()-2592000))
            if not endTime :
                endTime = time.strftime("%Y-%m-%d 23:59:59",time.localtime(time.time()))

            items = MySQLConn.get().db.query("select * from "+tableName+" where TO_DAYS(refTime)>=TO_DAYS('" + beginTime +  "') and TO_DAYS(refTime)<=TO_DAYS('" + endTime +  "') and ("+freq+"=0 or freq="+freq+")")

            return titled_render().antenna(items = items,key = key,freq=freq,beginTime= beginTime,endTime=endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

class AntennaOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()
            key = str(req.key)
            action = str(req.action)

            tableName = ""
            if key == "Position" :
                tableName = "p_antenna_position"
            elif key == "Flag" :
                tableName = "p_antenna_flag"
            elif key == "Delay" :
                tableName = "p_antenna_delay"
            elif key == "Status" :
                tableName = "p_instrument_status"
            else:
                raise Exception("invalid key")

            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""
            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d",time.localtime(time.time())) + " 00:00:00"
            if not endTime :
                endTime = time.strftime("%Y-%m-%d",time.localtime(time.time())) + " 23:59:59"

            if req.has_key("freq") :
                freq = str(req.freq)
            else :
                freq = "0"

            if action == "remove" :
                id = str(req.id)
                MySQLConn.get().db.delete(tableName, where="id=" + id)

            if action == "add" :
                refTime = req.refTime
                theValue = req.theValue
                formFreq = req.formFreq
                sequence_id = MySQLConn.get().db.insert(tableName, refTime=refTime, theValue=theValue,freq = formFreq)

            if action == "edit" :
                id = str(req.id)
                refTime = req.refTime
                theValue = req.theValue
                formFreq = req.formFreq
                MySQLConn.get().db.update(tableName,where="id="+id,  refTime=refTime, theValue=theValue,freq = formFreq)

            web.seeother("antenna?freq="+freq+"&key=" + key + "&beginTime="+beginTime + "&endTime="+ endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

class CalibrationOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()

            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""
            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d",time.localtime(time.time())) + " 00:00:00"
            if not endTime :
                endTime = time.strftime("%Y-%m-%d",time.localtime(time.time())) + " 23:59:59"

            key = str(req.key)
            id = str(req.id)
            action = str(req.action)

            if action == "remove" :
                MySQLConn.get().db.delete('t_calibration', where="id=" + id)
            if action == "recalucation" :
                items = MySQLConn.get().db.query("select * from t_calibration where id=" + id)
                obj = items[0]
                result = ""
                # result = satellite_Vis.calCalibration(str(obj.freq),int(obj.offset or "0"),obj.priority)
                MySQLConn.get().db.update('t_calibration',where="id="+id, theValue=result,status=1)

            if action == "add" :
                ctime = req.ctime
                priority = req.priority
                offset = req.offset
                # value = req.value
                sequence_id = MySQLConn.get().db.insert('t_calibration', ctime=ctime,priority=priority, offset=offset, freq=key)

            if action == "edit" :
                ctime = req.ctime
                priority = req.priority
                offset = req.offset
                # value = req.value
                MySQLConn.get().db.update('t_calibration',where="id="+id, ctime=ctime,priority=priority,offset=offset, freq=key)

            web.seeother("calibration?key=" + key + "&beginTime="+beginTime + "&endTime="+ endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

class RawFile(object):
    def GET(self):
        try :
            req = web.input()
            if req.has_key("btime") :
                beginTime = str(req.btime)
            else:
                beginTime = ""
            if req.has_key("etime") :
                endTime = str(req.etime)
            else :
                endTime = ""
            if req.has_key("freq") :
                freq = str(req.freq)
            else :
                freq = "1"

            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            if not endTime :
                endTime = time.strftime("%Y-%m-%d%H:%M:%S",time.localtime(time.time()))


            datasets = MySQLConn.get().db.query("select * from t_raw_file where freq=" + freq + " and startTime>='" + beginTime + "' and startTime<='" + endTime +"'")
            items = []
            for it in datasets:
                item = {}
                item["id"] = it.id
                item["startTime"] = str(it.startTime)
                item["freq"] = it.freq
                item["path"] = str(it.path or "")
                items.append(item)

            return json.dumps(items)

        except Exception, e:
            return "{\"success\":false,\"error\":\""+e.message+"\"}"

class CalibrationList(object):
    def GET(self):
        try :
            req = web.input()
            if req.has_key("action") :
                id = req.id
                items = MySQLConn.get().db.query("select * from t_calibration where id=" + id)
                obj = items[0]
                item = {}
                item["ctime"] = str(obj.ctime)
                item["priority"] = obj.priority
                item["offset"] = int(obj.offset)
                item["status"] = obj.status
                item["freq"] = obj.freq
                item["description"] = str(obj.description or "")
                jsonStr = str(item)
                return jsonStr.replace("'","\"")

            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""

            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d",time.localtime(time.time()))
            if not endTime :
                endTime = time.strftime("%Y-%m-%d",time.localtime(time.time()))

            v_beginTime = time.mktime(time.strptime(beginTime,"%Y-%m-%d"))
            v_endTime = time.mktime(time.strptime(endTime,"%Y-%m-%d"))


            if not req.has_key("key") :
                raise Exception("lack key!!!")
            key = req.key

            items = MySQLConn.get().db.query("select * from t_calibration where TO_DAYS(ctime)>=TO_DAYS('" + beginTime +  "') and TO_DAYS(ctime)<=TO_DAYS('" + endTime +  "') and freq=" + str(key))

            return titled_render().calibration(items = items,key = key,beginTime= beginTime,endTime=endTime)
        except Exception, e:
            return titled_render().error(error=e.message)
class ConfigOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()

            key = str(req.key)
            id = str(req.id)
            action = str(req.action)

            if action == "remove" :
                MySQLConn.get().db.delete('t_config', where="id=" + id)

            if action == "add" :
                keyName = req.keyName
                createTime = req.createTime
                theValue = req.theValue
                sequence_id = MySQLConn.get().db.insert('t_config', keyName=keyName,theValue=theValue,createTime=createTime)

            if action == "edit" :
                keyName = req.keyName
                theValue = req.theValue
                createTime = req.createTime
                MySQLConn.get().db.update('t_config',where="id="+id, keyName=keyName,theValue=theValue,createTime=createTime)


            web.seeother("config?key=" + key)
        except Exception, e:
            return titled_render().error(error=e.message)
class ConfigInfo(object):
    def GET(self):
        try :
            req = web.input()
            id = req.id
            items = MySQLConn.get().db.query("select * from t_config where id=" + id)
            obj = items[0]
            item = {}
            item["keyName"] = str(obj.keyName)
            item["theValue"] = str(obj.theValue or "")
            item["createTime"] = str(obj.createTime or "")

            jsonStr = str(item)
            return jsonStr.replace("'","\"")
        except Exception, e:
            return titled_render().error(error=e.message)
class ConfigList(object):
    def GET(self):
        try :
            req = web.input()
            key = req.key

            keyNames = MySQLConn.get().db.query("select * from t_config_key")
            items = MySQLConn.get().db.query("select * from t_config where keyName like '%" + key +  "%'")

            return titled_render().config(items = items,key = key,keyNames=keyNames)
        except Exception, e:
            return titled_render().error(error=e.message)

#--------------------------integration---------------begin---------------------------
class IntegrationResults(object):
    def GET(self):
        try :
            req = web.input()
            action = str(req.action)
            id = str(req.id)

            if action == "results" :
                items = MySQLConn.get().db.query("select * from t_integration_task where status=2 and int_id=" + id)
                results = []
                for it in items :
                    results.extend(str(it.results).split(","))

                integrations = MySQLConn.get().db.query("select * from t_integration where id=" + id)
                obj = integrations[0]

                return titled_render().integrationResults(items = results,integration = obj)

            if action=="download":
                web.header('Content-Type','application/octet-stream')
                web.header('Content-disposition', 'attachment; filename=%s' % id.split("/")[-1])
                try :
                    f = open(id,"rb")
                    return f.read()
                finally:
                    if f :
                        f.close()

            web.seeother("integrationTasks?id="+id)
        except Exception, e:
            traceback.print_exc()
            return titled_render().error(error=e.message)
class Download(object):
    def GET(self):
        try :
            req = web.input()
            fileName = str(req.file)

            web.header('Content-Type','application/octet-stream')
            web.header('Content-disposition', 'attachment; filename=%s' % fileName.split("/")[-1])
            f = None
            try :
                f = open(fileName,"rb")
                return f.read()
            finally:
                if f :
                    f.close()

        except Exception, e:
            traceback.print_exc()
            return titled_render().error(error=e.message)

class IntegrationTasks(object):
    def GET(self):
        try :
            req = web.input()
            action = str(req.action)
            id = str(req.id)
            if action == "detail":
                integration = {}
                integrations = MySQLConn.get().db.query("select * from t_integration where id=" + id)
                obj = integrations[0]
                item = {}
                item["beginTime"] = str(obj.beginTime)
                item["endTime"] = str(obj.endTime)
                item["status"] = obj.status
                item["freq"] = obj.freq
                item["seconds"] = obj.seconds
                item["format"] = str(obj.format)
                item["job_id"] = str(obj.job_id)
                item["is_specified_file"] = obj.is_specified_file
                item["specified_file"] =str(obj.specified_file or "")
                item["big_file"] = str(obj.big_file or "")
                item["description"] = str(obj.description or "")
                item["task_num"] = obj.task_num
                item["results"] = str(obj.results or "")
                item["task_percent"] = 0
                item["task_remain"] = 0


                items = MySQLConn.get().db.query("select count(*) as d,repeat_num from t_integration_task where (status=2 or status=3) and int_id="+id)
                if len(items)>0 and item["task_num"] > 0:
                    tasksNum = items[0]
                    item["task_percent"] = int((float(tasksNum.d or 0)/item["task_num"])*100)
                    item["task_remain"] = int(item["seconds"]*60*float(tasksNum.repeat_num or 0)*(item["task_num"]-float(tasksNum.d or 0))) + 60

                    print tasksNum
                    print item["task_percent"]

                tasks = MySQLConn.get().db.query("select * from t_integration_task where int_id="+id)
                return titled_render().integrationTasks(items = tasks,integration = item)
            web.seeother("integrationTasks?id="+id)
        except Exception, e:
            traceback.print_stack()
            return titled_render().error(error=e.message)

class IntegrationTaskOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()
            id = str(req.id)
            action = str(req.action)
            int_id = "0"
            if action == "reset" :
                MySQLConn.get().db.update('t_integration_task',where="task_id='" + id + "'", status=1)
                items = MySQLConn.get().db.query("select * from t_integration_task where task_id='" + id + "'")

                if len(items)>0 :
                    obj = items[0]
                    int_id = str(obj.int_id)

                    task_data = ObjValue()
                    task_data.setObj("timeStr",str(obj.timeStr))
                    task_data.setObj("freq",obj.freq)
                    task_data.setObj("integralNumber",obj.int_number)
                    task_data.setObj("repeat",obj.repeat_num)

                    task = Task(id=id,data=task_data,\
                                workerClass="integrationWorker.IntegrationWorker",workDir = os.path.dirname(os.path.abspath(__file__)) + "/../ocscripts",priority=3,\
                                resources={"cpus":1,"mem":500,"gpus":0},\
                                warehouse="OpenCluster3",\
                                jobName=str(obj.job_id))

                    kafkaUtil.produceTasks([task])
            web.seeother("integrationTasks?action=detail&id=" + int_id)

        except Exception, e:
            traceback.print_stack()
            return titled_render().error(error=e.message)

class IntegrationOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()
            id = str(req.id)
            action = str(req.action)
            beginTime = req.beginTime
            endTime = req.endTime
            if action == "remove" :
                MySQLConn.get().db.delete('t_integration', where="id=" + id)

            if action == "detail" :
                items = MySQLConn.get().db.query("select * from t_integration_task where int_id=" + id)
                return titled_render().integrationTasks(items = items,beginTime = beginTime,endTime=endTime)

            if action == "add" :
                fBeginTime = req.fBeginTime
                fEndTime = req.fEndTime
                seconds = float(req.seconds)
                freq = int(req.freq)
                description = req.description
                format = req.format
                is_specified_file = int(req.is_specified_file)
                specified_file = req.specified_file

                sequence_id = MySQLConn.get().db.insert('t_integration', status=0, beginTime=fBeginTime, endTime=fEndTime, \
                                        seconds=seconds, freq=freq, format=format, job_id=time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())),\
                                        is_specified_file=is_specified_file, specified_file=specified_file, \
                                        createTime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())),\
                                        description=description)

            if action == "edit" :
                fBeginTime = req.fBeginTime
                fEndTime = req.fEndTime
                seconds = req.seconds
                freq = int(req.freq)
                description = req.description
                format = req.format
                is_specified_file = int(req.is_specified_file)
                specified_file = req.specified_file

                MySQLConn.get().db.update('t_integration',where="id="+id, beginTime=fBeginTime,endTime=fEndTime,seconds=seconds,freq=freq,format=format, \
                                        is_specified_file=is_specified_file, specified_file=specified_file, \
                                        description=description)

            if action == "begin" :
                MySQLConn.get().db.delete('t_integration_task', where="int_id=" + id)
                MySQLConn.get().db.update('t_integration',where="id="+id, status=1)

            web.seeother("integration?beginTime="+beginTime + "&endTime="+ endTime)
        except Exception, e:
            traceback.print_exc()
            return titled_render().error(error=e.message)

def pagerBuild(totalCount,page) :
    totalPage = int(totalCount/PageSize)

    if totalCount%PageSize or totalCount == 0:
        totalPage += 1

    if page > totalPage:
        page = totalPage

    if page < 1:
        page = 1

    iBegin = (page-1)*PageSize
    iEnd = page*PageSize

    if iBegin > totalCount:
        iBegin = totalCount

    if iEnd > totalCount:
        iEnd = totalCount

    return {"page":page,"totalPage":totalPage,"iBegin":iBegin,"iEnd":iEnd,"pageSize":PageSize,"totalCount":totalCount}

class IntegrationList(object):
    def GET(self):
        try :
            req = web.input()
            if req.has_key("action") :
                id = req.id
                items = MySQLConn.get().db.query("select * from t_integration where id=" + id)
                obj = items[0]
                item = {}
                item["beginTime"] = str(obj.beginTime)
                item["endTime"] = str(obj.endTime)
                item["status"] = obj.status
                item["freq"] = obj.freq
                item["seconds"] = obj.seconds
                item["format"] = str(obj.format)
                item["is_specified_file"] = obj.is_specified_file
                item["specified_file"] =str(obj.specified_file or "")
                item["big_file"] = str(obj.big_file or "")
                item["description"] = str(obj.description or "")
                jsonStr = str(item)
                return jsonStr.replace("'","\"")


            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""



            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d 00:00:00",time.localtime(time.time()-2592000))
            if not endTime :
                endTime = time.strftime("%Y-%m-%d 23:59:59",time.localtime(time.time()))

            totalCount_cur = MySQLConn.get().db.query("select count(*) as d from t_integration where TO_DAYS(createTime)>=TO_DAYS('" + beginTime +  "') and TO_DAYS(createTime)<=TO_DAYS('" + endTime +  "')")[0]
            totalCount = totalCount_cur.d

            if req.has_key("page") :
                page = int(req.page)
            else :
                page = 1

            pager = pagerBuild(totalCount,page)

            items = MySQLConn.get().db.query("select * from t_integration where TO_DAYS(createTime)>=TO_DAYS('" + beginTime +  "') and TO_DAYS(createTime)<=TO_DAYS('" + endTime +  "') order by createTime DESC limit "+str(pager["iBegin"])+","+str(pager["iEnd"]))
            return titled_render().integration(items = items,beginTime = beginTime,endTime=endTime,pager=pager)
        except Exception, e:
            return titled_render().error(error=e.message)

#--------------------------integration---------------end-----------------------------

#--------------------------Weather---------------begin-----------------------------
class WeatherOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()

            action = str(req.action)
            beginTime = req.beginTime
            endTime = req.endTime

            if action == "remove" :
                id = str(req.id)
                MySQLConn.get().db.delete('p_weather', where="id=" + id)

            if action == "add" :
                refTime = req.refTime
                theValue = req.theValue
                sequence_id = MySQLConn.get().db.insert('p_weather', refTime=refTime, theValue=theValue)

            if action == "edit" :
                id = str(req.id)
                refTime = req.refTime
                theValue = req.theValue
                MySQLConn.get().db.update('p_weather',where="id="+id,  refTime=refTime, theValue=theValue)

            web.seeother("weather?beginTime="+beginTime + "&endTime="+ endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

class WeatherList(object):
    def GET(self):
        try :
            req = web.input()
            if req.has_key("action") :
                id = req.id
                items = MySQLConn.get().db.query("select * from p_weather where id=" + id)
                obj = items[0]
                item = {}
                item["refTime"] = str(obj.refTime)
                item["theValue"] = str(obj.theValue)
                item["id"] = str(obj.id)
                return JsonUtil.successObjJson(item)

            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""

            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d 00:00:00",time.localtime(time.time()-2592000))
            if not endTime :
                endTime = time.strftime("%Y-%m-%d 23:59:59",time.localtime(time.time()))

            items = MySQLConn.get().db.query("select * from p_weather where TO_DAYS(refTime)>=TO_DAYS('" + beginTime +  "') and TO_DAYS(refTime)<=TO_DAYS('" + endTime +  "')")
            return titled_render().weather(items = items,beginTime = beginTime,endTime=endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

#--------------------------Weather---------------end-----------------------------

#--------------------------imaging---------------begin---------------------------

class ImagingOperation(object):
    def GET(self):
        self.operation()
    def POST(self):
        self.operation()
    def operation(self):
        try :
            req = web.input()
            id = str(req.id)
            action = str(req.action)
            beginTime = req.beginTime
            endTime = req.endTime
            if action == "remove" :
                MySQLConn.get().db.delete('t_imaging', where="id=" + id)

            if action == "add" :
                fBeginTime = req.fBeginTime
                fEndTime = req.fEndTime
                seconds = req.seconds
                freq = req.freq
                description = req.description
                format = req.format
                is_specified_file = req.is_specified_file
                specified_file = req.specified_file
                gen_result = req.gen_result
                with_axis = req.with_axis

                sequence_id = MySQLConn.get().db.insert('t_imaging', status=0, beginTime=fBeginTime, endTime=fEndTime, \
                                        seconds=seconds, freq=freq, format=format, job_id=time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())),\
                                        is_specified_file=is_specified_file, specified_file=specified_file, \
                                        gen_result=gen_result, with_axis=with_axis, \
                                        createTime = time.strftime("%Y-%m-%d H%:M:%S",time.localtime(time.time())),\
                                        description=description)

            if action == "edit" :
                fBeginTime = req.fBeginTime
                fEndTime = req.fEndTime
                seconds = req.seconds
                freq = req.freq
                description = req.description
                format = req.format
                is_specified_file = req.is_specified_file
                specified_file = req.specified_file
                gen_result = req.gen_result
                with_axis = req.with_axis
                MySQLConn.get().db.update('t_imaging',where="id="+id, beginTime=fBeginTime,endTime=fEndTime,seconds=seconds,freq=freq,format=format, \
                                        is_specified_file=is_specified_file, specified_file=specified_file, \
                                    gen_result=gen_result, with_axis=with_axis, \
                                    description=description)

            if action == "begin" :
                MySQLConn.get().db.update('t_imaging',where="id="+id, status=1)

            web.seeother("imaging?beginTime="+beginTime + "&endTime="+ endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

class ImagingList(object):
    def GET(self):
        try :
            req = web.input()
            if req.has_key("action") :
                id = req.id
                items = MySQLConn.get().db.query("select * from t_imaging where id=" + id)
                obj = items[0]
                item = {}
                item["beginTime"] = str(obj.beginTime)
                item["endTime"] = str(obj.endTime)
                item["status"] = obj.status
                item["freq"] = obj.freq
                item["seconds"] = obj.seconds
                item["format"] = str(obj.format)
                item["is_specified_file"] = obj.is_specified_file
                item["specified_file"] =str(obj.specified_file or "")
                item["big_file"] = str(obj.big_file or "")
                item["description"] = str(obj.description or "")
                item["gen_result"] = str(obj.gen_result or "")
                item["with_axis"] = obj.with_axis
                jsonStr = str(item)
                return jsonStr.replace("'","\"")


            if req.has_key("beginTime") :
                beginTime = str(req.beginTime)
            else:
                beginTime = ""
            if req.has_key("endTime") :
                endTime = str(req.endTime)
            else :
                endTime = ""

            if not beginTime :
                beginTime = time.strftime("%Y-%m-%d 00:00:00",time.localtime(time.time()-2592000))
            if not endTime :
                endTime = time.strftime("%Y-%m-%d 23:59:59",time.localtime(time.time()))

            items = MySQLConn.get().db.query("select * from t_imaging where TO_DAYS(createTime)>=TO_DAYS('" + beginTime +  "') and TO_DAYS(createTime)<=TO_DAYS('" + endTime +  "')")
            return titled_render().imaging(items = items,beginTime = beginTime,endTime=endTime)
        except Exception, e:
            return titled_render().error(error=e.message)

#--------------------------imaging---------------end-----------------------------

class Dump2Sqlite(object):
    def GET(self):
        try :
            dump2sqlite_config()
            dump2sqlite_delay()
            dump2sqlite_flag()
            dump2sqlite_position()
            dump2sqlite_status()
            dump2sqlite_weather()
            t = update_sqlite_version()
            web.seeother("/")
        except Exception, e:
            return titled_render().error(error=e.message)

class Dump2Redis(object):
    def GET(self):
        try :
            dump2redis_config()
            dump2redis_position()
            dump2redis_delay()
            dump2redis_flag()
            dump2redis_status()
            dump2redis_weather()

            RedisCache().set_data("dumpver",int(time.time()))

            web.seeother("/")
        except Exception, e:
            return titled_render().error(error=e.message)

def dump2sqlite_config():
    import sqlite3
    cx = sqlite3.connect(os.path.join(MUSERConf.getSqliteDir(),"muser.db"))
    try:
        cx.execute("DELETE from t_global")
        items = MySQLConn.get().db.query("select * from t_config")
        list = []
        for item in items:
            t = (str(item.keyName),str(item.theValue),str(item.createTime))
            list.append(t)
        cx.executemany("insert into t_global values(?,?,?)",list)
        cx.commit()
    finally:
        cx.close()

def dump2redis_config():
    items = MySQLConn.get().db.query("select * from t_config")

    keyItems = MySQLConn.get().db.query("select * from t_config_key")

    for kitem in keyItems:
        RedisCache().del_data("Muser%s"%(str(kitem.keyName)))

    for item in items:
        RedisCache().muser_set_global(str(item.keyName),str(item.createTime),str(item.theValue))

def dump2sqlite_position():
    import sqlite3
    cx = sqlite3.connect(os.path.join(MUSERConf.getSqliteDir(),"muser.db"))
    try:
        cx.execute("DELETE from p_antenna_position")
        items = MySQLConn.get().db.query("select * from p_antenna_position")
        list = []
        for item in items:
            t = (str(item.freq),str(item.theValue),str(item.refTime))
            list.append(t)
        cx.executemany("insert into p_antenna_position values(?,?,?)",list)
        cx.commit()
    finally:
        cx.close()

def dump2redis_position():
    items = MySQLConn.get().db.query("select * from p_antenna_position")
    RedisCache().del_data("MuserPosition1")
    RedisCache().del_data("MuserPosition2")
    for item in items:
        RedisCache().muser_set_position(str(item.freq),str(item.refTime),str(item.theValue))

def dump2sqlite_delay():
    import sqlite3
    cx = sqlite3.connect(os.path.join(MUSERConf.getSqliteDir(),"muser.db"))
    try:
        cx.execute("DELETE from p_antenna_delay")
        items = MySQLConn.get().db.query("select * from p_antenna_delay")
        list = []
        for item in items:
            t = (str(item.freq),str(item.theValue),str(item.refTime))
            list.append(t)
        cx.executemany("insert into p_antenna_delay values(?,?,?)",list)
        cx.commit()
    finally:
        cx.close()

def dump2redis_delay():
    items = MySQLConn.get().db.query("select * from p_antenna_delay")
    RedisCache().del_data("MuserDelay1")
    RedisCache().del_data("MuserDelay2")
    for item in items:
        RedisCache().muser_set_delay(str(item.freq),str(item.refTime),str(item.theValue))

def dump2sqlite_flag():
    import sqlite3
    cx = sqlite3.connect(os.path.join(MUSERConf.getSqliteDir(),"muser.db"))
    try:
        cx.execute("DELETE from p_antenna_flag")
        items = MySQLConn.get().db.query("select * from p_antenna_flag")
        list = []
        for item in items:
            t = (str(item.freq),str(item.theValue),str(item.refTime))
            list.append(t)
        cx.executemany("insert into p_antenna_flag values(?,?,?)",list)
        cx.commit()
    finally:
        cx.close()

def dump2redis_flag():
    items = MySQLConn.get().db.query("select * from p_antenna_flag")
    RedisCache().del_data("MuserFlag1")
    RedisCache().del_data("MuserFlag2")
    for item in items:
        RedisCache().muser_set_flag(str(item.freq),str(item.refTime),str(item.theValue))

def dump2sqlite_status():
    import sqlite3
    cx = sqlite3.connect(os.path.join(MUSERConf.getSqliteDir(),"muser.db"))
    try:
        cx.execute("DELETE from p_instrument_status")
        items = MySQLConn.get().db.query("select * from p_instrument_status")
        list = []
        for item in items:
            t = (str(item.freq),str(item.theValue),str(item.refTime))
            list.append(t)
        cx.executemany("insert into p_instrument_status values(?,?,?)",list)
        cx.commit()
    finally:
        cx.close()

def dump2redis_status():
    items = MySQLConn.get().db.query("select * from p_instrument_status")
    RedisCache().del_data("MuserStatus1")
    RedisCache().del_data("MuserStatus2")
    for item in items:
        RedisCache().muser_set_status(str(item.freq),str(item.refTime),str(item.theValue))

def dump2sqlite_weather():
    import sqlite3
    cx = sqlite3.connect(os.path.join(MUSERConf.getSqliteDir(),"muser.db"))
    try:
        cx.execute("DELETE from p_weather")
        items = MySQLConn.get().db.query("select * from p_weather")
        list = []
        for item in items:
            t = (str(item.theValue),str(item.refTime))
            list.append(t)
        cx.executemany("insert into p_weather values(?,?)",list)
        cx.commit()
    finally:
        cx.close()

def dump2redis_weather():
    items = MySQLConn.get().db.query("select * from p_weather")
    RedisCache().del_data("MuserWeather")
    for item in items:
        RedisCache().muser_set_weather(str(item.refTime),str(item.theValue))

def update_sqlite_version():
    file_version = None
    try:
        file_version = open(os.path.join(MUSERConf.getSqliteDir(),"muser.db.version"),'w')
        t = int(time.time())
        file_version.write(str(t))
        return t
    finally:
        if file_version is not None:
            file_version.close()

def get_sqlite_version():
    file_version = None
    try:
        file_version = open(os.path.join(MUSERConf.getSqliteDir(),"muser.db.version"),'r')
        t = file_version.read()
        return t
    except:
        print os.path.join(MUSERConf.getSqliteDir(),"muser.db.version")
        return "ERROR"
    finally:
        if file_version is not None:
            file_version.close()
#--------------------------dump to sqlite---------------end-----------------------------
if __name__ == "__main__" :
    thisServer = WebServer(servers)
    thisServer.start()