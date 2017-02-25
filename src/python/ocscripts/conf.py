import os,sys
import ConfigParser
import logging
import logging.config

class MUSERConf(object):
    cf = ConfigParser.ConfigParser()
    configFilePath = os.path.join(os.path.dirname(__file__),"muser-config.ini")
    cf.read(configFilePath)

    def __init__(self):
        if not MUSERConf.cf :
            MUSERConf.cf = ConfigParser.ConfigParser()
            MUSERConf.cf.read(MUSERConf.configFilePath)
            
    @classmethod
    def setConfigFile(cls, filePath):
        cls.configFilePath = filePath
        cls.cf = ConfigParser.ConfigParser()
        cls.cf.read(MUSERConf.configFilePath)

    @classmethod
    def getWebStaticFullPath(cls):
        return cls.cf.get("webapp", "static_full_path")

    @classmethod
    def getWebMuserStaticFullPath(cls):
        return cls.cf.get("webapp", "muser_static_full_path")

    @classmethod
    def getWebTemplatesPath(cls):
        return cls.cf.get("webapp", "templates_path")

    @classmethod
    def getMuserconfServer(cls):
        return cls.cf.get("webapp", "muserconfserver")

    @classmethod
    def getMySQL(cls):
        return cls.cf.get("db", "mysql")

    @classmethod
    def getRedis(cls):
        return cls.cf.get("db", "redis")

    @classmethod
    def getKafka(cls):
        return cls.cf.get("db", "kafka")

    @classmethod
    def getSqliteDir(cls):
        return cls.cf.get("db", "sqliteDir")

    @classmethod
    def getRtServer(cls):
        return cls.cf.get("rtserver", "rtserver")
    @classmethod
    def getCalibrationFileRoot(cls):
        return cls.cf.get("calibration", "calibrationFileRoot")
    @classmethod
    def getCalibrationOutputRoot(cls):
        return cls.cf.get("calibration", "calibrationOutputRoot")

    @classmethod
    def getGeneratePhotoRoot(cls):
        return cls.cf.get("files", "generatePhotoRoot")

    @classmethod
    def getZipRoot(cls):
        return cls.cf.get("files", "zipRoot")
