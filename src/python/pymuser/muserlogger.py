import logging
import sys, os
#from muserenv import *

class MuserLogger():

    def __init__(self):
        self.logger = logging.getLogger('muser')
        #muserenv = MuserEnv()

    def setLevel(self, level):
        level = level.upper().strip()
        if level=='DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif level=='INFO':
            self.logger.setLevel(logging.INFO)
        elif level=='WARNING':
            self.logger.setLevel(logging.WARNING)
        elif level == 'NOTSET':
            self.logger.setLevel(logging.NOTSET)
        elif level == 'CRITICAL':
            self.logger.setLevel(logging.CRITICAL)
        elif level == 'ERROR':
            self.logger.setLevel(logging.ERROR)

    def setLogger(self, file, levelfile=logging.DEBUG, levelconsole=logging.INFO, filelog=True, consolelog=True):
        #logger = logging.getLogger('muser')
        if self.logger.handlers:
            self.logger.removeHandler( i for i in self.logger.handlers)
        self.logger.setLevel(logging.DEBUG)
        if filelog:
            rollfdlr = logging.handlers.RotatingFileHandler(filename=file,mode='a', maxBytes=10*1024*1024,backupCount=1)
            formatter = logging.Formatter('[%(asctime)s](%(levelname)s)-%(filename)s:%(lineno)s: %(message)s')
            rollfdlr.setFormatter(formatter)
            rollfdlr.setLevel(levelfile)
            self.logger.addHandler(rollfdlr)
        if consolelog:
            consoledlr = logging.StreamHandler(sys.stdout)
            formatter_console = logging.Formatter('%(message)s')
            consoledlr.setFormatter(formatter_console)
            consoledlr.setLevel(levelconsole)
            self.logger.addHandler(consoledlr)

    def origin(self,Message = None, level="DEBUG"):
        Msg = "Function Origin: "+Message.strip()
        level = level.upper().strip()
        if level=='DEBUG':
            self.logger.debug(Msg)
        elif level=='INFO':
            self.logger.info(Msg)
        elif level=='WARNING':
            self.logger.warning(Msg)
        elif level=='ERROR':
            self.logger.error(Msg)
        elif level == 'CRITICAL':
            self.logger.critical(Msg)


    def post(self,Msg, level="DEBUG"):
        level = level.upper().strip()
        if level=='DEBUG':
            self.logger.debug(Msg)
        elif level=='INFO':
            self.logger.info(Msg)
        elif level=='WARNING':
            self.logger.warning(Msg)
        elif level=='ERROR':
            self.logger.error(Msg)
        elif level == 'CRITICAL':
            self.logger.critical(Msg)

    def info(self, Msg):
        self.logger.info(Msg)

    def debug(self,Msg):
        self.logger.debug(Msg)

    def warning(self, Msg):
        self.logger.warning(Msg)

    def error(self, Msg):
        self.logger.error(Msg)

    def critical(self, Msg):
        self.logger.critical(Msg)


#logger = MuserLogger()