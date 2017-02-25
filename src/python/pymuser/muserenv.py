#! /usr/bin/env python
# encoding: utf-8
"""
muserself.py
=====================
Environment MUSER-I/II
"""

import sys, os, datetime, time, math
import logging
import logging.handlers
from xml.dom.minidom import parse, parseString


class MuserEnv(object):
    __instance = None
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(MuserEnv,cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if(self.__initialized): return
        self.__initialized = True
        currentdir = os.path.abspath(os.path.dirname(__file__))
        if os.environ.has_key('MUSERHOME'):
            homedir = os.getenv('MUSERHOME')+os.path.sep+'resource'+os.path.sep+'xml'
            conffile = os.path.join(homedir,'system.xml')
        else:
            conffile = os.path.join(currentdir,'env.xml')
        print "Environment file: %s" % conffile
        parentdir = os.path.abspath(os.path.join(currentdir, os.path.pardir))
        pparentdir = os.path.abspath(os.path.join(parentdir, os.path.pardir))
        if currentdir.find('museros') == -1:
            print("Cannot locate proper directory")
            exit(0)
        path1 = currentdir[0:currentdir.find('museros') + 7]
        self.MUSER_HOME = path1
        self.datasource = ''
        self.sqlite = ''
        self.redis = ''

        if not os.path.exists(conffile):
            if self.MUSER_HOME.find('opt') == -1:
                self.MUSER_WORK = os.path.join(pparentdir, "work")
                self.MUSER_ARfileCH = os.path.join(pparentdir, "archive")
            else:
                self.MUSER_WORK = os.path.join('/opt', "work")
                self.MUSER_ARCH = os.path.join('/opt', "archive")
        else:
            dom1 = parse(conffile) # parse an XML file by name
            config_element = dom1.getElementsByTagName("config")[0]

            servers = config_element.getElementsByTagName("work")
            for server in servers:
                self.MUSER_WORK = self.getText(server.childNodes)

            servers = config_element.getElementsByTagName("archive")
            for server in servers:
                self.MUSER_ARCH = self.getText(server.childNodes)

            ds = config_element.getElementsByTagName("datasource")
            for d in ds:
                self.datasource = self.getText(d.childNodes).lower()

            ds = config_element.getElementsByTagName("sqlite")
            for d in ds:
                self.sqlite = self.getText(d.childNodes).lower()

            ds = config_element.getElementsByTagName("redis")
            for d in ds:
                self.redis = self.getText(d.childNodes).lower()

            #nodelist = config_element.getElementsByTagName("archive")
            #self.MUSER_ARCH = nodelist.childNodes[0].data

        #print "ARCHIVE Dir:", self.MUSER_ARCH
        self.muser_name = "MUSER"
        self.muser_ant = []
        self.antenna_loaded = False
        if not os.environ.has_key('MUSERHOME'):
            os.environ['MUSERHOME'] = self.MUSER_HOME
        # if len(logging.getLogger('muser').handlers) == 0 :
        #     self.set_logger()

    def get_datasource(self):

        return self.datasource

    def get_sqlite(self):
        return self.sqlite

    def get_redis(self):
        return self.redis

    def getText(self, nodelist):
        rc = ""
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc = rc + node.data
        return rc

    def get_home_dir(self):
        return self.MUSER_HOME

    def get_work_dir(self):
        return self.MUSER_WORK

    def get_archive_dir(self):
        return self.MUSER_ARCH

    def file_exist(self, file_name):
        if not os.path.exists(file_name):
            return False
        else:
            return True

    def cal_file(self, sub_array, year, month, day, cal):
        file_name = ('%04d%02d%02d-%1d') % (year, month, day, cal) + ".cal"
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/cal/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        full_file_name = os.path.join(file_path, file_name)
        return full_file_name

    def uvw_file(self, sub_array, file_name):
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/uvw_vis/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        full_file_name = os.path.join(file_path, file_name)
        return full_file_name

    def vis_file(self, sub_array, file_name):
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/uvw_vis/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        full_file_name = os.path.join(file_path, file_name)
        return full_file_name

    def data_dir(self, sub_array, year, month, day, hour, minute):
        file_name = ('%04d%02d%02d-%02d%02d') % (year, month, day, hour, minute)
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/dat/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        #full_file_name = os.path.join(file_path, file_name)
        return file_path

    def data_file(self, sub_array, year, month, day, hour, minute):
        file_name = ('%04d%02d%02d-%02d%02d') % (year, month, day, hour, minute)
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/dat/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        full_file_name = os.path.join(file_path, file_name)
        return full_file_name

    def uvfits_file(self, sub_array, file_name):
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/uvfits/"
        full_file_name = os.path.join(file_path, file_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        return full_file_name

    def uvfits_dir(self, sub_array, year, month, day, hour, minute):
        file_name = ('%04d%02d%02d-%02d%02d') % (year, month, day, hour, minute)
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/uvfits/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        #full_file_name = os.path.join(file_path, file_name)
        return file_path

    def png_file(self, sub_array, file_name):
        file_path = self.MUSER_ARCH + "/" + file_name[:8] + "/MUSER-" + str(sub_array) + "/png/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        full_file_name = os.path.join(file_path, file_name)
        return full_file_name

    def rt_display_file(self, sub_array):
        file_path = self.MUSER_WORK + "/temp/display/MUSER-" + str(sub_array)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        return file_path

    def get_log_dir(self):
        log_dir = self.MUSER_WORK + '/log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir


muserenv = MuserEnv()

#MuserEnv.set_logger()

