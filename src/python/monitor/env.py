#! /usr/bin/env python
# encoding: utf-8
"""
muserenv.py
=====================
Environment MUSER-I/II
"""

import sys, os, datetime, time, math


class ENV:
    MUSER_HOME = "/opt"
    MUSER_WORK = "/opt/work"
    MUSER_ARCH = "/opt/archive"

    NMUSER_HOME = "/opt"
    NMUSER_WORK = "/opt/work"
    NMUSER_ARCH = "/opt/archive"

    # LMUSER_HOME = "/opt"
    # LMUSER_WORK = "/opt/work"
    # LMUSER_ARCH = "/opt/archive"

    LMUSER_HOME = "~/dev"
    LMUSER_WORK = "~/dev/work"
    LMUSER_ARCH = "~/dev/archive"

    sub_array = 0

    Antennas = 0
    SubChannels = 0

    def __init__(self, muser=1):
        ENV.sub_array = muser
        if ENV.sub_array == 1:
            ENV.Antennas = 60
            ENV.SubChannels = 66
            ENV.MUSER_HOME = ENV.NMUSER_HOME
            ENV.MUSER_WORK = ENV.NMUSER_WORK
            ENV.MUSER_ARCH = ENV.NMUSER_ARCH
        else:
            ENV.Antennas = 40
            ENV.SubChannels = 8
            ENV.MUSER_HOME = os.path.expanduser(ENV.LMUSER_HOME)
            ENV.MUSER_WORK = os.path.expanduser(ENV.LMUSER_WORK)
            ENV.MUSER_ARCH = os.path.expanduser(ENV.LMUSER_ARCH)

    def calFile(self, year, month, day, cal):
        fileName = ('%04d%02d%02d-%1d') % (year, month, day, cal)+".cal"
        filePath = ENV.MUSER_ARCH + "/" + fileName[:8] + "/MUSER-" + str(ENV.sub_array + 1) + "/cal/"
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        fullFileName = os.path.join(filePath, fileName)
        if not os.path.exists(fullFileName):
            return False, fullFileName
        else:
            return True, fullFileName

    def dataFile(self, year, month, day, hour, minute):
        fileName = ('%04d%02d%02d-%02d%02d') % (year, month, day, hour, minute)
        filePath = ENV.MUSER_ARCH + "/" + fileName[:8] + "/MUSER-" + str(ENV.sub_array + 1) + "/dat/"
        fullFileName = os.path.join(filePath, fileName)
        if not os.path.exists(fullFileName):
            return False, fullFileName
        else:
            return True, fullFileName

    def uvfitsFile(self, fileName):
        filePath = ENV.MUSER_ARCH + "/" + fileName[:8] + "/MUSER-" + str(ENV.sub_array + 1) + "/uvfits/"
        fullFileName = os.path.join(filePath, fileName)
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        if not os.path.exists(fullFileName):
            return False, fullFileName
        else:
            return True, fullFileName

    def pngFile(self, fileName):
        filePath = ENV.MUSER_ARCH + "/" + fileName[:8] + "/MUSER-" + str(ENV.sub_array + 1) + "/png/"
        fullFileName = os.path.join(filePath, fileName)
        if not os.path.exists(fullFileName):
            return False, fullFileName
        else:
            return True, fullFileName

    def rtDisplayFile(self, Pol, Chan):
        filePath = ENV.MUSER_WORK + "/temp/display/" + "/MUSER-" + str(ENV.sub_array + 1)
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        fileName = ('%02d%02d') % (Pol, Chan)
        fullFileName = os.path.join(filePath, fileName)
        if not os.path.exists(fullFileName):
            return False, fullFileName
        else:
            return True, fullFileName

    def getTimeStamp(self, Year, Month, Day, Hour, Minute, Second):
        obsTIME = ('%4d-%02d-%02d %02d:%02d:%02d') % ( Year, Month, Day, Hour, Minute, Second)

        # The date and time are Beijing Time of China, UTC = CST - 8
        Date = time.strptime(obsTIME, '%Y-%m-%d %H:%M:%S')
        timeStamp = time.mktime(Date)
        return timeStamp

    def getDateTime(self, Year, Month, Day, Hour, Minute, Second, MilliSecond):
        obsTIME = ('%4d-%02d-%02d %02d:%02d:%02d %06d') % ( Year, Month, Day, Hour, Minute, Second, MilliSecond)
        return datetime.datetime.strptime(obsTIME, "%Y-%m-%d %H:%M:%S %f")

    def getLocalTime(self, time):
        return (
            time.date().year, time.date().month, time.date().day, time.time().hour, time.time().minute,
            time.time().second)
