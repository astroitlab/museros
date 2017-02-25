__author__ = 'mying'

from src.python.pymuser.muserframe import *
from src.python.pymuser.muserenv import *

class dataInfo:

    def __init__(self, sub_array=2):
        self.sub_array = sub_array
        self.manualOrAuto = 0
        self.env = ENV(self)

    def getDataInfo(self, Year, Month, Day, Hour, Minute):

        '''Year = string.atoi(obsDateTime[:4])
        Month = string.atoi(obsDateTime[4:6])
        Day = string.atoi(obsDateTime[6:8])
        Hour = string.atoi(obsDateTime[8:10])
        Minute = string.atoi(obsDateTime[10:12])'''

        obsFileStatus, obsDataFile =  self.env.dataFile(Year, Month, Day, Hour, Minute)

        if obsFileStatus == False:
            print "Cannot find observational data file:"+obsDataFile
            return False

        print "Processing: ", obsDataFile

        inFile= open(obsDataFile, 'rb')
        inFile.seek(0)

        csrh = CSRHRawData(inFile, self.sub_array, self.manualOrAuto)

        if csrh.readOneFrame()==False:
            exit()

        start_TIME = csrh.obsdate + " "+csrh.obstime

        print start_TIME

        if csrh.sub_array == 0:
            csrh.inFile.seek(-100001, 2)
        else:
            csrh.inFile.seek(-204801, 2)

        if csrh.readOneFrame()==False:
            exit()
        end_TIME = csrh.obsdate + " " + csrh.obstime

        inFile.close()
        print start_TIME,"-", end_TIME

if __name__ == '__main__':
    print "MUSER Observational Data File Information V1.0"
    if len(sys.argv) <= 1:
        print ("Usage:")
        print ("python datafileinfo.py <yyyymmddhhmm>  <sub_array>")
        exit(0)

    obsDateTime = sys.argv[1]
    sub_array = string.atoi(sys.argv[2])


    Year = string.atoi(obsDateTime[:4])
    Month = string.atoi(obsDateTime[4:6])
    Day = string.atoi(obsDateTime[6:8])
    Hour = string.atoi(obsDateTime[8:10])
    Minute = string.atoi(obsDateTime[10:12])

    dat = dataInfo(sub_array)
    dat.getDataInfo(Year, Month, Day, Hour, Minute)















