#!/usr/bin/env python
import urllib2
import htmllib
import formatter
import re
import time
import HTMLParser
import os
import struct
import datetime
import sys
import socket
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class MyParser(HTMLParser.HTMLParser):
    def __init__(self):
        self.listDate=[]
        HTMLParser.HTMLParser.__init__(self)
    def handle_starttag(self, tag, attrs):
        if tag == 'span':
            pass
    def handle_endtag(self, tag):
        pass
    def handle_data(self, data):
        if re.search('\d\d\d\d-\d\d-\d\d', data) != None:
            self.listDate.append(data)


class MuserIers:

    def __init__(self, iersFilePath=''):
        self.iersFilePath = iersFilePath
        if len(self.iersFilePath) == 0:
            path = os.path.abspath(os.path.dirname(__file__))
            self.iersFilePath = os.path.abspath(os.path.dirname(__file__))[0:path.find('museros')+7]

    def getDates(self):
        ListDate = []
        try:
            website = urllib2.urlopen('https://datacenter.iers.org/web/guest/eop/-/somos/5Rgv/version/6', timeout=300)
        except (urllib2.URLError, socket.timeout):
            try:
                website = urllib2.urlopen('https://datacenter.iers.org/web/guest/eop/-/somos/5Rgv/version/6', timeout=300)
            except (urllib2.URLError, socket.timeout):
                print('Network is not available. Check network.')
                sys.stdout.flush()
                sys.exit()
        try:
            data = website.read()
            website.close()
            pDate = MyParser()
            pDate.feed(data)
        except:
            try:
                website = urllib2.urlopen('https://datacenter.iers.org/web/guest/eop/-/somos/5Rgv/version/6', timeout=300)
            except (urllib2.URLError, socket.timeout):
                print('Network is not available. Check network.')
                sys.stdout.flush()
                sys.exit()
            else:
                data = website.read()
                website.close()
                pDate = MyParser()
                pDate.feed(data)
        listDates = pDate.listDate
        for i in listDates[-1:-(len(listDates)+1):-1]:
            ListDate.append(i)
        return ListDate

    def getURLs(self):
        try:
            website = urllib2.urlopen('https://datacenter.iers.org/web/guest/eop/-/somos/5Rgv/version/6', timeout=300)
        except (urllib2.URLError, socket.timeout):
            try:
                website = urllib2.urlopen('https://datacenter.iers.org/web/guest/eop/-/somos/5Rgv/version/6', timeout=300)
            except (urllib2.URLError, socket.timeout):
                print('Network is not available. Check network.')
                sys.stdout.flush()
                sys.exit()
        try:
            data = website.read()
            website.close()
            format1 = formatter.AbstractFormatter(formatter.NullWriter())
            ptext = htmllib.HTMLParser(format1)
            ptext.feed(data)
            links = []
            links = ptext.anchorlist
        except:
            try:
                website = urllib2.urlopen('https://datacenter.iers.org/web/guest/eop/-/somos/5Rgv/version/6', timeout=300)
            except (urllib2.URLError, socket.timeout):
                print('Network is not available. Check network.')
                sys.stdout.flush()
                sys.exit()
            else:
                data = website.read()
                website.close()
                format1 = formatter.AbstractFormatter(formatter.NullWriter())
                ptext = htmllib.HTMLParser(format1)
                ptext.feed(data)
                links = []
                links = ptext.anchorlist
        linkURLs = []
        for link in links[-1:-(len(links)+1):-1]:
                if (re.search('http.*getTX.*?txt$', link)) != None:
                    linkURLs.append(link)
        return linkURLs

    def saveinfo(self, listDate11, listURL11):
        d1 = datetime.date(2004, 12, 30)
        listDate = listDate11
        listURL = listURL11
        leapDate = []
        listInfoUTC = []
        lostList1 = ['5  1 28  53398  -1 .00007  -1 .00004  -1  .000006',
                     '5  1 29  53399  -1 .00007  -1 .00004  -1  .000006',
                     '5  1 30  53400  -1 .00007  -1 .00004  -1  .000006',
                     '5  1 31  53401  -1 .00007  -1 .00004  -1  .000006',
                     '5  2  1  53402  -1 .00006  -1 .00004  -1  .000018',
                     '5  2  2  53403  -1 .00006  -1 .00004  -1  .000018',
                     '5  2  3  53404  -1 .00006  -1 .00004  -1  .000018']
        lostList2 = ['10  4 23  55309 -.06991 .00010 0.37179 .00009 -0.010971 0.000009',
                     '10  4 24  55310 -.07029 .00010 0.37405 .00009 -0.012739 0.000011',
                     '10  4 25  55311 -.07076 .00010 0.37629 .00009 -0.014613 0.000012',
                     '10  4 26  55312 -.07109 .00010 0.37854 .00009 -0.016485 0.000012',
                     '10  4 27  55313 -.07088 .00009 0.38094 .00009 -0.018286 0.000012',
                     '10  4 28  55314 -.07073 .00009 0.38339 .00009 -0.019923 0.000013',
                     '10  4 29  55315 -.07124 .00009 0.38572 .00009 -0.021289 0.000055s']
        pattern0 = re.compile(r'^Beginning(.*?):?$')
        pattern1 = re.compile(r"TAI-UTC.*?=(.*?)seconds")
        pattern2 = re.compile('.*?IERS Rapid Service.*?')
        pattern3 = re.compile('.*?MJD.*? x\(arcsec\).*?y\(arcsec\).*?UT1-UTC\(sec\).*?')
        pattern4 = re.compile('.*?These.predictions.are.based.on.all.announced.leap.seconds\..*?')
        if not os.path.isfile(self.iersFilePath+'/data/iersDate.log'):
            with open(self.iersFilePath+'/data/iersDate.log', 'w+')as fileDate:
                with open(self.iersFilePath+'/data/iersdata.dat', 'wb+') as fileReal:
                    with open(self.iersFilePath+'/data/infoLeapSecondDate.dat', 'w')as fileUTCDate:
                        with open(self.iersFilePath+'/data/leapsec.dat', 'w') as fileUTC:
                            for i in range(len(listURL)):
                                if  i<5:
                                    print(listDate[i]),
                                    sys.stdout.flush()
                                elif i==5:
                                    print(listDate[i])
                                    sys.stdout.flush()
                                elif (i-5)%7==0:
                                    print(listDate[i])
                                    sys.stdout.flush()
                                else:
                                    print(listDate[i]),
                                    sys.stdout.flush()
                                if (i == 3) or (i == 276):
                                    if i == 3:
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines(5000)
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines(5000)
                                                f.close()
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and j == flag0+13:
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                        print('2005-02-03'),
                                        sys.stdout.flush()
                                        fileDate.write('2005-02-03')
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        for ii in range(7):
                                            tempData = lostList1[ii].split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                    elif i == 276:
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        for jj in range(7):
                                            tempData = lostList2[jj].split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                elif i == (len(listURL)-1):
                                    fileDate.write(listDate[i])
                                    fileDate.write('\n')
                                    fileDate.flush()
                                    try:
                                        f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines()
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines()
                                            f.close()
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                            if digitalDate not in leapDate:
                                                leapDate.append(digitalDate)
                                                fileUTCDate.write(digitalDate)
                                                fileUTCDate.write('\n')
                                                fileUTCDate.flush()
                                                h = temp[j+1]
                                                if pattern1.match(str(h).strip()):
                                                    htime = pattern1.match(str(h).strip()).group(1)
                                                    stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                    if stringUTC not in listInfoUTC:
                                                        listInfoUTC.append(stringUTC)
                                                        fileUTC.write(stringUTC)
                                                        fileUTC.write('\n')
                                                        fileUTC.flush()
                                                        break
                                            else:
                                                break
                                    flag0 = 0
                                    flag1 = 0
                                    for j in range(27, len(temp)):
                                        t = temp[j]
                                        if pattern2.match(str(t)):
                                            flag0 = j
                                            flag1 = 1
                                            continue
                                        if flag1 and (j == flag0+13):
                                            break
                                        if flag1 and (j > flag0+2) and (j < flag0+13):
                                            stringReal = str(t).strip()
                                            if len(stringReal) == 0:
                                                break
                                            tempData = stringReal.split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                    flag = 0
                                    for ii in range(35, len(temp)):
                                        t = temp[ii]
                                        if pattern3.match(str(t)):
                                            flag = 1
                                            continue
                                        if flag and (not pattern4.match(str(t))):
                                            stringPrediction = str(t).strip()
                                            tempData1 = stringPrediction.split()
                                            fileReal.write(struct.pack('H', int(tempData1[0])))
                                            fileReal.write(struct.pack('B', int(tempData1[1])))
                                            fileReal.write(struct.pack('B', int(tempData1[2])))
                                            fileReal.write(struct.pack('I', int(tempData1[3])))
                                            fileReal.write(struct.pack('d', float(tempData1[4])))
                                            fileReal.write(struct.pack('d', float(tempData1[5])))
                                            fileReal.write(struct.pack('d', float(tempData1[6])))
                                            fileReal.flush()
                                            continue
                                        elif pattern4.match(str(t)):
                                            break

                                else:
                                    fileDate.write(listDate[i])
                                    fileDate.write('\n')
                                    fileDate.flush()
                                    try:
                                        f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines(5000)
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines(5000)
                                            f.close()
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                            if digitalDate not in leapDate:
                                                leapDate.append(digitalDate)
                                                fileUTCDate.write(digitalDate)
                                                fileUTCDate.write('\n')
                                                fileUTCDate.flush()
                                                h = temp[j+1]
                                                if pattern1.match(str(h).strip()):
                                                    htime = pattern1.match(str(h).strip()).group(1)
                                                    stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                    if stringUTC not in listInfoUTC:
                                                        listInfoUTC.append(stringUTC)
                                                        fileUTC.write(stringUTC)
                                                        fileUTC.write('\n')
                                                        fileUTC.flush()
                                                        break
                                            else:
                                                break
                                    flag0 = 0
                                    flag1 = 0
                                    for j in range(27, len(temp)):
                                        t = temp[j]
                                        if pattern2.match(str(t)):
                                            flag0 = j
                                            flag1 = 1
                                            continue
                                        if flag1 and (j == flag0+13):
                                            break
                                        if flag1 and (j > flag0+2) and (j < flag0+13):
                                            stringReal = str(t).strip()
                                            if len(stringReal) == 0:
                                                break
                                            tempData = stringReal.split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                            print('IERS data retrieved.')
        else:
            with open(self.iersFilePath+'/data/iersDate.log', 'r+')as fileDate:
                with open(self.iersFilePath+'/data/iersdata.dat', 'rb+') as fileReal:
                    with open(self.iersFilePath+'/data/infoLeapSecondDate.dat', 'r+') as fileUTCDate:
                        with open(self.iersFilePath+'/data/leapsec.dat', 'r+') as fileUTC:
                            Datelines = fileDate.readlines()
                            lines0 = fileUTCDate.readlines()
                            lines1 = fileUTC.readlines()
                            if len(Datelines) == (len(listURL)+1):
                                print('Update IERS data.')
                                sys.stdout.flush()
                                digitalDate = ''
                                stringUTC = ''
                                if re.compile('^\d\d\d\d-\d\d-\d\d$').match(Datelines[-1].strip()):
                                    dateline = Datelines[-2].strip()
                                    datelist = dateline.split('-')
                                    d2 = datetime.date(int(datelist[0]), int(datelist[1]), int(datelist[2]))
                                    interval = str(d2-d1).split()
                                    fileReal.seek(int(interval[0])*32)
                                    try:
                                        f = urllib2.urlopen(str(listURL[-1]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[-1]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines()
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[-1]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines()
                                            f.close()
                                    flag0 = 0
                                    flag1 = 0
                                    for j in range(27, len(temp)):
                                        t = temp[j]
                                        if pattern2.match(str(t)):
                                            flag0 = j
                                            flag1 = 1
                                            continue
                                        if flag1 and (j == flag0+13):
                                            break
                                        if flag1 and (j > flag0+2) and (j < flag0+13):
                                            stringReal = str(t).strip()
                                            if len(stringReal) == 0:
                                                break
                                            tempData = stringReal.split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                    flag = 0
                                    for ii in range(35, len(temp)):
                                        t = temp[ii]
                                        if pattern3.match(str(t)):
                                            flag = 1
                                            continue
                                        if flag and (not pattern4.match(str(t))):
                                            stringPrediction = str(t).strip()
                                            tempData1 = stringPrediction.split()
                                            fileReal.write(struct.pack('H', int(tempData1[0])))
                                            fileReal.write(struct.pack('B', int(tempData1[1])))
                                            fileReal.write(struct.pack('B', int(tempData1[2])))
                                            fileReal.write(struct.pack('I', int(tempData1[3])))
                                            fileReal.write(struct.pack('d', float(tempData1[4])))
                                            fileReal.write(struct.pack('d', float(tempData1[5])))
                                            fileReal.write(struct.pack('d', float(tempData1[6])))
                                            fileReal.flush()
                                            continue
                                        elif pattern4.match(str(t)):
                                            break
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate = ('%04d %02d %02d' % (int(listString[0]), int(listString[1]), int(listString[2])))
                                            h = temp[j+1]
                                            if pattern1.match(str(h).strip()):
                                                htime = pattern1.match(str(h).strip()).group(1)
                                                stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                break
                                    if len(lines1) == len(lines0):
                                        if lines0[-1].strip() == digitalDate and (lines1[-1]).strip() == stringUTC:
                                            pass
                                        elif lines0[-1].strip() == digitalDate and not(lines1[-1].strip() == stringUTC):
                                            fileUTC.seek((len(lines1)-1)*21)
                                            fileUTC.write(stringUTC)
                                            fileUTC.write('\n')
                                            fileUTC.flush()
                                        elif not(lines0[-1].strip() == digitalDate) and not(lines1[-1].strip() == stringUTC):
                                            try:
                                                f1 = urllib2.urlopen(str(listURL[-2]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                try:
                                                    f1 = urllib2.urlopen(str(listURL[-2]), timeout=300)
                                                except (urllib2.URLError, socket.timeout):
                                                    print('Network is not available. Check network.')
                                                    sys.stdout.flush()
                                                    sys.exit()
                                            try:
                                                temp1 = f1.readlines(2500)
                                                f1.close()
                                            except:
                                                try:
                                                    f1 = urllib2.urlopen(str(listURL[-2]),timeout=300)
                                                except (urllib2.URLError, socket.timeout):
                                                    print('Network is not available. Check network.')
                                                    sys.stdout.flush()
                                                    sys.exit()
                                                else:
                                                    temp1 = f1.readlines(2500)
                                                    f1.close()
                                            digitalDate1 = ''
                                            stringUTC1 = ''
                                            for jj in range(18, len(temp1)):
                                                t1 = temp1[jj]
                                                if pattern0.match(str(t1).strip()):
                                                    hdate1 = str(pattern0.match(str(t1).strip()).group(1)).strip()
                                                    listString1 = list(time.strptime(hdate1, '%d %B %Y'))
                                                    digitalDate1='%04d %02d %02d'%(int(listString1[0]),int(listString1[1]),int(listString1[2]))
                                                    h1 = temp1[jj+1]
                                                    if pattern1.match(str(h1).strip()):
                                                        htime1 = pattern1.match(str(h1).strip()).group(1)
                                                        stringUTC1 = (digitalDate1+' ' + ''.join(str(htime1).split(' '))).strip()
                                                        break
                                            if lines0[-1].strip() == digitalDate1 and lines1[-1].strip() == stringUTC1:
                                                fileUTCDate.seek(len(lines0)*11)
                                                fileUTC.seek(len(lines1)*21)
                                                fileUTCDate.write(digitalDate)
                                                fileUTCDate.write('\n')
                                                fileUTCDate.flush()
                                                fileUTC.write(stringUTC)
                                                fileUTC.write('\n')
                                                fileUTC.flush()
                                            elif not(lines0[-1].strip() == digitalDate1) and not(lines1[-1].strip() == stringUTC1):
                                                fileUTCDate.seek(len(lines0)*11)
                                                fileUTC.seek(len(lines1)*21)
                                                fileUTCDate.write(digitalDate)
                                                fileUTCDate.write('\n')
                                                fileUTCDate.flush()
                                                fileUTC.write(stringUTC)
                                                fileUTC.write('\n')
                                                fileUTC.flush()
                                    else:
                                        fileUTCDate.seek((len(lines0)-1)*11)
                                        fileUTCDate.write(digitalDate)
                                        fileUTCDate.write('\n')
                                        fileUTCDate.flush()
                                        fileUTC.seek((len(lines1))*21)
                                        fileUTC.write(stringUTC)
                                        fileUTC.write('\n')
                                        fileUTC.flush()

                                else:
                                    len1 = len(Datelines)-1
                                    fileDate.seek(len1*11)
                                    fileDate.write(listDate[-1])
                                    fileDate.write('\n')
                                    fileDate.flush()
                                    dateline = Datelines[-2].strip()
                                    datelist = dateline.split('-')
                                    d2 = datetime.date(int(datelist[0]), int(datelist[1]), int(datelist[2]))
                                    interval = str(d2-d1).split()
                                    fileReal.seek(int(interval[0])*32)
                                    try:
                                        f = urllib2.urlopen(str(listURL[-1]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[-1]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines()
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[-1]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines()
                                            f.close()
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                            if digitalDate not in leapDate:
                                                leapDate.append(digitalDate)
                                                fileUTCDate.write(digitalDate)
                                                fileUTCDate.write('\n')
                                                fileUTCDate.flush()
                                                h = temp[j+1]
                                                if pattern1.match(str(h).strip()):
                                                    htime = pattern1.match(str(h).strip()).group(1)
                                                    stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                    if stringUTC not in listInfoUTC:
                                                        listInfoUTC.append(stringUTC)
                                                        fileUTC.write(stringUTC)
                                                        fileUTC.write('\n')
                                                        fileUTC.flush()
                                                        break
                                            else:
                                                break
                                    flag0 = 0
                                    flag1 = 0
                                    for j in range(27, len(temp)):
                                        t = temp[j]
                                        if pattern2.match(str(t)):
                                            flag0 = j
                                            flag1 = 1
                                            continue
                                        if flag1 and (j == flag0+13):
                                            break
                                        if flag1 and (j > flag0+2) and (j < flag0+13):
                                            stringReal = str(t).strip()
                                            if len(stringReal) == 0:
                                                break
                                            tempData = stringReal.split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                    flag = 0
                                    for ii in range(35, len(temp)):
                                        t = temp[ii]
                                        if pattern3.match(str(t)):
                                            flag = 1
                                            continue
                                        if flag and (not pattern4.match(str(t))):
                                            stringPrediction = str(t).strip()
                                            tempData1 = stringPrediction.split()
                                            fileReal.write(struct.pack('H', int(tempData1[0])))
                                            fileReal.write(struct.pack('B', int(tempData1[1])))
                                            fileReal.write(struct.pack('B', int(tempData1[2])))
                                            fileReal.write(struct.pack('I', int(tempData1[3])))
                                            fileReal.write(struct.pack('d', float(tempData1[4])))
                                            fileReal.write(struct.pack('d', float(tempData1[5])))
                                            fileReal.write(struct.pack('d', float(tempData1[6])))
                                            fileReal.flush()
                                            continue
                                        elif pattern4.match(str(t)):
                                            break
                                print('\n')
                                print('IERS data updated.')
                                sys.stdout.flush()
                            elif len(Datelines) == len(listURL):
                                print('Update IERS data.')
                                sys.stdout.flush()
                                dateline = Datelines[-2].strip()
                                datelist = dateline.split('-')
                                d2 = datetime.date(int(datelist[0]), int(datelist[1]), int(datelist[2]))
                                interval = str(d2-d1).split()
                                fileReal.seek(int(interval[0])*32)
                                digitalDate = ''
                                stringUTC = ''
                                if Datelines[-1].strip() == listDate[-2].strip():
                                    try:
                                        f = urllib2.urlopen(str(listURL[-2]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[-2]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines(2500)
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[-2]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines(2500)
                                            f.close()
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate = '%04d %02d %02d' % (int(listString[0]), int(listString[1]), int(listString[2]))
                                            h = temp[j+1]
                                            if pattern1.match(str(h).strip()):
                                                htime = pattern1.match(str(h).strip()).group(1)
                                                stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                break
                                    if len(lines0) == len(lines1):
                                        if lines0[-1].strip() == digitalDate and lines1[-1].strip() == stringUTC:
                                            fileUTCDate.seek(len(lines0)*11)
                                            fileUTC.seek(len(lines1)*21)
                                            for ii in range(len(lines0)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                        elif lines0[-1].strip() == digitalDate and not(lines1[-1].strip() == stringUTC):
                                            fileUTC.seek((len(lines1)-1)*21)
                                            fileUTCDate.seek((len(lines0)-1)*11)
                                            for ii in range((len(lines0)-1)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                        elif not(lines0[-1].strip() == digitalDate) and not(lines1[-1].strip() == stringUTC):
                                            fileUTCDate.seek(len(lines0)*11)
                                            fileUTC.seek(len(lines1)*21)
                                            for ii in range(len(lines0)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                    else:
                                        fileUTCDate.seek((len(lines0)-1)*11)
                                        fileUTC.seek((len(lines1))*21)
                                        for ii in range((len(lines0)-1)):
                                            leapDate.append(lines0[ii].strip())
                                            listInfoUTC.append(lines1[ii].strip())
                                else:
                                    fileUTCDate.seek((len(lines0))*11)
                                    fileUTC.seek((len(lines1))*21)
                                    for ii in range(len(lines0)):
                                        leapDate.append(lines0[ii].strip())
                                        listInfoUTC.append(lines1[ii].strip())
                                for i in range(len(listURL)-2, len(listURL)):
                                    if i < (len(listURL)-1):
                                        fileDate.seek((len(Datelines)-2)*11)
                                        fileDate.write(listDate[-2])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines()
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines()
                                                f.close()
                                        for j in range(18, len(temp)):
                                            t = temp[j]
                                            if pattern0.match(str(t).strip()):
                                                hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                                listString = list(time.strptime(hdate, '%d %B %Y'))
                                                digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                                if digitalDate not in leapDate:
                                                    leapDate.append(digitalDate)
                                                    fileUTCDate.write(digitalDate)
                                                    fileUTCDate.write('\n')
                                                    fileUTCDate.flush()
                                                    h = temp[j+1]
                                                    if pattern1.match(str(h).strip()):
                                                        htime = pattern1.match(str(h).strip()).group(1)
                                                        stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                        if stringUTC not in listInfoUTC:
                                                            listInfoUTC.append(stringUTC)
                                                            fileUTC.write(stringUTC)
                                                            fileUTC.write('\n')
                                                            fileUTC.flush()
                                                            break
                                                else:
                                                    break
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and (j == flag0+13):
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                    else:
                                        fileDate.write(listDate[-1])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines()
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines()
                                                f.close()
                                        for j in range(18, len(temp)):
                                            t = temp[j]
                                            if pattern0.match(str(t).strip()):
                                                hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                                listString = list(time.strptime(hdate, '%d %B %Y'))
                                                digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                                if digitalDate not in leapDate:
                                                    leapDate.append(digitalDate)
                                                    fileUTCDate.write(digitalDate)
                                                    fileUTCDate.write('\n')
                                                    fileUTCDate.flush()
                                                    h = temp[j+1]
                                                    if pattern1.match(str(h).strip()):
                                                        htime = pattern1.match(str(h).strip()).group(1)
                                                        stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                        if stringUTC not in listInfoUTC:
                                                            listInfoUTC.append(stringUTC)
                                                            fileUTC.write(stringUTC)
                                                            fileUTC.write('\n')
                                                            fileUTC.flush()
                                                            break
                                                else:
                                                    break
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and (j == flag0+13):
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                        flag = 0
                                        for ii in range(35, len(temp)):
                                            t = temp[ii]
                                            if pattern3.match(str(t)):
                                                flag = 1
                                                continue
                                            if flag and (not pattern4.match(str(t))):
                                                stringPrediction = str(t).strip()
                                                tempData1 = stringPrediction.split()
                                                fileReal.write(struct.pack('H', int(tempData1[0])))
                                                fileReal.write(struct.pack('B', int(tempData1[1])))
                                                fileReal.write(struct.pack('B', int(tempData1[2])))
                                                fileReal.write(struct.pack('I', int(tempData1[3])))
                                                fileReal.write(struct.pack('d', float(tempData1[4])))
                                                fileReal.write(struct.pack('d', float(tempData1[5])))
                                                fileReal.write(struct.pack('d', float(tempData1[6])))
                                                fileReal.flush()
                                                continue
                                            elif pattern4.match(str(t)):
                                                break
                                print('\n')
                                print('IERS data updated.')
                                sys.stdout.flush()
                            elif len(Datelines) == 278:
                                print('Update IERS data.')
                                sys.stdout.flush()
                                fileDate.seek(277*11)
                                dateline = Datelines[-2].strip()
                                datelist = dateline.split('-')
                                d2 = datetime.date(int(datelist[0]), int(datelist[1]), int(datelist[2]))
                                interval = str(d2-d1).split()
                                fileReal.seek(int(interval[0])*32)
                                digitalDate = ''
                                stringUTC = ''
                                if Datelines[-1].strip() == listDate[len(Datelines)-2].strip():
                                    try:
                                        f = urllib2.urlopen(str(listURL[len(Datelines)-2]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[len(Datelines)-2]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines(2500)
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[len(Datelines)-2]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines(2500)
                                            f.close()
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate = '%04d %02d %02d' % (int(listString[0]), int(listString[1]), int(listString[2]))
                                            h = temp[j+1]
                                            if pattern1.match(str(h).strip()):
                                                htime = pattern1.match(str(h).strip()).group(1)
                                                stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                break
                                    if len(lines0) == len(lines1):
                                        if lines0[-1].strip() == digitalDate and lines1[-1].strip() == stringUTC:
                                            fileUTCDate.seek(len(lines0)*11)
                                            fileUTC.seek(len(lines1)*21)
                                            for ii in range(len(lines0)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                        elif lines0[-1].strip() == digitalDate and not(lines1[-1].strip() == stringUTC):
                                            fileUTC.seek((len(lines1)-1)*21)
                                            fileUTCDate.seek((len(lines0)-1)*11)
                                            for ii in range((len(lines0)-1)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                        elif not(lines0[-1].strip() == digitalDate) and not(lines1[-1].strip() == stringUTC):
                                            fileUTCDate.seek(len(lines0)*11)
                                            fileUTC.seek(len(lines1)*21)
                                            for ii in range(len(lines0)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                    else:
                                        fileUTCDate.seek((len(lines0)-1)*11)
                                        fileUTC.seek((len(lines1))*21)
                                        for ii in range((len(lines0)-1)):
                                            leapDate.append(lines0[ii].strip())
                                            listInfoUTC.append(lines1[ii].strip())
                                else:
                                    fileUTCDate.seek((len(lines0))*11)
                                    fileUTC.seek((len(lines1))*21)
                                    for ii in range(len(lines0)):
                                        leapDate.append(lines0[ii].strip())
                                        listInfoUTC.append(lines1[ii].strip())
                                for i in range(276, len(listURL)):
                                    if (i-6)%7==0:
                                        print(listDate[i])
                                        sys.stdout.flush()
                                    else:
                                        print(listDate[i]),
                                        sys.stdout.flush()
                                    if i == 276:
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        for jj in range(7):
                                            tempData = lostList2[jj].split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                    elif i == (len(listURL)-1):
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines()
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines()
                                                f.close()
                                        for j in range(18, len(temp)):
                                            t = temp[j]
                                            if pattern0.match(str(t).strip()):
                                                hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                                listString = list(time.strptime(hdate, '%d %B %Y'))
                                                digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                                if digitalDate not in leapDate:
                                                    leapDate.append(digitalDate)
                                                    fileUTCDate.write(digitalDate)
                                                    fileUTCDate.write('\n')
                                                    fileUTCDate.flush()
                                                    h = temp[j+1]
                                                    if pattern1.match(str(h).strip()):
                                                        htime = pattern1.match(str(h).strip()).group(1)
                                                        stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                        if stringUTC not in listInfoUTC:
                                                            listInfoUTC.append(stringUTC)
                                                            fileUTC.write(stringUTC)
                                                            fileUTC.write('\n')
                                                            fileUTC.flush()
                                                            break
                                                else:
                                                    break
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and (j == flag0+13):
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                        flag = 0
                                        for ii in range(35, len(temp)):
                                            t = temp[ii]
                                            if pattern3.match(str(t)):
                                                flag = 1
                                                continue
                                            if flag and (not pattern4.match(str(t))):
                                                stringPrediction = str(t).strip()
                                                tempData1 = stringPrediction.split()
                                                fileReal.write(struct.pack('H', int(tempData1[0])))
                                                fileReal.write(struct.pack('B', int(tempData1[1])))
                                                fileReal.write(struct.pack('B', int(tempData1[2])))
                                                fileReal.write(struct.pack('I', int(tempData1[3])))
                                                fileReal.write(struct.pack('d', float(tempData1[4])))
                                                fileReal.write(struct.pack('d', float(tempData1[5])))
                                                fileReal.write(struct.pack('d', float(tempData1[6])))
                                                fileReal.flush()
                                                continue
                                            elif pattern4.match(str(t)):
                                                break
                                    else:
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines(5000)
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines(5000)
                                                f.close()
                                        for j in range(18, len(temp)):
                                            t = temp[j]
                                            if pattern0.match(str(t).strip()):
                                                hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                                listString = list(time.strptime(hdate, '%d %B %Y'))
                                                digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                                if digitalDate not in leapDate:
                                                    leapDate.append(digitalDate)
                                                    fileUTCDate.write(digitalDate)
                                                    fileUTCDate.write('\n')
                                                    fileUTCDate.flush()
                                                    h = temp[j+1]
                                                    if pattern1.match(str(h).strip()):
                                                        htime = pattern1.match(str(h).strip()).group(1)
                                                        stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                        if stringUTC not in listInfoUTC:
                                                            listInfoUTC.append(stringUTC)
                                                            fileUTC.write(stringUTC)
                                                            fileUTC.write('\n')
                                                            fileUTC.flush()
                                                            break
                                                else:
                                                    break
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and (j == flag0+13):
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                print('\n')
                                print('IERS data updated.')
                                sys.stdout.flush()
                            elif len(Datelines)>40:
                                print('Update IERS data.')
                                sys.stdout.flush()
                                fileDate.seek((len(Datelines)-1)*11)
                                dateline = Datelines[-2].strip()
                                datelist = dateline.split('-')
                                d2 = datetime.date(int(datelist[0]), int(datelist[1]), int(datelist[2]))
                                interval = str(d2-d1).split()
                                fileReal.seek(int(interval[0])*32)
                                digitalDate = ''
                                stringUTC = ''
                                if Datelines[-1].strip() == listDate[len(Datelines)-2].strip():
                                    try:
                                        f = urllib2.urlopen(str(listURL[len(Datelines)-2]), timeout=300)
                                    except (urllib2.URLError, socket.timeout):
                                        try:
                                            f = urllib2.urlopen(str(listURL[len(Datelines)-2]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                    try:
                                        temp = f.readlines(2500)
                                        f.close()
                                    except:
                                        try:
                                            f = urllib2.urlopen(str(listURL[len(Datelines)-2]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            print('Network is not available. Check network.')
                                            sys.stdout.flush()
                                            sys.exit()
                                        else:
                                            temp = f.readlines(2500)
                                            f.close()
                                    for j in range(18, len(temp)):
                                        t = temp[j]
                                        if pattern0.match(str(t).strip()):
                                            hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                            listString = list(time.strptime(hdate, '%d %B %Y'))
                                            digitalDate = '%04d %02d %02d' % (int(listString[0]), int(listString[1]), int(listString[2]))
                                            h = temp[j+1]
                                            if pattern1.match(str(h).strip()):
                                                htime = pattern1.match(str(h).strip()).group(1)
                                                stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                break
                                    if len(lines0) == len(lines1):
                                        if lines0[-1].strip() == digitalDate and lines1[-1].strip() == stringUTC:
                                            fileUTCDate.seek(len(lines0)*11)
                                            fileUTC.seek(len(lines1)*21)
                                            for ii in range(len(lines0)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                        elif lines0[-1].strip() == digitalDate and not(lines1[-1].strip() == stringUTC):
                                            fileUTC.seek((len(lines1)-1)*21)
                                            fileUTCDate.seek((len(lines0)-1)*11)
                                            for ii in range((len(lines0)-1)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                        elif not(lines0[-1].strip() == digitalDate) and not(lines1[-1].strip() == stringUTC):
                                            fileUTCDate.seek(len(lines0)*11)
                                            fileUTC.seek(len(lines1)*21)
                                            for ii in range(len(lines0)):
                                                leapDate.append(lines0[ii].strip())
                                                listInfoUTC.append(lines1[ii].strip())
                                    else:
                                        fileUTCDate.seek((len(lines0)-1)*11)
                                        fileUTC.seek((len(lines1))*21)
                                        for ii in range((len(lines0)-1)):
                                            leapDate.append(lines0[ii].strip())
                                            listInfoUTC.append(lines1[ii].strip())
                                else:
                                    fileUTCDate.seek((len(lines0))*11)
                                    fileUTC.seek((len(lines1))*21)
                                    for ii in range(len(lines0)):
                                        leapDate.append(lines0[ii].strip())
                                        listInfoUTC.append(lines1[ii].strip())
                                for i in range((len(Datelines)-2), len(listURL)):
                                    if (i-6)%7==0:
                                        print(listDate[i])
                                        sys.stdout.flush()
                                    else:
                                        print(listDate[i]),
                                        sys.stdout.flush()
                                    if i == 276:
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        for jj in range(7):
                                            tempData = lostList2[jj].split()
                                            fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                            fileReal.write(struct.pack('B', int(tempData[1])))
                                            fileReal.write(struct.pack('B', int(tempData[2])))
                                            fileReal.write(struct.pack('I', int(tempData[3])))
                                            fileReal.write(struct.pack('d', float(tempData[4])))
                                            fileReal.write(struct.pack('d', float(tempData[6])))
                                            fileReal.write(struct.pack('d', float(tempData[8])))
                                            fileReal.flush()
                                    elif i == (len(listURL)-1):
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines()
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines()
                                                f.close()
                                        for j in range(18, len(temp)):
                                            t = temp[j]
                                            if pattern0.match(str(t).strip()):
                                                hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                                listString = list(time.strptime(hdate, '%d %B %Y'))
                                                digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                                if digitalDate not in leapDate:
                                                    leapDate.append(digitalDate)
                                                    fileUTCDate.write(digitalDate)
                                                    fileUTCDate.write('\n')
                                                    fileUTCDate.flush()
                                                    h = temp[j+1]
                                                    if pattern1.match(str(h).strip()):
                                                        htime = pattern1.match(str(h).strip()).group(1)
                                                        stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                        if stringUTC not in listInfoUTC:
                                                            listInfoUTC.append(stringUTC)
                                                            fileUTC.write(stringUTC)
                                                            fileUTC.write('\n')
                                                            fileUTC.flush()
                                                            break
                                                else:
                                                    break
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and (j == flag0+13):
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                        flag = 0
                                        for ii in range(35, len(temp)):
                                            t = temp[ii]
                                            if pattern3.match(str(t)):
                                                flag = 1
                                                continue
                                            if flag and (not pattern4.match(str(t))):
                                                stringPrediction = str(t).strip()
                                                tempData1 = stringPrediction.split()
                                                fileReal.write(struct.pack('H', int(tempData1[0])))
                                                fileReal.write(struct.pack('B', int(tempData1[1])))
                                                fileReal.write(struct.pack('B', int(tempData1[2])))
                                                fileReal.write(struct.pack('I', int(tempData1[3])))
                                                fileReal.write(struct.pack('d', float(tempData1[4])))
                                                fileReal.write(struct.pack('d', float(tempData1[5])))
                                                fileReal.write(struct.pack('d', float(tempData1[6])))
                                                fileReal.flush()
                                                continue
                                            elif pattern4.match(str(t)):
                                                break
                                    else:
                                        fileDate.write(listDate[i])
                                        fileDate.write('\n')
                                        fileDate.flush()
                                        try:
                                            f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                        except (urllib2.URLError, socket.timeout):
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                        try:
                                            temp = f.readlines(5000)
                                            f.close()
                                        except:
                                            try:
                                                f = urllib2.urlopen(str(listURL[i]), timeout=300)
                                            except (urllib2.URLError, socket.timeout):
                                                print('Network is not available. Check network.')
                                                sys.stdout.flush()
                                                sys.exit()
                                            else:
                                                temp = f.readlines(5000)
                                                f.close()
                                        for j in range(18, len(temp)):
                                            t = temp[j]
                                            if pattern0.match(str(t).strip()):
                                                hdate = str(pattern0.match(str(t).strip()).group(1)).strip()
                                                listString = list(time.strptime(hdate, '%d %B %Y'))
                                                digitalDate='%04d %02d %02d'%(int(listString[0]),int(listString[1]),int(listString[2]))
                                                if digitalDate not in leapDate:
                                                    leapDate.append(digitalDate)
                                                    fileUTCDate.write(digitalDate)
                                                    fileUTCDate.write('\n')
                                                    fileUTCDate.flush()
                                                    h = temp[j+1]
                                                    if pattern1.match(str(h).strip()):
                                                        htime = pattern1.match(str(h).strip()).group(1)
                                                        stringUTC = (digitalDate+' ' + ''.join(str(htime).split(' '))).strip()
                                                        if stringUTC not in listInfoUTC:
                                                            listInfoUTC.append(stringUTC)
                                                            fileUTC.write(stringUTC)
                                                            fileUTC.write('\n')
                                                            fileUTC.flush()
                                                            break
                                                else:
                                                    break
                                        flag0 = 0
                                        flag1 = 0
                                        for j in range(27, len(temp)):
                                            t = temp[j]
                                            if pattern2.match(str(t)):
                                                flag0 = j
                                                flag1 = 1
                                                continue
                                            if flag1 and (j == flag0+13):
                                                break
                                            if flag1 and (j > flag0+2) and (j < flag0+13):
                                                stringReal = str(t).strip()
                                                if len(stringReal) == 0:
                                                    break
                                                tempData = stringReal.split()
                                                fileReal.write(struct.pack('H', (int(tempData[0])+2000)))
                                                fileReal.write(struct.pack('B', int(tempData[1])))
                                                fileReal.write(struct.pack('B', int(tempData[2])))
                                                fileReal.write(struct.pack('I', int(tempData[3])))
                                                fileReal.write(struct.pack('d', float(tempData[4])))
                                                fileReal.write(struct.pack('d', float(tempData[6])))
                                                fileReal.write(struct.pack('d', float(tempData[8])))
                                                fileReal.flush()
                                print('\n')
                                print('IERS data updated.')
                                sys.stdout.flush()
                            else:
                                print('Rebuild IERS data.')
                                sys.stdout.flush()
                                if os.path.isfile(self.iersFilePath+'/data/iersDate.log'):
                                    os.remove(self.iersFilePath+'/data/iersDate.log')
                                if os.path.isfile(self.iersFilePath+'/data/infoLeapSecondDate.dat'):
                                    os.remove(self.iersFilePath+'/data/infoLeapSecondDate.dat')
                                if os.path.isfile(self.iersFilePath+'/data/leapsec.dat'):
                                    os.remove(self.iersFilePath+'/data/leapsec.dat')
                                if os.path.isfile(self.iersFilePath+'/data/iersdata.dat'):
                                    os.remove(self.iersFilePath+'/data/iersdata.dat')
                                self.saveinfo(listDate11, listURL11,self.iersFilePath)
                                print('IERS data rebuilt.')
                                sys.stdout.flush()
                            
    def update(self):
        if (not os.path.isfile(self.iersFilePath+'/data/iersdata.dat')) and (not os.path.isfile(self.iersFilePath+'/data/leapsec.dat')):
            print('Initialize iersSync.')
            sys.stdout.flush()
            print('Connect to IERS.')
            sys.stdout.flush()
            print('Retrieve the release dates.')
            sys.stdout.flush()
            listDate1 = self.getDates()
            print('The release dates retrieved.')
            sys.stdout.flush()
            print('Retrieve the urls.')
            sys.stdout.flush()
            listURL1 = self.getURLs()
            print('The urls retrieved.')
            sys.stdout.flush()
            self.saveinfo(listDate1, listURL1)
        else:
            print('Determine data status.')
            sys.stdout.flush()
            listDate1 = self.getDates()
            listURL1 = self.getURLs()
            num1=0
            with open(self.iersFilePath+'/data/iersDate.log')as f1:
                num1=len(f1.readlines())
            if num1 == len(listDate1)+1:
                self.saveinfo(listDate1, listURL1)
            else:
                self.saveinfo(listDate1, listURL1)

    def query(self, year, month, day):
        d1 = datetime.date(2004, 12, 31)
        d2 = datetime.date(int(year), int(month), int(day))
        interval=str(d2-d1).split()
        if interval[0].strip()=='0:00:00':
            num=0
        else:
            num = int(interval[0])

        iersFile  = self.iersFilePath +"/data/iersdata.dat"

        if os.path.exists(self.iersFilePath+'/data/iersDate.log'):
            with open(iersFile,'rb')as f:
                f.seek(num*32)
                return  str(struct.unpack('H',f.read(2))[0])+' ' +str(struct.unpack('B',f.read(1))[0])+' ' + str(struct.unpack('B',f.read(1))[0])+' ' + str(struct.unpack('I',f.read(4))[0])+' ' + str(struct.unpack('d',f.read(8))[0])+' ' + str(struct.unpack('d',f.read(8))[0])+' ' + str(struct.unpack('d',f.read(8))[0])

