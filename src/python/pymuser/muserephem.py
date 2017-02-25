import sys
import math
import datetime
import struct
import os
from muserenv import *

import novas


DATAPATH='../data'


class Observatory:
    name = ''
    latitude = 0.0
    longitude = 0.0
    altitude = 0.0
    temperature = 0.0
    pressure = 0.0
    obstime = ''

    def __init__(self, name="MUSER", lon=115.2505, lat=42.2118333333333333, altitude =1365., temp=0, pre=0):
        self.name = name
        self.latitude = lat
        self.longitude = lon
        self.altitude = altitude
        self.temperature = temp
        self.pressure = pre

class Body:
    obstime = ''
    obsdate = ''

    midnightJD =0.
    def __init__(self, **params):
        if len(params) == 1:
            self.body = Planet(**params)
            self.name = params['name']
        elif len(params) == 3:
            self.body = FixedBody(**params)
            self.name = params['name']

    def compute(self, **param):
        self.JD, self.deltat, self.eqeq, self.obj_tupapp, self.obj_tuptopo, self.gast, self.last = (self.body).compute(**param)
        self.appra =  self.obj_tupapp[1]
        self.appdec =  self.obj_tupapp[2]
        self.topora = self.obj_tuptopo[1]
        self.topodec  = self.obj_tuptopo[2]
        return self.JD, self.deltat, self.eqeq, self.obj_tupapp, self.obj_tuptopo, self.gast, self.last


planetmember = {'mercury': 1,
                'venus': 2,
                'earth': 3,
                'mars': 4,
                'jupiter': 5,
                'saturn': 6,
                'uranus': 7,
                'neptune': 8,
                'pluto': 9,
                'sun': 10,
                'moon': 11
}


class Star:
    def __init__(self, name='sun', ra=0.0, dec=0.0, promotionra=0.0, promotiondec=0.0,
                 parallax=0.0, radialvelocity=0.0):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.promotionra = promotionra
        self.promotiondec = promotiondec
        self.parallax = parallax
        self.radialvelocity = radialvelocity
        self.ephem = MuserEphem()

    def compute(self, cobs, cdate='2013-10-10', ctime='0:0:0'):
        cat = novas.Cat()
        cat.catalog = 'STAR'
        cat.starname = self.name
        cat.starnumber = 1
        cat.ra = self.ra
        cat.dec = self.dec
        cat.promora = self.promotionra
        cat.promodec = self.promotiondec
        cat.parallax = self.parallax
        cat.radialvelocity = self.radialvelocity

        JD, mjd = self.ephem.convert_date(cdate, ctime)
        tt, deltat,ut1offset  = self.ephem.delta_t(JD)
        jd_tt = JD + tt/86400
        jd_ut1 = JD + ut1offset/86400.

        loc = novas.On_surface()
        # Emmen, The Netherlands
        loc.latitude = float(cobs.latitude)
        loc.longitude = float(cobs.longitude)
        loc.height = float(cobs.altitude)
        loc.temperature = 10.0
        loc.pressure = 1010.0

        # AppPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
        # See almanac.py
        eqeq = self.ephem.eq_of_equinox(jd_tt)

        # TopoPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
        obj_tupapp = novas.AppStar(jd_tt, cat, 0)
        obj_tuptopo = novas.TopoStar(jd_tt, cat, deltat, loc, 0)

        gast = novas.SiderealTime(jd_ut1, 0.0, deltat, 1, 1, 0)
        last = gast + float(cobs.longitude) / 15.0
        if (last >= 24.0):
            last -= 24.0
        if (last < 0.0):
            last += 24.0
            #theta = era (jd_ut1,0.0);
        return JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last


class FixedBody:
    def __init__(self, name='sun', sra='', sdec=''):
        self.name = name
        self.ephem = MuserEphem()
        self.ra = self.ephem.covert_dms(sra)
        self.dec = self.ephem.covert_dms(sdec)
        #print "RA-DEC",self.ra, self.dec

    def compute(self, cobs, cdate='2013-10-10', ctime='0:0:0'):
        cat = novas.Cat()
        cat.catalog = 'FK6'
        cat.starname = self.name
        cat.starnumber = 1
        cat.ra = self.ra
        cat.dec = self.dec
        cat.promora = 0.0
        cat.promodec = 0.0
        cat.parallax = 0.0
        cat.radialvelocity = 0.0

        JD, mjd = self.ephem.convert_date(cdate, ctime)

        tt, deltat,ut1offset  = self.ephem.delta_t(JD)
        jd_tt = JD + tt/86400
        jd_ut1 = JD + ut1offset/86400.


        loc = novas.On_surface()
        # Emmen, The Netherlands
        loc.latitude = float(cobs.latitude)
        loc.longitude = float(cobs.longitude)
        loc.height = float(cobs.altitude)
        loc.temperature = 10.0
        loc.pressure = 1010.0

        # AppPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
        # See almanac.py
        eqeq = self.ephem.eq_of_equinox(jd_tt)

        #fpath = os.path.split(os.path.realpath(__file__))[0]
        #fpath =  os.path.join(self.ephem.env.get_home_dir()+'/data','JPLEPH')
        #        print "FPATH:", fpath
        fil = novas.EphemOpen()

        # TopoPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
        obj_tupapp = novas.AppStar(jd_tt, cat, 0)

        obj_tuptopo = novas.TopoStar(jd_tt, deltat, cat, loc, 0)

        gast = novas.SiderealTime(jd_ut1, 0.0, deltat, 1, 1, 0)
        last = gast + float(cobs.longitude) / 15.0

        if (last >= 24.0):
            last -= 24.0
        if (last < 0.0):
            last += 24.0
            #theta = era (jd_ut1,0.0);
        novas.EphemClose()

        return JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last


class Planet:


    def __init__(self, name=''):
        self.name = name
        self.ephem = MuserEphem()

    def compute(self, cobs, cdate='2013-10-10', ctime='0:0:0'):
        try:
            cat = novas.Cat()
            cat.catalog = 'xxx'
            cat.starname = 'DUMMY'
            cat.starnumber = 0
            cat.ra = 0.0
            cat.dec = 0.0
            cat.promora = 0.0
            cat.promodec = 0.0
            cat.parallax = 0.0
            cat.radialvelocity = 0.0

            fil = novas.EphemOpen()

            JD, mjd =self.ephem.convert_date(cdate, ctime)

            tt, deltat,ut1offset  =self.ephem.delta_t(JD)
            jd_tt = JD + tt/86400
            jd_ut1 = JD + ut1offset/86400.

            loc = novas.On_surface()
            # Emmen, The Netherlands
            loc.latitude = float(cobs.latitude)
            loc.longitude = float(cobs.longitude)
            loc.height = float(cobs.altitude)
            loc.temperature = 10.0
            loc.pressure = 1010.0

            # Sun
            sun = novas.Object()
            sun.type = 0
            #print 'NAME:', self.name


            sun.number = planetmember[self.name.lower()]
            sun.name = self.name.upper()
            sun.star = cat

            # AppPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
            # See almanac.py

            eqeq = self.ephem.eq_of_equinox(jd_tt)



            #obj_tup = novas.AppPlanet(tjd, sun, 0)
            #print obj_tup
            #sha = 360.0 - obj_tup[0] * 15.0
            #eadst = (0.0 + obj_tup[2])

            # TopoPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
            obj_tupapp = novas.AppPlanet(jd_tt, sun, 0)
            obj_tuptopo = novas.TopoPlanet(jd_tt, sun, deltat, loc, 0)

            gast = novas.SiderealTime(jd_ut1, 0.0, deltat, 1, 1, 0)
            last = gast + float(cobs.longitude) / 15.0
            if (last >= 24.0):
                last -= 24.0
            if (last < 0.0):
                last += 24.0

                #theta = era (jd_ut1,0.0);
        finally:
            novas.EphemClose()
        return JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last

class MuserEphem(object):
    def __init__(self):
        self.sub_array = 1
        self.data_source = 0
        self.env = MuserEnv()

    def convert_date(self, cdate, ctime):
        ccdate = cdate + ' ' + ctime
        try:
            npos = ccdate.index('.')
            if npos < 0:
                dtDate = datetime.datetime.strptime(cdate + ' ' + ctime, '%Y-%m-%d %H:%M:%S')
                ms = 0.
            else:
                ccdate2 = ccdate[:npos]
                dtDate = datetime.datetime.strptime(ccdate2, '%Y-%m-%d %H:%M:%S')
                ms = float('0.'+ccdate[npos+1:])
        except:
            dtDate = datetime.datetime.strptime(cdate + ' ' + ctime, '%Y-%m-%d %H:%M:%S')
            ms = 0
        #print "CONVERT DATE:", dtDate.hour , dtDate.minute , dtDate.second,  ms
        dtTime = dtDate.hour + dtDate.minute / 60. + (dtDate.second + ms) / 3600.
        JD = novas.Cal2JD(dtDate.year, dtDate.month, dtDate.day, dtTime)
        mjd = int(JD - 2400000.5)

        return JD, mjd


    def covert_dms(self, DDMMSS):
        sign = True
        if DDMMSS[0] == '-':
            sign = False
            DDMMSS = DDMMSS[1:]
        dms = DDMMSS.split(':')

        dtTime = float(dms[0]) + float(dms[1]) / 60. + float(dms[2])/ 3600.
        if sign==False:
           dtTime = -dtTime
        return dtTime


    def delta_t(self, JulD):
        """
       self.delta_t(JD) returns a prediction ofself.delta_t
        for Julian Day JulD according to IERS Bulletin-A
        This will work for a few years from now: 20050123
        Be aware of TAI-UTC (leap seconds: diff3)

        delt_t = (TT-TAI) + (TAI-UTC) - (UT1-UTC) = (TT-UT1)
        Source: IERS Bulletin A Rapid Service/Prediction of
        Earth Orientation
        """
        pi = math.pi
        # Modified Julian Date
        MJD = JulD - 2400000.5
        IMJD = int(MJD)
        # Besselian year
        T = 2000.0 + (MJD - 51544.03) / 365.2422
        #TT-TAI
        diff0 = 32.184
        #UT2-UT1
        diff1 = 0.022 * math.sin(2 * pi * T) - 0.012 * math.cos(2 * pi * T) \
                - 0.006 * math.sin(4 * pi * T) + 0.007 * math.cos(4 * pi * T)
        #UT1-UTC
        #diff2 = 0.3278 - 0.00041 * (MJD - 53759) - diff1
        y,m,d,smjd,xx,yy,diff2 = self.getIERS(IMJD)
        #print "SMJD",y,m,d,smjd, diff2,xx, yy
        #diff2 =  0.318725
        #TAI-UTC
        #There has been a leap second in december 2005
        leapsec = novas.GetLeapSec(IMJD, 0)
        diff3 = leapsec[1]
        return diff0+diff3, diff0 + diff3  - diff2, diff2

    # from degree to (degree , minute)
    def d_to_dm(self, degree):
        "This function translates from degree to (degree , minute)"
        if degree < 0.0:
            neg = 1
        else:
            neg = 0

        degree = abs(degree)
        deg = int(degree)
        degree = degree - deg
        minute = degree * 60.0

        if neg:
            if deg > 0:
                deg = -deg
            else:
                minute = -minute
        return deg, minute

    # from degree to (degree ,min, sec)
    def d_to_dms(self, degree):
        "This function translates from degree to (degree , minute, second)"
        if degree < 0.0:
            neg = 1
        else:
            neg = 0
        degree = float(degree)
        degree = abs(degree)
        deg = int(degree)
        degree = degree - deg
        minute = int(degree * 60.0)
        degree = degree - minute / 60.0
        sec = degree * 3600

        if neg:
            if deg > 0:
                deg = -deg
            elif minute > 0:
                minute = -minute
            else:
                sec = -sec
        return deg, minute, sec


    # degree to degree within range 0..360
    def mod360(self, degree):
        "This function translates degree to degree within range 0..360"
        degree = math.fmod(degree, 360.0)
        if degree < 0.0:
            degree = degree + 360.0
        return degree

    # SiderealTime needs argument ee (equation of equinox) to calculate
    # apparent sidereal time (ee = 0.0 returns mean sidereal time)
    # ee is calculatedself with EarthTilt a low level novas function

    def eq_of_equinox(self, tjdate):
        """ SiderealTime needs argument ee (equation of equinox) to calculate
          apparent sidereal time (ee = 0.0 returns mean sidereal time)
          ee is calculated with EarthTilt a low level novas function"""

        out = novas.ETilt(tjdate, 0)
        eq_eq = out[2]
        return eq_eq

    def getIERS(self, MJD):

        #JD, xmjd =self.convert_date('2004-12-31','0:0:0')
        #print JD, xmjd
        iersFile = self.env.get_home_dir()

        if iersFile==None:
            iersFile=".."
        iersFile  = iersFile +"/data/iersdata.dat"
        d1 = 53370 #datetime.date(2004, 12, 31)
        #print "MMMJJJDDD",MJD

        num = MJD - d1
        with open(iersFile, 'rb') as f:
            f.seek(num*32)
            return struct.unpack('H',f.read(2))[0], struct.unpack('B',f.read(1))[0], struct.unpack('B',f.read(1))[0], struct.unpack('I',f.read(4))[0],struct.unpack('d',f.read(8))[0], struct.unpack('d',f.read(8))[0],struct.unpack('d',f.read(8))[0]
            f.close()




