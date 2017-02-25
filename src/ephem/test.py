#!/usr/bin/env python

import math
import novas


date = (2006, 3, 29, 10.30)


def delta_t(JulD):
    """
    delta_t(JD) returns a prediction of delta_t
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
    # Besselian year
    T = 2000.0 + (MJD - 51544.03) / 365.2422
    #TT-TAI
    diff0 = 32.184
    #UT2-UT1
    diff1 = 0.022 * math.sin(2 * pi * T) - 0.012 * math.cos(2 * pi * T) \
            - 0.006 * math.sin(4 * pi * T) + 0.007 * math.cos(4 * pi * T)
    #UT1-UTC
    diff2 = 0.3278 - 0.00041 * (MJD - 53759) - diff1
    #TAI-UTC
    #There has been a leap second in december 2005
    diff3 = 33.0000000

    return diff0 + diff3 - diff2

# from degree to (degree , minute)
def d_to_dm(degree):
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
def d_to_dms(degree):
    "This function translates from degree to (degree , minute, second)"
    if degree < 0.0:
        neg = 1
    else:
        neg = 0

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
def mod360(degree):
    "This function translates degree to degree within range 0..360"
    degree = math.fmod(degree, 360.0)
    if degree < 0.0:
        degree = degree + 360.0
    return degree

# SiderealTime needs argument ee (equation of equinox) to calculate
# apparent sidereal time (ee = 0.0 returns mean sidereal time)
# ee is calculated with EarthTilt a low level novas function

def eq_of_equinox(tjdate):
    """ SiderealTime needs argument ee (equation of equinox) to calculate
      apparent sidereal time (ee = 0.0 returns mean sidereal time)
      ee is calculated with EarthTilt a low level novas function"""

    out = novas.ETilt(tjdate, 0)
    eq_eq = out[2]
    return eq_eq


y, m, d, t = date
ZJD = novas.Cal2JD(y,m,d,0)
JD = novas.Cal2JD(y, m, d, t)
mjd = int (ZJD - 2400000.5)
deltat = delta_t(JD)
tjd = JD + deltat / 86400.0

y, m, d, t = novas.JD2Cal(JD)
print "Almanac entries for: %d %d %d %4.2f" % (y, m, d, t)
print "JD = ", JD, " tjd = ", tjd, mjd

leapsec = novas.GetLeapSec('leapsec.tab',mjd,0)
print 'leapsec:', leapsec

cat = novas.Cat()
cat.catalog='xxx'
cat.starname = 'DUMMY'
cat.starnumber = 0
cat.ra =0.0
cat.dec = 0.0
cat.promora = 0.0
cat.promodec = 0.0
cat.parallax = 0.0
cat.radialvelocity = 0.0


loc = novas.On_surface()
# Emmen, The Netherlands
loc.latitude = 42.0
loc.longitude = -70.
loc.height = 0.0
loc.temperature =10.0
loc.pressure = 1010.0


# Moon
moon = novas.Object()
moon.type = 0
moon.number = 11
moon.name = "Moon"
moon.star = cat
#moon.radius = 1737.4

# Sun
sun = novas.Object()
sun.type = 0
sun.number = 10
sun.name = "Sun"
sun.star = cat
#sun_radius = 695500.0


# Earth
earth = novas.Object()
earth.type = 0
earth.number = 3
earth.star = cat


print "Test 1 AppPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))"
# AppPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
# See almanac.py
print "tjd", tjd
eqeq = eq_of_equinox(tjd)
print "equation of equinox ", eqeq
ast = 0.0
deltat = 34.0
eqeq = novas.SiderealTime(math.floor(JD), JD - math.floor(JD), deltat, 1, 1, 0)
print "eqeq", eqeq
fil=novas.EphemOpen('JPLEPH')
obj_tup = novas.AppPlanet(tjd, moon, 0)
print obj_tup
sha = 360.0 - obj_tup[0] * 15.0
eadst = (0.0 + obj_tup[2])
# * novas.KMAU
gha = mod360(sha + 360 * ast / 24.0)
gha_s = "%3d %5.2f  " % d_to_dm(gha)
decl_s = ""
# "%3d %5.2f  " % d_to_dm(obj_tup[1])
#sd_s =  "%2d %5.2f  " % d_to_dm(180 * math.atan(moon.radius/eadst)/ math.pi)
#str_buf =  gha_s + decl_s + sd_s

#print str_buf


print "Test 2 TopoPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))"

# TopoPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
for obj in [sun, moon]:
    obj_tup = novas.TopoPlanet(tjd, obj, deltat, loc, 0)
    print obj.name, obj_tup
    ra_s = "%3d %2d %6.3f  " % d_to_dms(obj_tup[0])
    decl_s = "%3d %2d %5.2f  " % d_to_dms(obj_tup[1])
    print  ra_s + decl_s, obj_tup[2]
    print

novas.EphemClose()
