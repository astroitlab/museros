#!/usr/bin/env python

import math
import novas


date = (2006, 3, 29, 0.0)

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
    diff1 = 0.022*math.sin(2*pi*T)-0.012*math.cos(2*pi*T)\
        -0.006*math.sin(4*pi*T)+0.007*math.cos(4*pi*T)
#UT1-UTC
    diff2 = 0.3278 - 0.00041*(MJD-53759) - diff1
#TAI-UTC
#There has been a leap second in december 2005
    diff3 = 33.0000000

    return diff0 + diff3 - diff2


y, m, d, t = date
JD = novas.Cal2JD(y, m, d, t)
deltat = delta_t(JD)
tjd = JD + deltat / 86400.0

y, m, d, t = novas.JD2Cal(JD)
print "Almanac entries for: %d %d %d %4.2f" % (y, m, d, t)
print "JD = ", JD, " tjd = ", tjd

# Manavgat Turkey
loc = novas.Location()
loc.lat = 36.75
loc.lon = 32.40
loc.h = 0.0


# Moon
moon = novas.Body()
moon.type = 0
moon.number = 11
moon.name = "Moon"
moon.radius = 1737.4

# Sun
sun = novas.Body()
sun.type = 0
sun.number = 10
sun.name = "Sun"
sun.radius = 695500.0


# Earth
earth = novas.Body()
earth.type = 0
earth.number = 3

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

    out = novas.EarthTilt(tjdate)
    eq_eq = out[2]
    return eq_eq

   

header = """UT    RA             Decl         Semi-Di     RA             Decl         Semi-Di
      Moon                                    Sun
"""

# AppPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))
# TopoPlanet returns a tuple (RA (h), Decl (deg), Dist (AU))

def format(obj):
    "Returns a formatted string"

    obj_tup = novas.TopoPlanet(tjd, obj, earth, deltat, loc)
    RA_s = "%3d %2d %6.3f  " % d_to_dms(obj_tup[0])
    Decl_s = "%3d %2d %5.2f  " % d_to_dms(obj_tup[1])
    eadst = (0.0 + obj_tup[2]) * novas.KMAU
    sd_s =  "%2d %5.2f  " % d_to_dm(180 * math.atan(obj.radius/eadst)/ math.pi)
    str_buf =  RA_s + Decl_s + sd_s
    return str_buf


print "Eclipse Manavgat Turkey 29032006"
print
print header
for i in range(25):
    print "%02i " % i,
    print format(moon), format(sun)
    tjd += (1.0 / 24.0)
