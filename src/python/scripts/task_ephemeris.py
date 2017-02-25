import os
import os, sys
path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserephem import *
from pymuser.muserenv import *
from pymuser.muserobs import muserobservatory
class ephemeris:
    def __init__(self, cdate=None, ctime=None, cplanet=None, ra=0., dec=0., debug=None):
        self.cdate = cdate
        self.ctime = ctime
        self.cplanet = cplanet
        self.debug = debug
        self.ra = ra
        self.dec = dec
        self.muser_name = muserenv.muser_name
        self.longitude,self.latitude,self.altitude = muserobservatory.get_muser_observatory()  # (42.211833333333333, 115.2505, 1365)
        if (self.ra == 0. or self.dec == 0.):
            return self.ccli_planet_DATE_TIME_PLANET()
        else:
            return self.ccli_fixed_DATE_TIME_STRING_RA_DEC()
    def ccli_planet_DATE_TIME_PLANET(self):
        """Calculate apparent position of planet
        Date of the observation, ISO date format: YYYY-MM-DD
        Time of the observation, format: HH:MM:SS.ssss
        Observational object (i.e., mercury|venus|earth|mars|jupiter|saturn|uranus|neptune|pluto|sun|moon)
        """
        global DATAPATH
        obs = Observatory(self.muser_name, self.longitude, self.latitude,self.altitude, 0., 0.)
        e = MuserEphem()

        print "---------------------------------------------------------------------"
        print "PLANET:\t", self.cplanet, "\tCurrent Ephemris: JPL/DE405\n"
        print " "
        print "Current Observatory information:\n"
        print "Name:\t\t", self.muser_name
        dd, mm, ss = e.d_to_dms(self.longitude)
        print "Longitude\t", self.longitude,
        print "(%4d D %2d M %f S)" % (dd, mm, ss)
        dd, mm, ss = e.d_to_dms(self.latitude)
        print "Latitude\t", self.latitude,
        print "(%4d D %2d M %f S)" % (dd, mm, ss)
        print "Height\t\t", self.altitude, ' (Meters)'
        print " \n"

        planet = Body(name=self.cplanet)
        JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last = planet.compute(cobs=obs, cdate=self.cdate, ctime=self.ctime)

        print "Julian Date\t", JD
        print "DeltaT\t\t", deltat, " (s)"
        print "Equinox\t\t", eqeq
        print " \n"

        print self.cplanet + " geocentric and topocentric positions:\n"
        #print obj.name, obj_tup
        print "      RA                   DEC                   DIST"
        print "%15.10f        %15.10f        %15.12f\n" % ( obj_tupapp[1], obj_tupapp[2], obj_tupapp[3])
        print "%15.10f        %15.10f        %15.12f\n" % ( obj_tuptopo[1], obj_tuptopo[2], obj_tuptopo[3])

        print "      RA(HMS)              DEC(DMS)"
        rad, ram, ras = e.d_to_dms(obj_tupapp[1])
        decd, decm, decs = e.d_to_dms(obj_tupapp[2])


        print "%-3dH%2dM%8.4f          %-3dD%2dM%8.4f\n" % (rad, ram, ras, decd, decm, decs)

        rad, ram, ras = e.d_to_dms(obj_tuptopo[1])
        decd, decm, decs = e.d_to_dms(obj_tuptopo[2])

        print "%-3dH%2dM%8.4f          %-3dD%2dM%8.4f\n" % (rad, ram, ras, decd, decm, decs)

        print ("Greenwich and local sidereal time:\n")
        print ("%16.11f        %16.11f\n" % ( gast, last))
        print ("\n");

        print
    def ccli_fixed_DATE_TIME_STRING_RA_DEC(self):
        """Calculate apparent position of fixed body based on mean position
        Date of the observation, ISO date format: YYYY-MM-DD
        Time of the observation, format: HH:MM:SS.ssss
        Observational object (i.e., star name )
        Right Asceniton (float format)
        Declination (float format)
        """
        global DATAPATH
        e = MuserEphem()
        obs = Observatory(self.muser_name, self.longitude, self.latitude, self.altitude, 0., 0.)

        print "---------------------------------------------------------------------"
        print "Fixed Object:\t", self.cplanet
        print " "
        print "Current Observatory information:\n"
        print "Name:\t\t", self.muser_name
        dd, mm, ss = e.d_to_dms(self.longitude)
        print "Longitude\t", self.longitude,
        print "(%4d D %2d M %f S)" % (dd, mm, ss)
        dd, mm, ss = e.d_to_dms(self.latitude)
        print "Latitude\t", self.latitude,
        print "(%4d D %2d M %f S)" % (dd, mm, ss)
        print "Height\t\t", self.altitude, ' (Meters)'
        print " \n"



        planet = FixedBody(self.planet, self.ra, self.dec)
        JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last = planet.compute(cobs=obs, obsdate=self.cdate, obstime=self.ctime)

        print "Julian Date\t", JD
        print "DeltaT\t\t", deltat, " (s)"
        print "Equinox\t\t", eqeq
        print " \n"

        print self.cplanet + " geocentric and topocentric positions:\n"
        #print obj.name, obj_tup
        print "      RA                   DEC"
        print "%15.10f        %15.10f\n" % ( obj_tupapp[1], obj_tupapp[2])

        print "%15.10f        %15.10f\n" % ( obj_tuptopo[1], obj_tuptopo[2])
        #print obj.name, obj_tup
        print "      RA(HMS)         DEC(DMS)"

        rad, ram, ras = e.d_to_dms(obj_tupapp[1])
        decd, decm, decs = e.d_to_dms(obj_tupapp[2])

        print "%-3dH%2dM%8.4f %-3dD%2dM%8.4f\n" % (rad, ram, ras, decd, decm, decs)

        print ("Greenwich and local sidereal time:\n")
        print ("%16.11f        %16.11f\n" % ( gast, last))
        print ("\n")

        print
