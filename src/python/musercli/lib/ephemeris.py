#!/usr/bin/env python
# Modle for JPL Ephemeris 4.5

import privileges
import pymuser.muserephem as ephem

get_cli().set_privilege(privileges.ENABLE, privileges.EPHEMERIS, 'ccli_planet_DATE_TIME_PLANET')
get_cli().set_privilege(privileges.ENABLE, privileges.EPHEMERIS, 'ccli_fixed_DATE_TIME_STRING_RA_DEC')


def ccli_planet_DATE_TIME_PLANET(cdate, ctime, cplanet):
    """Calculate apparent position of planet
    Date of the observation, ISO date format: YYYY-MM-DD
    Time of the observation, format: HH:MM:SS.ssss
    Observational object (i.e., mercury|venus|earth|mars|jupiter|saturn|uranus|neptune|pluto|sun|moon)
    """
    global DATAPATH
    obs = ephem.Observatory(get_cli().get_obsname(), get_cli().get_obslongitude(), get_cli().get_obslatitude(),
                            get_cli().get_obsaltitude(), 0., 0.)

    ephem.DATAPATH = DATAPATH
    e = ephem.MuserEphem()

    print "---------------------------------------------------------------------"
    print "PLANET:\t", cplanet, "\tCurrent Ephemris: JPL/DE405\n"
    print " "
    print "Current Observatory information:\n"
    print "Name:\t\t", obs.name
    dd, mm, ss = e.d_to_dms(obs.longitude)
    print "Longitude\t", obs.longitude,
    print "(%4d D %2d M %f S)" % (dd, mm, ss)
    dd, mm, ss = e.d_to_dms(obs.latitude)
    print "Latitude\t", obs.latitude,
    print "(%4d D %2d M %f S)" % (dd, mm, ss)
    print "Height\t\t", obs.altitude, ' (Meters)'
    print " \n"

    planet = ephem.Body(name=cplanet)
    JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last = planet.compute(cobs=obs, cdate=cdate, ctime=ctime)

    print "Julian Date\t", JD
    print "DeltaT\t\t", deltat, " (s)"
    print "Equinox\t\t", eqeq
    print " \n"

    print cplanet + " geocentric and topocentric positions:\n"
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


def ccli_fixed_DATE_TIME_STRING_RA_DEC(cdate, ctime, cplanet, ra, dec):
    """Calculate apparent position of fixed body based on mean position
    Date of the observation, ISO date format: YYYY-MM-DD
    Time of the observation, format: HH:MM:SS.ssss
    Observational object (i.e., star name )
    Right Asceniton (float format)
    Declination (float format)
    """
    global DATAPATH
    obs = ephem.Observatory(get_cli().get_obsname(), get_cli().get_obslongitude(), get_cli().get_obslatitude(),
                            get_cli().get_obsaltitude(), 0., 0.)

    ephem.DATAPATH = DATAPATH
    print "---------------------------------------------------------------------"
    print "Fixed Object:\t", cplanet
    print " "
    print "Current Observatory information:\n"
    print "Name:\t\t", obs.name
    dd, mm, ss = ephem.d_to_dms(obs.longitude)
    print "Longitude\t", obs.longitude,
    print "(%4d D %2d M %f S)" % (dd, mm, ss)
    dd, mm, ss = ephem.d_to_dms(obs.latitude)
    print "Latitude\t", obs.latitude,
    print "(%4d D %2d M %f S)" % (dd, mm, ss)
    print "Height\t\t", obs.altitude, ' (Meters)'
    print " \n"

    e = ephem.MuserEphem()

    planet = ephem.FixedBody(cplanet, ra, dec)
    JD, deltat, eqeq, obj_tupapp, obj_tuptopo, gast, last = planet.compute(cobs=obs, obsdate=cdate, obstime=ctime)

    print "Julian Date\t", JD
    print "DeltaT\t\t", deltat, " (s)"
    print "Equinox\t\t", eqeq
    print " \n"

    print cplanet + " geocentric and topocentric positions:\n"
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
