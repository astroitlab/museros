import os, sys
import novas
sys.path.extend([os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')])


from numpy import array, cross, einsum, zeros_like
from numpy import arcsin, arctan2, array, cos, sin, sqrt

from sgp4gravity import wgs72
from sgp4io import twoline2rv
from muserconstants import AU_KM, DAY_S, T0, tau, TAU
from musertime import JulianDate
from sgp4propagation import sgp4
import pymuser.muserephem as ephem


# important ones:
# jdsatepoch
# bstar
# inclo - inclination
# nodeo - right ascension of ascending node
# ecco - eccentricity
# argpo - argument of perigee
# mo - mean anomaly
# no - mean motion


class EarthSatellite(object):
    """An Earth satellite loaded from a TLE file and propagated with SGP4."""

    def __init__(self, text, earth):
        lines = text.splitlines()
        sat = twoline2rv(*lines[-2:], whichconst=wgs72)
        self._sgp4_satellite = sat
        self._earth = earth
        self.epoch = JulianDate(utc=(sat.epochyr, 1, sat.epochdays - 1.0))

    def __repr__(self):
        sat = self._sgp4_satellite
        return '<EarthSatellite number={1!r} epoch={0}>'.format(
            self.epoch.utc_iso(), sat.satnum)

    def _position_and_velocity_TEME_km(self, jd):
        """Return the raw true equator mean equinox (TEME) vectors from SGP4.

        Returns a tuple of NumPy arrays ``([x y z], [xdot ydot zdot])``
        expressed in kilometers and kilometers per second.  Note that we
        assume the TLE epoch to be a UTC date, per AIAA 2006-6753.

        """
        sat = self._sgp4_satellite
        epoch = sat.jdsatepoch
        minutes_past_epoch = (jd._utc_float() - epoch) * 1440.
        if getattr(minutes_past_epoch, 'shape', None):
            position = []
            velocity = []
            error = []
            for m in minutes_past_epoch:
                p, v = sgp4(sat, m)
                position.append(p)
                velocity.append(v)
                error.append(sat.error_message)
            return array(position).T, array(velocity).T, error
        else:
            position, velocity = sgp4(sat, minutes_past_epoch)
            return array(position), array(velocity), sat.error_message

    def _compute_GCRS(self, jd):
        """Compute where satellite is in space on a given date."""

        rTEME, vTEME, error = self._position_and_velocity_TEME_km(jd)
        rTEME /= AU_KM
        vTEME /= AU_KM
        vTEME *= DAY_S

        rITRF, vITRF = TEME_to_ITRF(jd.ut1, rTEME, vTEME)
        #print "RITRF:", rITRF, rITRF.shape
        a = rITRF.tolist()
        print "ITRF"
        print a

        rUz = novas.Ter2Cel(jd.ut1, 0.0, jd.delta_t, 1, 0, 1, 0, 0, a[0], a[1], a[2]);
        #print rUz, len(rUz)
        print "TER"
        print rUz
        stat, ra, dec = novas.Vector2Radec(rUz[1],rUz[2], rUz[3])

        print "RE: ITRS"
        stat, b1, b2, b3 =  novas.Cel2Ter(jd.ut1, 0, jd.delta_t, 1, 0, 0, 0, 0, rUz[1],rUz[2], rUz[3])
        print b1,b2, b3
        #stat, raequ, decequ = novas.GCRS2Equ (jd.tt, 1, 0, ra, dec)
        #print raequ, decequ
        # stat, lon, lat = novas.Equ2Ecl (jd.tt, 2, 0, ra, dec)
        #
        # print lon, lat 102.7972, 25.0299, 1991.83

        loc = novas.On_surface()
        # Emmen, The Netherlands
        loc.latitude = 25.0299
        loc.longitude = 102.7972
        loc.height = 1991.83
        loc.temperature = 0.0
        loc.pressure = 0

        # print novas.Equ2Hor (jd.ut1, jd.delta_t, 0, 0,0,
        #          loc, raequ, decequ, 0)
        print novas.Vector2Radec(b1, b2, b3)

        return ra, dec
        #print ra,dec
        # rGCRS = ITRF_to_GCRS(jd, rITRF)
        #vGCRS = zeros_like(rGCRS)  # todo: someday also compute vGCRS?

        #return rGCRS, vGCRS, error

    def gcrs(self, jd):
        """Return a GCRS position for this Earth satellite.

        Uses standard SGP4 theory to predict the satellite location.

        """
        position_AU, velociy_AU_per_d, error = self._compute_GCRS(jd)
        # g = Geocentric(position_AU, velociy_AU_per_d, jd)
        #g.sgp4_error = error
        return position_AU, velociy_AU_per_d, error  #g

    def _observe_from_bcrs(self, observer):
        # TODO: what if someone on Mars tries to look at the ISS?

        jd = observer.jd
        rGCRS, vGCRS, error = self._compute_GCRS(jd)
        rGCRS - observer.rGCRS
        vGCRS - observer.vGCRS
        # g = Apparent(rGCRS - observer.rGCRS, vGCRS - observer.vGCRS, jd)
        # g.sgp4_error = error
        # g.observer = observer
        # # g.distance = euclidian_distance
        # return g


_second = 1.0 / (24.0 * 60.0 * 60.0)


def ITRF_to_GCRS(jd, rITRF):  # todo: velocity

    # Todo: wobble

    spin = rot_z(jd.gast * TAU / 24.0)
    position = einsum('ij...,j...->i...', spin, array(rITRF))
    return einsum('ij...,j...->i...', jd.MT, position)


def rot_x(theta):
    c = cos(theta)
    s = sin(theta)
    return array([(1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)])


def rot_y(theta):
    c = cos(theta)
    s = sin(theta)
    return array([(c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)])


def rot_z(theta):
    c = cos(theta)
    s = sin(theta)
    zero = theta * 0.0
    one = zero + 1.0
    return array(((c, -s, zero), (s, c, zero), (zero, zero, one)))


def theta_GMST1982(jd_ut1):
    """Return the angle of Greenwich Mean Standard Time 1982 given the JD.

    This angle defines the difference between the idiosyncratic True
    Equator Mean Equinox (TEME) frame of reference used by SGP4 and the
    more standard Pseudo Earth Fixed (PEF) frame of reference.

    From AIAA 2006-6753 Appendix C.

    """
    t = (jd_ut1 - T0) / 36525.0
    g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    theta = (jd_ut1 % 1.0 + g * _second % 1.0) * tau
    theta_dot = (1.0 + dg * _second / 36525.0) * tau
    return theta, theta_dot


def TEME_to_ITRF(jd_ut1, rTEME, vTEME, xp=0.0, yp=0.0):
    """Convert TEME position and velocity into standard ITRS coordinates.

    This converts a position and velocity vector in the idiosyncratic
    True Equator Mean Equinox (TEME) frame of reference used by the SGP4
    theory into vectors into the more standard ITRS frame of reference.
    The velocity should be provided in units per day, not per second.

    From AIAA 2006-6753 Appendix C.

    """
    theta, theta_dot = theta_GMST1982(jd_ut1)
    zero = theta_dot * 0.0
    angular_velocity = array([zero, zero, -theta_dot])
    R = rot_z(-theta)

    if len(rTEME.shape) == 1:
        rPEF = (R).dot(rTEME)
        vPEF = (R).dot(vTEME) + cross(angular_velocity, rPEF)
    else:
        rPEF = einsum('ij...,j...->i...', R, rTEME)
        vPEF = einsum('ij...,j...->i...', R, vTEME) + cross(
            angular_velocity, rPEF, 0, 0).T

    if xp == 0.0 and yp == 0.0:
        rITRF = rPEF
        vITRF = vPEF
    else:
        W = (rot_x(yp)).dot(rot_y(xp))
        rITRF = (W).dot(rPEF)
        vITRF = (W).dot(vPEF)
    return rITRF, vITRF


if __name__ == '__main__':
    line = '''
    ISS (ZARYA)
    1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
    2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
    '''
    text = '''
0 SIRIO
1 10294U 77080A   15117.62524215 -.00000121  00000-0  00000-0 0 09995
2 10294 014.3697 340.9030 0006229 080.3260 094.2763 01.00274599077120
'''
    sat = EarthSatellite(text, None)
    jd = JulianDate(utc=(2015, 4, 28, 5, 50, 33))
    print sat._compute_GCRS(jd)
    # satellite = twoline2rv(line1, line2, wgs72)


    # position, velocity = satellite.propagate(2000, 6, 29, 12, 50, 19)
    # print(satellite.error)
    # print(satellite.error_message)
    # print(position)
    # print(velocity)