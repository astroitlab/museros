"""An interface between Skyfield and the Python ``sgp4`` library."""

from numpy import array, cross, einsum, zeros_like
from earth_gravity import wgs72
from io import twoline2rv

from propagation import sgp4
from constants import AU_KM, DAY_S, T0, tau
from .functions import rot_x, rot_y, rot_z
from .positionlib import Apparent, Geocentric, ITRF_to_GCRS
from timelib import JulianDate, takes_julian_date



# important ones:
# jdsatepoch
# bstar
# inclo - inclination
# nodeo - right ascension of ascending node
# ecco - eccentricity
# argpo - argument of perigee
# mo - mean anomaly
# no - mean motion

_minutes_per_day = 1440.

class EarthSatellite(object):
    """An Earth satellite loaded from a TLE file and propagated with SGP4."""

    def __init__(self, lines, earth):
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
        rGCRS = ITRF_to_GCRS(jd, rITRF)
        vGCRS = zeros_like(rGCRS)  # todo: someday also compute vGCRS?

        return rGCRS, vGCRS, error

    @takes_julian_date
    def gcrs(self, jd):
        """Return a GCRS position for this Earth satellite.

        Uses standard SGP4 theory to predict the satellite location.

        """
        position_AU, velociy_AU_per_d, error = self._compute_GCRS(jd)
        g = Geocentric(position_AU, velociy_AU_per_d, jd)
        g.sgp4_error = error
        return g

    def _observe_from_bcrs(self, observer):
        # TODO: what if someone on Mars tries to look at the ISS?

        jd = observer.jd
        rGCRS, vGCRS, error = self._compute_GCRS(jd)
        rGCRS - observer.rGCRS
        vGCRS - observer.vGCRS
        g = Apparent(rGCRS - observer.rGCRS, vGCRS - observer.vGCRS, jd)
        g.sgp4_error = error
        g.observer = observer
        # g.distance = euclidian_distance
        return g


_second = 1.0 / (24.0 * 60.0 * 60.0)

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
