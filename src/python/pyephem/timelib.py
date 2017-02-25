from datetime import date, datetime, timedelta, tzinfo
from time import strftime

from numpy import array, einsum, rollaxis, searchsorted, sin, where, zeros_like

from constants import T0, DAY_S
from .earthlib import sidereal_time
from .framelib import ICRS_to_J2000 as B
from .nutationlib import compute_nutation, earth_tilt
from .precessionlib import compute_precession


try:
    from pytz import utc
except ImportError:

    class UTC(tzinfo):
        'UTC'
        zero = timedelta(0)
        def utcoffset(self, dt):
            return self.zero
        def tzname(self, dt):
            return 'UTC'
        def dst(self, dt):
            return self.zero

    utc = UTC()

# Much of the following code is adapted from the USNO's "novas.c".

_half_second = 0.5 / DAY_S
_half_millisecond = 0.5e-3 / DAY_S
_half_microsecond = 0.5e-6 / DAY_S
_months = array(['Month zero', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

extra_documentation = """

        This routine takes a date as its argument.  You can either
        provide a `jd=` keyword argument with a `JulianDate` you have
        built yourself, or use one of these keyword arguments::

            # Coordinated Universal Time
            utc=(1973, 12, 29, 23, 59, 48.0)
            utc=datetime(1973, 12, 29, 23, 59, 48.0)

            # International Atomic Time
            tai=2442046.5

            # Terrestrial Time
            tt=2442046.5

"""

def takes_julian_date(function):
    """Wrap `function` so it accepts the standard Julian date arguments.

    A function that takes two arguments, `self` and `jd`, may be wrapped
    with this decorator if it wants to support optional auto-creation of
    its `jd` argument by accepting all of the same keyword arguments
    that the JulianDate constructor itself supports.

    """
    def wrapper(self, jd=None, utc=None, tai=None, tt=None, tdb=None,
                delta_t=0.0, cache=None):
        if jd is None:
            jd = JulianDate(utc, tai, tt, tdb, delta_t, cache)
        elif not isinstance(jd, JulianDate):
            s = 'your "jd" argument is not a JulianDate: {0!r}'.format(jd)
            raise ValueError(s)
        return function(self, jd)
    wrapper.__name__ = function.__name__
    synopsis, blank_line, description = function.__doc__.partition('\n\n')
    wrapper.__doc__ = synopsis + extra_documentation + description
    return wrapper

def _to_array(value):
    """When `value` is a plain Python sequence, return it as a NumPy array."""
    if not hasattr(value, 'shape') and hasattr(value, '__len__'):
        return array(value)
    else:
        return value

tt_minus_tai = array(32.184 / DAY_S)

class JulianDate(object):
    """A single date and time, or an array, stored as a Julian date.

    You can import this class from ``skyfield.api``.  For the Julian
    date of the current date and time, use the separate function
    ``skyfield.api.now()``.

    Every Julian date object understands five different time scales,
    which can be used during instantiation::

        JulianDate(utc=(year, month, day, hour, minute, second))
                       or Standard Library datetime or date)
        JulianDate(tai=(year, month, day, ...) or float)
        JulianDate(tt=(year, month, day, ...) or float)
        JulianDate(tdb=(year, month, day, ...) or float)
        JulianDate(ut1=(year, month, day, ...) or float)

    """
    def __init__(self, utc=None, tai=None, tt=None, tdb=None,
                 delta_t=0.0, cache=None):

        self.delta_t = _to_array(delta_t)

        if cache is None:
            from data import cache
        self.cache = cache

        if tai is None and utc is not None:
            leap_dates, leap_offsets = cache.run(usno_leapseconds)
            if isinstance(utc, datetime):
                tai = _utc_datetime_to_tai(leap_dates, leap_offsets, utc)
            elif isinstance(utc, date):
                tai = _utc_date_to_tai(leap_dates, leap_offsets, utc)
            elif isinstance(utc, tuple):
                values = [_to_array(value) for value in utc]
                tai = _utc_to_tai(leap_dates, leap_offsets, *values)
            else:
                tai = array([
                    _utc_datetime_to_tai(leap_dates, leap_offsets, dt)
                    for dt in utc])

        if tai is not None:
            if isinstance(tai, tuple):
                tai = julian_date(*tai)
            self.tai = _to_array(tai)
            if tt is None:
                tt = tai + tt_minus_tai

        if tdb is not None:
            if isinstance(tdb, tuple):
                tdb = julian_date(*tdb)
            self.tdb = _to_array(tdb)
            if tt is None:
                tt = tdb - tdb_minus_tt(tdb) / DAY_S

        if tt is None:
            raise ValueError('You must supply either utc, tai, tt, or tdb'
                             ' when building a JulianDate')
        elif isinstance(tt, tuple):
            tt = julian_date(*tt)

        self.tt = _to_array(tt)
        self.shape = getattr(self.tt, 'shape', ())
        self.delta_t = delta_t

    def __repr__(self):
        return '<JulianDate tt={0}>'.format(self.tt)

    def __getitem__(self, index):
        # TODO: also copy cached matrices?
        jd = JulianDate(tt=self.tt[index])
        for name in 'tai', 'tdb', 'ut1', 'delta_t':
            value = getattr(self, name, None)
            if value is not None:
                if getattr(value, 'shape', None):
                    value = value[index]
                setattr(jd, name, value)
        return jd

    def astimezone(self, tz):
        """Return as a Python ``datetime`` in a ``pytz`` provided timezone.

        Convert this Julian date to a ``datetime`` in the timezone `tz`,
        which should be one of the timezones provided by the third-party
        ``pytz`` package.  If this Julian date is an array, then an
        array of datetimes is returned instead of a single value.

        """
        dt, leap_second = self.astimezone_and_leap_second(tz)
        return dt

    def astimezone_and_leap_second(self, tz):
        """Return as a ``datetime`` plus leap second in a ``pytz`` timezone.

        Convert this Julian date to a ``datetime`` and a leap second::

            dt, leap_second = jd.astimezone_and_leap_second(tz)

        The argument `tz` should be a timezone from the third-party
        ``pytz`` package, which must be installed separately.  The date
        and time returned will be for that time zone.

        The leap second value is provided because a Python ``datetime``
        can only number seconds ``0`` through ``59``, but leap seconds
        have a designation of at least ``60``.  The leap second return
        value will normally be ``0``, but will instead be ``1`` if the
        date and time are a UTC leap second.  Add the leap second value
        to the ``second`` field of the ``datetime`` to learn the real
        name of the second.

        If this Julian date is an array, then an array of ``datetime``
        objects and an array of leap second integers is returned,
        instead of a single value each.

        """
        dt, leap_second = self.utc_datetime_and_leap_second()
        normalize = getattr(tz, 'normalize', None)
        if self.shape and normalize is not None:
            dt = array([normalize(d.astimezone(tz)) for d in dt])
        elif self.shape:
            dt = array([d.astimezone(tz) for d in dt])
        elif normalize is not None:
            dt = normalize(dt.astimezone(tz))
        else:
            dt = dt.astimezone(tz)
        return dt, leap_second

    def toordinal(self):
        """Return the proleptic Gregorian ordinal of the TAI date.

        This method makes Skyfield `JulianDate` objects compatible with
        Python `datetime` objects, which also provide a ``toordinal()``
        method.  Thanks to this method, a `JulianDate` can often be used
        directly as a coordinate for a plot.

        """
        return self._utc_float() - 1721424.5

    def utc_datetime(self):
        """Return a Python ``datetime`` for this Julian, expressed as UTC.

        If the third-party ``pytz`` package is available, then its
        ``utc`` timezone will be used as the timezone of the return
        value.  Otherwise, an equivalent Skyfield ``utc`` timezone
        object is used.  If this Julian date is an array, then a
        sequence of datetimes is returned instead of a single value.

        """
        dt, leap_second = self.utc_datetime_and_leap_second()
        return dt

    def utc_datetime_and_leap_second(self):
        """Return a ``datetime`` in UTC, plus a leap second value.

        Convert this Julian date to a ``datetime`` and a leap second::

            dt, leap_second = jd.utc_datetime_and_leap_second()

        If the third-party ``pytz`` package is available, then its
        ``utc`` timezone will be used as the timezone of the return
        value.  Otherwise, Skyfield uses its own ``utc`` timezone.

        The leap second value is provided because a Python ``datetime``
        can only number seconds ``0`` through ``59``, but leap seconds
        have a designation of at least ``60``.  The leap second return
        value will normally be ``0``, but will instead be ``1`` if the
        date and time are a UTC leap second.  Add the leap second value
        to the ``second`` field of the ``datetime`` to learn the real
        name of the second.

        If this Julian date is an array, then an array of ``datetime``
        objects and an array of leap second integers is returned,
        instead of a single value each.

        """
        year, month, day, hour, minute, second = self._utc_tuple(
            _half_millisecond)
        second, fraction = divmod(second, 1.0)
        second = second.astype(int)
        leap_second = second // 60
        second -= leap_second
        milli = (fraction * 1000).astype(int) * 1000
        if self.shape:
            utcs = [utc] * self.shape[0]
            argsets = zip(year, month, day, hour, minute, second, milli, utcs)
            dt = array([datetime(*args) for args in argsets])
        else:
            dt = datetime(year, month, day, hour, minute, second, milli, utc)
        return dt, leap_second

    def utc_iso(self, places=0):
        """Return an ISO 8601 string like ``2014-01-18T01:35:38Z`` in UTC.

        If this Julian date is an array of dates, then a sequence of
        strings is returned instead of a single string.

        """
        if places:
            power_of_ten = 10 ** places
            offset = _half_second / power_of_ten
            year, month, day, hour, minute, second = self._utc_tuple(offset)
            second, fraction = divmod(second, 1.0)
            fraction *= power_of_ten
            format = '%%04d-%%02d-%%02dT%%02d:%%02d:%%02d.%%0%ddZ' % places
            args = (year, month, day, hour, minute, second, fraction)
        else:
            format = '%04d-%02d-%02dT%02d:%02d:%02dZ'
            args = self._utc_tuple(_half_second)

        if self.shape:
            return [format % tup for tup in zip(*args)]
        else:
            return format % args

    def utc_jpl(self):
        """Convert to a string like ``A.D. 2014-Jan-18 01:35:37.5000 UT``.

        Returns a string for this date and time in UTC, in the format
        used by the JPL HORIZONS system.  If this Julian date is an
        array of dates, then a sequence of strings is returned instead
        of a single string.

        """
        offset = _half_second / 1e4
        year, month, day, hour, minute, second = self._utc_tuple(offset)
        second, fraction = divmod(second, 1.0)
        fraction *= 1e4
        bc = year < 1
        year = abs(year - bc)
        era = where(bc, 'B.C.', 'A.D.')
        format = '%s %04d-%s-%02d %02d:%02d:%02d.%04d UT'
        args = (era, year, _months[month], day, hour, minute, second, fraction)

        if self.shape:
            return [format % tup for tup in zip(*args)]
        else:
            return format % args

    def utc_strftime(self, format):
        """Format this UTC time according to a Python date-formatting string.

        This internally calls the Python ``strftime()`` routine from the
        Standard Library ``time()`` module, for which you can find a
        quick reference at ``http://strftime.org/``.  If this Julian
        date is an array of dates, then a sequence of strings is
        returned instead of a single string.

        """
        tup = self._utc_tuple(_half_second)
        year, month, day, hour, minute, second = tup
        second = second.astype(int)
        zero = zeros_like(year)
        tup = (year, month, day, hour, minute, second, zero, zero, zero)
        if self.shape:
            return [strftime(format, item) for item in zip(*tup)]
        else:
            return strftime(format, tup)

    def _utc_tuple(self, offset=0.0):
        """Return UTC as (year, month, day, hour, minute, second.fraction).

        The `offset` is added to the UTC time before it is split into
        its components.  This is useful if the user is going to round
        the result before displaying it.  If the result is going to be
        displayed as seconds, for example, set `offset` to half a second
        and then throw away the fraction; if the result is going to be
        displayed as minutes, set `offset` to thirty seconds and then
        throw away the seconds; and so forth.

        """
        tai = self.tai + offset
        leap_dates, leap_offsets = self.cache.run(usno_leapseconds)
        leap_reverse_dates = leap_dates + leap_offsets / DAY_S
        i = searchsorted(leap_reverse_dates, tai, 'right')
        j = tai - leap_offsets[i] / DAY_S
        whole, fraction = divmod(j + 0.5, 1.0)
        whole = whole.astype(int)
        year, month, day = calendar_date(whole)
        hour, hfrac = divmod(fraction * 24.0, 1.0)
        minute, second = divmod(hfrac * 3600.0, 60.0)
        is_leap_second = j < leap_dates[i-1]
        second += is_leap_second
        return year, month, day, hour.astype(int), minute.astype(int), second

    def _utc_float(self):
        """Return UTC as a floating point Julian date."""
        tai = self.tai
        leap_dates, leap_offsets = self.cache.run(usno_leapseconds)
        leap_reverse_dates = leap_dates + leap_offsets / DAY_S
        i = searchsorted(leap_reverse_dates, tai, 'right')
        return tai - leap_offsets[i] / DAY_S

    def __getattr__(self, name):

        # Cache of several expensive functions of time.

        if name == 'P':
            self.P = P = compute_precession(self.tdb)
            return P

        if name == 'PT':
            self.PT = PT = rollaxis(self.P, 1)
            return PT

        if name == 'N':
            self.N = N = compute_nutation(self)
            return N

        if name == 'NT':
            self.NT = NT = rollaxis(self.N, 1)
            return NT

        if name == 'M':
            self.M = M = einsum('ij...,jk...,kl...->il...', self.N, self.P, B)
            return M

        if name == 'MT':
            self.MT = MT = rollaxis(self.M, 1)
            return MT

        # Conversion between timescales.

        if name == 'tai':
            self.tai = tai = self.tt - tt_minus_tai
            return tai

        if name == 'utc':
            utc = self._utc_tuple()
            utc = array(utc) if self.shape else utc
            self.utc = utc = utc
            return utc

        if name == 'tdb':
            tt = self.tt
            self.tdb = tdb = tt + tdb_minus_tt(tt) / DAY_S
            return tdb

        if name == 'ut1':
            self.ut1 = ut1 = self.tt - self.delta_t / DAY_S
            return ut1

        if name == 'gmst':
            self.gmst = gmst = sidereal_time(self)
            return gmst

        if name == 'gast':
            self.gast = gast = self.gmst + earth_tilt(self)[2] / 3600.0
            return gast

        raise AttributeError('no such attribute %r' % name)

    def __eq__(self, other_jd):
        return self.tt == other_jd.tt


def now():
    """Return the current date and time as a `JulianDate` object.

    For the return value to be correct, your operating system time and
    timezone settings must be set so that the Python Standard Library
    constructor ``datetime.datetime.utcnow()`` returns a correct UTC
    date and time.

    """
    return JulianDate(utc=datetime.utcnow().replace(tzinfo=utc))

def julian_day(year, month=1, day=1):
    """Given a proleptic Gregorian calendar date, return a Julian day int."""
    janfeb = month < 3
    return (day
            + 1461 * (year + 4800 - janfeb) // 4
            + 367 * (month - 2 + janfeb * 12) // 12
            - 3 * ((year + 4900 - janfeb) // 100) // 4
            - 32075)

def julian_date(year, month=1, day=1, hour=0, minute=0, second=0.0):
    """Given a proleptic Gregorian calendar date, return a Julian date float."""
    return julian_day(year, month, day) - 0.5 + (
        second + minute * 60.0 + hour * 3600.0) / DAY_S

def calendar_date(jd_integer):
    """Convert Julian Day `jd_integer` into a Gregorian (year, month, day)."""

    k = jd_integer + 68569
    n = 4 * k // 146097

    k = k - (146097 * n + 3) // 4
    m = 4000 * (k + 1) // 1461001
    k = k - 1461 * m // 4 + 31
    month = 80 * k // 2447
    day = k - 2447 * month // 80
    k = month // 11

    month = month + 2 - 12 * k
    year = 100 * (n - 49) + m + k

    return year, month, day

def tdb_minus_tt(jd_tdb):
    """Computes how far TDB is in advance of TT, given TDB.

    Given that the two time scales never diverge by more than 2ms, TT
    can also be given as the argument to perform the conversion in the
    other direction.

    """
    t = (jd_tdb - T0) / 36525.0

    # USNO Circular 179, eq. 2.6.
    return (0.001657 * sin ( 628.3076 * t + 6.2401)
          + 0.000022 * sin ( 575.3385 * t + 4.2970)
          + 0.000014 * sin (1256.6152 * t + 6.1969)
          + 0.000005 * sin ( 606.9777 * t + 4.0212)
          + 0.000005 * sin (  52.9691 * t + 0.4444)
          + 0.000002 * sin (  21.3299 * t + 5.5431)
          + 0.000010 * t * sin ( 628.3076 * t + 4.2490))

def usno_leapseconds(cache):
    """Download the USNO table of leap seconds as a ``(2, N+1)`` NumPy array.

    The array has two rows ``[leap_dates leap_offsets]``.  The first row
    is used to find where a given date ``jd`` falls in the table::

        index = np.searchsorted(leap_dates, jd, 'right')

    This can return a value from ``0`` to ``N``, allowing the
    corresponding UTC offset to be fetched with::

        offset = leap_offsets[index]

    The offset is the number of seconds that must be added to a UTC time
    to build the corresponding TAI time.

    """
    with cache.open_url('http://maia.usno.navy.mil/ser7/leapsec.dat') as f:
        lines = f.readlines()

    linefields = [line.split() for line in lines]
    dates = [float(fields[4]) for fields in linefields]
    offsets = [float(fields[6]) for fields in linefields]

    dates.insert(0, float('-inf'))
    dates.append(float('inf'))

    offsets.insert(0, offsets[0])
    offsets.insert(1, offsets[0])

    return array([dates, offsets])

def _utc_datetime_to_tai(leap_dates, leap_offsets, dt):
    try:
        utc_datetime = dt.astimezone(utc)
    except ValueError:
        raise ValueError(_naive_complaint)
    tup = utc_datetime.utctimetuple()
    year, month, day, hour, minute, second, wday, yday, dst = tup
    return _utc_to_tai(leap_dates, leap_offsets, year, month, day,
                       hour, minute, second + dt.microsecond / 1000000.00)

def _utc_date_to_tai(leap_dates, leap_offsets, d):
    return _utc_to_tai(leap_dates, leap_offsets, d.year, d.month, d.day)

def _utc_to_tai(leap_dates, leap_offsets, year, month=1, day=1,
                hour=0, minute=0, second=0.0):
    j = julian_day(year, month, day) - 0.5
    i = searchsorted(leap_dates, j, 'right')
    return j + (second + leap_offsets[i]
                + minute * 60.0
                + hour * 3600.0) / DAY_S


_naive_complaint = """cannot interpret a datetime that lacks a timezone

You must either specify that your datetime is in UTC:

    from skyfield.api import utc
    d = datetime(..., tzinfo=utc)  # to build a new datetime
    d = d.replace(tzinfo=utc)      # to fix an existing datetime

Or install the third-party `pytz` library and use any of its timezones:

    from pytz import timezone
    eastern = timezone('US/Eastern')
    d = eastern.localize(datetime(2014, 1, 16, 1, 32, 9))
"""
