"""Simple distance, velocity, and angle support for Skyfield.

"""
import numpy as np
from numpy import abs, array, copysign, isnan

from constants import AU_KM, AU_M, DAY_S, tau


def _auto_convert(value):
    """As a convenience, turn Python lists and tuples into NumPy arrays."""
    if isinstance(value, (tuple, list)):
        return array(value)
    else:
        return value

# Distance and velocity.

class UnpackingError(Exception):
    """You cannot iterate directly over a Skyfield measurement object."""

class Distance(object):
    """A distance, stored internally as AU and available in other units.

    You can initialize a ``Distance`` by providing a single float or a
    float array as either an ``AU=`` parameter or a ``km=`` parameter
    when building a ``Distance`` object.

    """
    def __init__(self, AU=None, km=None, m=None):
        if AU is not None:
            self.AU = _auto_convert(AU)
        elif km is not None:
            self.km = _auto_convert(km)
            self.AU = km / AU_KM
        elif m is not None:
            self.m = _auto_convert(m)
            self.AU = m / AU_M
        else:
            raise ValueError('to construct a Distance provide AU, km, or m')

    def __getattr__(self, name):
        if name == 'km':
            self.km = km = self.AU * AU_KM
            return km
        if name == 'm':
            self.m = m = self.AU * AU_M
            return m
        raise AttributeError('no attribute named %r' % (name,))

    def __str__(self):
        n = self.AU
        return ('{0} AU' if getattr(n, 'shape', 0) else '{0:.6} AU').format(n)

    def __repr__(self):
        return '<{0} {1}>'.format(type(self).__name__, self)

    def __iter__(self):
        raise UnpackingError(_iter_message % {
            'class': self.__class__.__name__, 'values': 'x, y, z',
            'attr1': 'AU', 'attr2': 'km'})

    def to(self, unit):
        """Return this distance in the given AstroPy units."""
        from astropy.units import AU
        return (self.AU * AU).to(unit)

class Velocity(object):
    """A velocity, stored internally as AU/day and available in other units.

    You can initialize a ``Velocity`` by providing a single float or a
    float array as either an ``AU_per_d=`` parameter.

    """
    def __init__(self, AU_per_d):
        self.AU_per_d = AU_per_d

    def __getattr__(self, name):
        if name == 'km_per_s':
            self.km_per_s = self.AU_per_d * AU_KM / DAY_S
            return self.km_per_s
        raise AttributeError('no attribute named %r' % (name,))

    def __str__(self):
        return '%s AU/day' % self.AU_per_d

    def __iter__(self):
        raise UnpackingError(_iter_message % {
            'class': self.__class__.__name__, 'values': 'xdot, ydot, zdot',
            'attr1': 'AU_per_d', 'attr2': 'km_per_s'})

    def to(self, unit):
        """Return this velocity in the given AstroPy units."""
        from astropy.units import AU, d
        return (self.AU_per_d * AU / d).to(unit)

_iter_message = """\
cannot directly unpack a %(class)s into several values

To unpack a %(class)s into three components, you need to ask for its
value in specific units through an attribute or method:

    %(values)s = velocity.%(attr1)s
    %(values)s = velocity.%(attr2)s
    %(values)s = velocity.to(astropy_unit)
"""

# Angle units.

_to_degrees = 360.0 / tau
_from_degrees = tau / 360.0

_to_hours = 24.0 / tau
_from_hours = tau / 24.0

_instantiation_instructions = """to instantiate an Angle, try one of:

Angle(angle=another_angle)
Angle(radians=value)
Angle(degrees=value)
Angle(hours=value)

where `value` can be either a Python float or a NumPy array of floats"""

class Angle(object):

    def __init__(self, angle=None, radians=None, degrees=None, hours=None,
                 preference=None, signed=False):

        if angle is not None:
            if not isinstance(angle, Angle):
                raise ValueError(_instantiation_instructions)
            self.radians = angle.radians
        elif radians is not None:
            self.radians = radians
        elif degrees is not None:
            self._degrees = degrees = _unsexagesimalize(degrees)
            self.radians = degrees * _from_degrees
        elif hours is not None:
            self._hours = hours = _unsexagesimalize(hours)
            self.radians = hours * _from_hours
            preference = 'hours'

        self.preference = preference or 'degrees'
        self.signed = signed

    def __getattr__(self, name):
        if name == '_hours':
            self._hours = _hours = self.radians * _to_hours
            return _hours
        if name == '_degrees':
            self._degrees = _degrees = self.radians * _to_degrees
            return _degrees
        if name == 'hours':
            if self.preference != 'hours':
                raise WrongUnitError('hours')
            self.hours = hours = self._hours
            return hours
        if name == 'degrees':
            if self.preference != 'degrees':
                raise WrongUnitError('degrees')
            self.degrees = degrees = self._degrees
            return degrees
        raise AttributeError('no attribute named %r' % (name,))

    def __str__(self):
        return self.dstr() if self.preference == 'degrees' else self.hstr()

    def __repr__(self):
        return '<{0} {1}>'.format(type(self).__name__, self)

    def hms(self, warn=True):
        if warn and self.preference != 'hours':
            raise WrongUnitError('hms')
        sign, units, minutes, seconds = _sexagesimalize_to_float(self._hours)
        return sign * units, sign * minutes, sign * seconds

    def signed_hms(self, warn=True):
        if warn and self.preference != 'hours':
            raise WrongUnitError('signed_hms')
        return _sexagesimalize_to_float(self._hours)

    def hstr(self, places=2, warn=True):
        if warn and self.preference != 'hours':
            raise WrongUnitError('hstr')
        hours = self._hours
        if getattr(hours, 'shape', None):
            return [_hstr(h, places) for h in hours]
        return _hstr(hours, places)

    def dms(self, warn=True):
        if warn and self.preference != 'degrees':
            raise WrongUnitError('dms')
        sign, units, minutes, seconds = _sexagesimalize_to_float(self._degrees)
        return sign * units, sign * minutes, sign * seconds

    def signed_dms(self, warn=True):
        if warn and self.preference != 'degrees':
            raise WrongUnitError('signed_dms')
        return _sexagesimalize_to_float(self._degrees)

    def dstr(self, places=1, warn=True):
        if warn and self.preference != 'degrees':
            raise WrongUnitError('dstr')
        degrees = self._degrees
        signed = self.signed
        if getattr(degrees, 'shape', None):
            return [_dstr(d, places, signed) for d in degrees]
        return _dstr(degrees, places, signed)

class WrongUnitError(ValueError):

    def __init__(self, name):
        unit = 'hours' if (name.startswith('h') or '_h' in name) else 'degrees'
        usual = 'hours' if (unit == 'degrees') else 'degrees'
        message = ('this angle is usually expressed in {0}, not {1};'
                   ' if you want to use {1} anyway,'.format(usual, unit))
        if name == unit:
            message += ' then please use the attribute _{0}'.format(unit)
        else:
            message += ' then call {0}() with warn=False'.format(name)
        self.args = (message,)

def _sexagesimalize_to_float(value):
    """Decompose `value` into units, minutes, and seconds.

    Note that this routine is not appropriate for displaying a value,
    because rounding to the smallest digit of display is necessary
    before showing a value to the user.  Use `_sexagesimalize_to_int()`
    for data being displayed to the user.

    This routine simply decomposes the floating point `value` into a
    sign (+1.0 or -1.0), units, minutes, and seconds, returning the
    result in a four-element tuple.

    >>> _sexagesimalize_to_float(12.05125)
    (1.0, 12.0, 3.0, 4.5)
    >>> _sexagesimalize_to_float(-12.05125)
    (-1.0, 12.0, 3.0, 4.5)

    """
    sign = np.sign(value)
    n = abs(value)
    minutes, seconds = divmod(n * 3600.0, 60.0)
    units, minutes = divmod(minutes, 60.0)
    return sign, units, minutes, seconds

def _sexagesimalize_to_int(value, places=0):
    """Decompose `value` into units, minutes, seconds, and second fractions.

    This routine prepares a value for sexagesimal display, with its
    seconds fraction expressed as an integer with `places` digits.  The
    result is a tuple of five integers:

    ``(sign [either +1 or -1], units, minutes, seconds, second_fractions)``

    The integers are properly rounded per astronomical convention so
    that, for example, given ``places=3`` the result tuple ``(1, 11, 22,
    33, 444)`` means that the input was closer to 11u 22' 33.444" than
    to either 33.443" or 33.445" in its value.

    """
    sign = int(np.sign(value))
    value = abs(value)
    power = 10 ** places
    n = int(7200 * power * value + 1) // 2
    n, fraction = divmod(n, power)
    n, seconds = divmod(n, 60)
    n, minutes = divmod(n, 60)
    return sign, n, minutes, seconds, fraction

def _hstr(hours, places=2):
    """Convert floating point `hours` into a sexagesimal string.

    >>> _hstr(12.125)
    '12h 07m 30.00s'
    >>> _hstr(12.125, places=4)
    '12h 07m 30.0000s'
    >>> _hstr(float('nan'))
    'nan'

    """
    if isnan(hours):
        return 'nan'
    sgn, h, m, s, etc = _sexagesimalize_to_int(hours, places)
    sign = '-' if sgn < 0.0 else ''
    return '%s%02dh %02dm %02d.%0*ds' % (sign, h, m, s, places, etc)

def _dstr(degrees, places=1, signed=False):
    r"""Convert floating point `degrees` into a sexagesimal string.

    >>> _dstr(12.125)
    '12deg 07\' 30.0"'
    >>> _dstr(12.125, places=3)
    '12deg 07\' 30.000"'
    >>> _dstr(12.125, signed=True)
    '+12deg 07\' 30.0"'
    >>> _dstr(float('nan'))
    'nan'

    """
    if isnan(degrees):
        return 'nan'
    sgn, d, m, s, etc = _sexagesimalize_to_int(degrees, places)
    sign = '-' if sgn < 0.0 else '+' if signed else ''
    return '%s%02ddeg %02d\' %02d.%0*d"' % (sign, d, m, s, places, etc)


def _unsexagesimalize(value):
    """Return `value` after interpreting a (units, minutes, seconds) tuple.

    When `value` is not a tuple, it is simply returned.

    >>> _unsexagesimalize(3.25)
    3.25

    An input tuple is interpreted as units, minutes, and seconds.  Note
    that only the sign of `units` is significant!  So all of the
    following tuples convert into exactly the same value:

    >>> '%f' % _unsexagesimalize((-1, 2, 3))
    '-1.034167'
    >>> '%f' % _unsexagesimalize((-1, -2, 3))
    '-1.034167'
    >>> '%f' % _unsexagesimalize((-1, -2, -3))
    '-1.034167'

    """
    if isinstance(value, tuple):
        for i, component in enumerate(value):
            if i:
                value = value + copysign(component, value) * 60.0 ** -i
            else:
                value = component
    return value

def _interpret_angle(name, angle_object, angle_float, unit='degrees'):
    """Return an angle in radians from one of two arguments.

    It is common for Skyfield routines to accept both an argument like
    `alt` that takes an Angle object as well as an `alt_degrees` that
    can be given a bare float or a sexagesimal tuple.  A pair of such
    arguments can be passed to this routine for interpretation.

    """
    if angle_object is not None:
        if isinstance(angle_object, Angle):
            return angle_object.radians
    elif angle_float is not None:
        return _unsexagesimalize(angle_float) * _from_degrees
    raise ValueError('you must either provide the {0}= parameter with'
                     ' an Angle argument or supply the {0}_{1}= parameter'
                     ' with a numeric argument'.format(name, unit))

def _interpret_ltude(value, name, psuffix, nsuffix):
    """Interpret a string as a latitude or longitude angle.

    `value` - The string to interpret.
    `name` - 'latitude' or 'longitude', for use in exception messages.
    `positive` - The string that indicates a positive angle ('N' or 'E').
    `negative` - The string that indicates a negative angle ('S' or 'W').

    """
    value = value.strip().upper()

    if value.endswith(psuffix):
        sign = +1.0
    elif value.endswith(nsuffix):
        sign = -1.0
    else:
        raise ValueError('your {0} string {1!r} does not end with either {2!r}'
                         ' or {3!r}'.format(name, value, psuffix, nsuffix))

    try:
        value = float(value[:-1])
    except ValueError:
        raise ValueError('your {0} string {1!r} cannot be parsed as a floating'
                         ' point number'.format(name, value))

    return Angle(degrees=sign * value)
