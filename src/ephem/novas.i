%module novas

%{
#include "novas.h"
%}

#%include typemaps.i

%include "novascon.h"

/*
   IPT and LPT defined as int to support 64 bit systems.
*/


/*
   Structures.
*/

/*
   struct body: designates a celestial object.

   type              = type of body
                     = 0 ... major planet, Sun, or Moon
                     = 1 ... minor planet
   number            = body number
                       For 'type' = 0: Mercury = 1, ..., Pluto = 9,
                                       Sun = 10, Moon = 11
                       For 'type' = 1: minor planet number
   name              = name of the body (limited to 99 characters)
*/

/*
   struct cat_entry: the astrometric catalog data for a star; equator
                     and equinox and units will depend on the catalog.
                     While this structure can be used as a generic
                     container for catalog data, all high-level
                     NOVAS-C functions require J2000.0 catalog data
                     with FK5-type units (shown in square brackets
                     below).

   catalog[4]         = 3-character catalog designator.
   starname[51]       = name of star.
   starnumber         = integer identifier assigned to star.
   ra                 = mean right ascension [hours].
   dec                = mean declination [degrees].
   promora            = proper motion in RA [seconds of time per
                        century].
   promodec           = proper motion in declination [arcseconds per
                        century].
   parallax           = parallax [arcseconds].
   radialvelocity     = radial velocity [kilometers per second].
*/

%rename (Cat) cat_entry;
typedef struct {
      cat_entry();
      ~cat_entry();

      char catalog[4];
      char starname[51];
      long starnumber;
      double ra;
      double dec;
      double promora;
      double promodec;
      double parallax;
      double radialvelocity;
} cat_entry;

%rename (Object) object;
typedef struct {
	object();
	~object();
	short  type;
	short number;
	char name[100];
    cat_entry star;
} object;

/*
 struct on_surface: data for an observer's location on the surface of
 the Earth.  The atmospheric parameters are used
 only by the refraction function called from
 function 'equ2hor'. Additional parameters can be
 added to this structure if a more sophisticated
 refraction model is employed.
 
 latitude           = geodetic (ITRS) latitude; north positive (degrees)
 longitude          = geodetic (ITRS) longitude; east positive (degrees)
 height             = height of the observer (meters)
 temperature        = temperature (degrees Celsius)
 pressure           = atmospheric pressure (millibars)
 */

%rename (On_surface) on_surface;
typedef struct
{
    on_surface();
    ~on_surface();
    double latitude;
    double longitude;
    double height;
    double temperature;
    double pressure;
} on_surface;


/*
 struct in_space:   data for an observer's location on a near-Earth
 spacecraft
 
 sc_pos[3]          = geocentric position vector (x, y, z), components
 in km
 sc_vel[3]          = geocentric velocity vector (x_dot, y_dot,
 z_dot), components in km/s
 
 Both vectors with respect to true equator and
 equinox of date
 */

%rename (Inspace) in_space;
typedef struct
{
    in_space();
    ~in_space();
    double sc_pos[3];
    double sc_vel[3];
} in_space;

/*
 struct observer:   data specifying the location of the observer
 
 where              = integer code specifying location of observer
 = 0: observer at geocenter
 = 1: observer on surface of earth
 = 2: observer on near-earth spacecraft
 on_surface         = structure containing data for an observer's
 location on the surface of the Earth (where = 1)
 near_earth         = data for an observer's location on a near-Earth
 spacecraft (where = 2)
 */

%rename (Observer) Observer;

typedef struct
{
    observer();
    ~observer();
    short where;
    on_surface on_surf;
    in_space near_earth;
} observer;

/*
 struct sky_pos:    data specifying a celestial object's place on the
 sky; contains the output from function 'place'
 
 r_hat[3]           = unit vector toward object (dimensionless)
 ra                 = apparent, topocentric, or astrometric
 right ascension (hours)
 dec                = apparent, topocentric, or astrometric
 declination (degrees)
 dis                = true (geometric, Euclidian) distance to solar
 system body or 0.0 for star (AU)
 rv                 = radial velocity (km/s)
 */

%rename (Sky) sky_pos;
typedef struct
{
    sky_pos();
    ~sky_pos();
    double r_hat[3];
    double ra;
    double dec;
    double dis;
    double rv;
} sky_pos;

/*
 struct ra_of_cio:  right ascension of the Celestial Intermediate
 Origin (CIO) with respect to the GCRS
 
 jd_tdb             = TDB Julian date
 ra_cio             = right ascension of the CIO with respect
 to the GCRS (arcseconds)
 */

%rename (RA) ra_of_cio;
typedef struct
{
    ra_of_cio();
    ~ra_of_cio();
    double jd_tdb;
    double ra_cio;
} ra_of_cio;


/*
   Function prototypes
*/

%rename (AppStar) app_star;
short app_star (double tjd, cat_entry *star, short accuracy,
			double *OUTPUT, double *OUTPUT);

%rename (AppPlanet) app_planet;
short app_planet (double tjd, object *ss_object, short accuracy,
                         double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (TopoStar) topo_star;
short topo_star (double tjd, double deltat,
                        cat_entry *star, on_surface *location, short accuracy,
                        double *OUTPUT, double *OUTPUT);

%rename (TopoPlanet) topo_planet;
short topo_planet (double tjd, object *ss_object,
                          double deltat, on_surface *location, short accuracy,
                          double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (VirtualStar) virtual_star; 
short virtual_star (double tjd, cat_entry *star, short accuracy,
                           double *OUTPUT, double *OUTPUT);

%rename (VirtualPlanet) virtual_planet; 
short virtual_planet (double tjd, object *ss_object, short accuracy,
                             double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (LocalStar) local_star; 
short local_star (double tjd,  double deltat,
                         cat_entry *star, on_surface *location, short accuracy,
                         double *OUTPUT, double *OUTPUT);

%rename (LocalPlanet) local_planet; 
short local_planet (double tjd, object *ss_object,
                           double deltat, on_surface *location,
                           short accuracy,
                           double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (AstroStar) astro_star;
short astro_star (double tjd,  cat_entry *star,
                         short accuracy,
                         double *OUTPUT, double *OUTPUT);

%rename (AstroPlanet) astro_planet;
short astro_planet (double tjd, object *ss_object,
                           short accuracy,
                           double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (MeanStar) mean_star; 
short mean_star (double tjd,  double ra, double dec,
                        short accuracy,
                        double *OUTPUT, double *OUTPUT);

%rename (SiderealTime) sidereal_time;
void sidereal_time (double julianhi, double julianlo,
                    double delta_t, short gst_type,
                    short method, short accuracy,
                       double *OUTPUT);

%rename (Equ2Hor) equ2hor; 
void equ2hor (double tjd, double deltat, short accuracy, double x, double y,
                 on_surface *location, double ra, double dec,
                 short ref_option,
                 double *OUTPUT, double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (Cal2JD) julian_date; 
double julian_date (short year, short month, short day,
                       double hour);

%rename (JD2Cal) cal_date; 
void cal_date (double tjd,
                  short *OUTPUT, short *OUTPUT, short *OUTPUT,
                  double *OUTPUT);
%rename (ETilt) e_tilt;
void e_tilt (double jd_tdb, short accuracy,
                double *OUTPUT, double *OUTPUT, double *OUTPUT, double *OUTPUT,
                double *OUTPUT);
%rename (EphemOpen) ephem_open;
short int ephem_open (double *OUTPUT,double *OUTPUT,
                      short int *OUTPUT);
%rename (EphemClose) ephem_close;
short int ephem_close (void);

%rename (GetLeapSec) get_leap_sec;
short get_leap_sec(long mjd, double utc,double *OUTPUT);

%rename (GetIersData) get_iers_data;
short get_iers_data(long mjd, double *ut1_utc, double *x, double *y);

%rename (Ter2Cel) ter2celp;
short int ter2celp (double jd_ut_high, double jd_ut_low, double delta_t,
                        short int method, short int accuracy, short int option,
                        double xp, double yp, double vec11, double vec12, double vec13,
                        double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (Cel2Ter) cel2terp;
short int cel2terp (double jd_ut_high, double jd_ut_low, double delta_t,
                   short int method, short int accuracy, short int option,
                        double xp, double yp, double vec11, double vec12, double vec13,
                        double *OUTPUT, double *OUTPUT, double *OUTPUT);

%rename (Vector2Radec) v2radecp;
short int v2radecp (double pos1,double pos2, double pos3, double *OUTPUT, double *OUTPUT);

%rename (Equ2Ecl) equ2ecl;
short int equ2ecl (double jd_tt, short int coord_sys,
                   short int accuracy, double ra, double dec,
                   double *OUTPUT, double *OUTPUT);

%rename (GCRS2Equ) gcrs2equ;
short int gcrs2equ (double jd_tt, short int coord_sys,
                    short int accuracy, double rag, double decg,
                    double *OUTPUT, double *OUTPUT);
