#include <stdio.h>
#include <stdlib.h>
#include "novas.h"

char * h2s(char buf[], double hour);
char * d2s(char buf[], double deg);

int main (void)
{

   short int error = 0;
   short int i, j;
   char buf1[256];
   char buf2[256];

/*
   'deltat' is te difference in time scales, TT - UT1.
   2001.5 : deltat = 64.198  (source IERS)

*/

/*
   in a private communication Mr Kaplan stated that MICA
   uses an estimation of deltat equal to 67 s for 2001.5
   So for comparision we set:
*/

   double deltat = 67.0;
/*
   tjd in TT
   20010621
   eclipse of the SUN in Africa
*/

   double tjd = 2452081.5  + deltat / 86400.0;
   double ra, dec, dis;


/*
   Te observer's terrestrial coordinates (latitude, longitude, height).
   Lusaka
*/
   const double latitude = -15.3333333333333333;
   const double longitude = 28.3333333333333333;
   const double height = 0.0;
   const double temperature = 10.0;
   const double pressure = 1010.0;
   //site_info geo_loc = {-15.3333333333333333, 28.3333333333333333, 0.0, 20.0, 1010.0};
   on_surface geo_loc;
   observer obs_loc;

   cat_entry star, dummy_star;

/*
   Structures containing te body designation for Earth, Moon & Sun.
*/

   object earth;
   object moon;
   object sun;

   sky_pos t_place;
/*
   Set up the structure containing the body designation for Earth, etc.
*/
   if ((error = make_object (0,11,"Moon",&dummy_star, &moon)) != 0)
   {
      printf ("Error %d from make_object (Moon)\n", error);
      return (error);
   }

   if ((error = make_object (0,10,"Sun",&dummy_star, &sun)) != 0)
   {
      printf ("Error %d from make_object (Sun)\n", error);
      return (error);
   }

   if ((error = make_object (0,3,"Earth",&dummy_star, &earth)) != 0)
   {
      printf ("Error %d from make_object (Earth)\n", error);
      return (error);
   }

/*
   print results for 24 hours

*/

   make_on_surface (latitude,longitude,height,temperature,pressure, &geo_loc);
   make_observer_on_surface (latitude,longitude,height,temperature,pressure,
      &obs_loc);

   printf("Moon and Sun positions at Lusaka 20010621\n\n");

   for(i=0;i<25;i++)
   {
       printf("%2i:00:00.0 ",i);

       topo_planet(tjd, &moon, deltat,&geo_loc, 0., &ra, &dec, &dis);
       printf ("%12s   %12s   %12.3f   ", h2s(buf1, ra), d2s(buf2, dec), dis );
       topo_planet(tjd, &sun, deltat,& geo_loc, 0., &ra, &dec, &dis);
       printf ("%12s   %12s   %12.9f  \n", h2s(buf1, ra), d2s(buf2, dec), dis);
       tjd = tjd+1.0/24.0;
   }
   exit (0);
}

char * h2s(char buf[], double hour)
{
	int h, min;
	double sec;
        char *p;

	p = buf;
	h = (int) floor(hour);

	min = (int) floor(hour*60 - 60.0*h);
	sec = hour*3600.0  - h*3600.0 - 60.0*min;
	sprintf(buf, "%2i %2i %6.3f", h, min, sec);
	return p;
}

char * d2s(char buf[], double deg)
{
	int d, min;
	double sec;
        char *p;

	p = buf;
	d = (int) floor(deg);

	min = (int) floor(deg*60 - 60.0*d);
	sec = deg*3600.0  - d*3600.0 - 60.0*min;
	sprintf(buf, "%2i %2i %6.2f", d, min, sec);
	return p;
}
