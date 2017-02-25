/*
 # The position calculation server based on ZeroMQ platform
 # CSRHOS - Chinese Solar Radio HelioGraph Operation System
 #
 # Created: Since 2013-1-1
 #     CSRHOS Team, Feng Wang
 #
 # This file is part of CSRH project
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <czmq.h>
#include "eph_manager.h" /* remove this line for use with solsys version 2 */
#include "novas.h"


struct Target
{
    int Mjd;  // Mjd
    int targetNo;  // Id of Target, 0: Sun
    double xpos, ypos, ut1_utc;
    double appra[49],appdec[49],gast[49],t[49];
}target[100];

int dataNumber=0;

const double PI=3.14159265358979;

int year,month,day;
double tt;

const short int accuracy = 0;
short int error = 0;
short int de_num = 0;
double ut1_utc, xx,yy;

const double height=1365.0;
const double latitude=42.21183333333333;
const double longitude=115.2505;
const double temperature = 10.0;
const double pressure = 1010.0;

double jd_beg,jd_end,jd_utc, jd_tt, jd_ut1, jd_tdb, delta_t,dis, ra,dec,rat,dect,dist,gast;

on_surface geo_loc;
observer   obs_loc;
cat_entry  dummy_star;

long MJD;
double JD;
double leapsecondt;

object mercury,venus,earth,mars,jupiter,saturn,uranus,neptune,pluto,sun,moon;
double Parabola(double *r,double *t,int len,double x);



int calcSun(int year, int month, int day, double hour, struct Target *t)
{

    int i;
    /*
     Make structures of type 'on_surface' and 'observer-on-surface' containing
     the observer's position and weather (latitude, longitude, height,
     temperature, and atmospheric pressure).
     */

    make_on_surface (latitude,longitude,height,temperature,pressure, &geo_loc);

    make_observer_on_surface (latitude,longitude,height,temperature,pressure,
                              &obs_loc);

    /*
     Make structures of type 'object' for the Mercury....Sun,Moon.
     */
    make_cat_entry ("DUMMY","xxx",0,0.0,0.0,0.0,0.0,0.0,0.0,&dummy_star);

	if ((error = make_object (0,10,"sun",&dummy_star, &sun)) != 0)
	{
		printf ("Error %d from make_object (sun)\n", error);
		return (error);
	}

    //printf("Midnight_MJD=%ld,MJD=%ld\n",Midnight_MJD,MJD);
    get_leap_sec(MJD,0.,&leapsecondt);
    //printf("Leap: %lf\n",leapsecondt);
    if (!get_iers_data(MJD, &ut1_utc, &xx, &yy))
    {
        printf("Error retrieve IERS Data\n");
        exit(1);
    };
    t->Mjd  = MJD;
    t->xpos = xx;
    t->ypos = yy;
    t->ut1_utc = ut1_utc;
    t->targetNo =0 ;
    //printf("IERS: %ld %lf %lf %lf\n",MJD, ut1_utc, xx,yy);

    hour = 0.;
    for(i=0;i<49;i++)
	{
        jd_utc = julian_date (year,month,day,hour);
        jd_tt = jd_utc+((double) leapsecondt + 32.184) / 86400.0;
        jd_ut1 = jd_utc + ut1_utc / 86400.0;
        delta_t = 32.184 + leapsecondt - ut1_utc;

        /*
         Apparent and topocentric place of the SUN.
         */

        if ((error = app_planet (jd_tt,&sun,accuracy, &ra,&dec,&dis)) != 0)
        {
            printf ("Error %d from app_planet.", error);
            return (error);
        }

        /* if ((error = topo_planet (jd_tt,&sun,delta_t,&geo_loc,accuracy,
         &rat,&dect,&dist)) != 0)
         {
         printf ("Error %d from topo_planet.", error);
         return (error);
         }
         */

        /*
         Greenwich and local apparent sidereal time and Earth Rotation Angle.
         */

        if ((error = sidereal_time (jd_ut1,0.0,delta_t,0,1,accuracy, &gast)) != 0)
        {
            printf ("Error %d from sidereal_time.", error);
            return (error);
        }
        //RA_SUN[i]=15*ra;
        t->appra[i]=ra;
        t->appdec[i]=dec;
        t->gast[i]=gast;
        t->t[i]=hour;
        hour=hour+0.5;
        //printf("RA_SUN[%d]=%.10f   DEC_SUN[%d] =%.10f   GMST_SUN[%d] =%.10f   t[%d]=%.10f\n",i,RA_SUN[i],i,DEC_SUN[i],i,GMST_SUN[i],i,t[i]);
	}
    //printf("Insert OK\n");
}

double Parabola(double *r,double *t,int len,double x)
{
    int i,k;
    double a,b,c,m;
    double* aa=(double*)malloc((len)*sizeof(double));
    double* bb=(double*)malloc((len)*sizeof(double));
    double* cc=(double*)malloc((len)*sizeof(double));
    double* mm=(double*)malloc((len)*sizeof(double));
    for (i=0;i<len;i++)
    {
        *(aa+i)=0.0;
        *(bb+i)=0.0;
        *(cc+i)=0.0;
        *(mm+i)=0.0;
    }
    double y,x1,y1,x2,y2,x3,y3;
    for(i=0;i<len-2;i++)
    {
        y1=r[i];
        y2=r[i+1];
        y3=r[i+2];
        x1=t[i];
        x2=t[i+1];
        x3=t[i+2];
        m=x1*x1*x2+x2*x2*x3+x1*x3*x3-x3*x3*x2-x2*x2*x1-x1*x1*x3;
        a=(y1*x2+y2*x3+y3*x1-y3*x2-y2*x1-y1*x3)/m;
        b=(x1*x1*y2+x2*x2*y3+x3*x3*y1-x3*x3*y2-x2*x2*y1-x1*x1*y3)/m;
        c=(x1*x1*x2*y3+x2*x2*x3*y1+x3*x3*x1*y2-x3*x3*x2*y1-x2*x2*x1*y3-x1*x1*x3*y2)/m;
        aa[i]=a;
        bb[i]=b;
        cc[i]=c;
        mm[i]=m;
        //printf("a=%.10f b=%.10f c=%.10f\n",aa[i],bb[i],cc[i]);
    }
    k=(x-0.25)/0.5;
    if(k<0) k=0;
    else  k=ceil(k);

    y=aa[k]*x*x+bb[k]*x+cc[k];
    //printf("a=%.10f b=%.10f c=%.10f\n",aa[k],bb[k],cc[k]);
    free(aa);
    free(bb);
    free(cc);
    free(mm);
    return y;
}


int main()
{
    int i;

    double GAST,RA,DEC;
    char result[250];
    zctx_t *context= zctx_new ();
    void *server;
    char *ymdt;
    int obsTarget;

    /*
     Open the JPL binary ephemeris file, here named "JPLEPH".
     Remove this block for use with solsys version 2.
     */
    if ((error = ephem_open (&jd_beg,&jd_end,&de_num)) != 0)
    {
        if (error == 1)
            printf ("JPL ephemeris file not found.\n");

        printf ("Error reading JPL ephemeris file header.\n");
        return (error);
    }

    printf ("Chinese Spectral RadioHeliograph Ephemerides \n");
    printf ("Foundamental ephemerides: DE405\n");
    printf( "Server is starting...\n");

    server = zsocket_new(context, ZMQ_REP);
    assert(server);
    zsocket_bind(server, "tcp://*:5555");
    printf ("Server started...");

    //load array
    while (!zctx_interrupted) {
        char *request = zstr_recv (server);
	printf("R:%s\n",request);
    	ymdt=strtok(request,"-");
        year=atof(ymdt);

        ymdt = strtok(NULL,"-");
        month=atof(ymdt);

    	ymdt =strtok(NULL,"-");
        day=atof(ymdt);

    	ymdt=strtok(NULL,"-");
        tt=atof(ymdt);

        ymdt=strtok(NULL,"-");
        obsTarget=atoi(ymdt);
	
	if ( (year<=2010 || year>2050) || (month<1 || month >12) || (day<1 || day>31) || (tt<0. || tt>24.) || (obsTarget <0 || obsTarget > 10))
	{
	   sprintf(result,"ERROR");
	   zstr_send(server, result);
	   continue;
	}
        jd_utc = julian_date (year,month,day,tt);

        MJD = (long)(julian_date (year,month,day,0.)-2400000.5);
        JD = julian_date (year,month,day,tt);

        // Check the data
        for (i=0;i<dataNumber;i++)
        {
            if (target[i].targetNo == obsTarget && target[i].Mjd == MJD)
                break;
        }
        // Data exist ?
        if (i==dataNumber)
        {
            // If not, calculate the coordinate of the target in the certain day
            calcSun(year, month, day, tt, &target[dataNumber++]);
		printf("Insert a new record\n");
        }
	printf("Q: %d %d %d: %lf\n",year,month,day,tt);
        //query from memery array
        RA=Parabola(target[i].appra,target[i].t,49,tt);
        DEC=Parabola(target[i].appdec,target[i].t,49,tt);
        GAST=Parabola(target[i].gast,target[i].t,49,tt);
        //printf("MJD=%ld;JD=%.15f;RA=%.15f;DEC=%.15f;GAST=%.15f;X=%.15f;Y=%.15f;UT1UTC=%.15f\n",MJD,JD,RA,DEC,GAST,target[i].xpos,target[i].ypos,target[i].ut1_utc);
        sprintf(result,"MJD=%ld;JD=%.15f;RA=%.15f;DEC=%.15f;GAST=%.15f;X=%.15f;Y=%.15f;UT1UTC=%.15f",MJD,JD,RA,DEC,GAST,target[i].xpos,target[i].ypos,target[i].ut1_utc);
        printf("%s\n",result);
        zstr_send(server,result);

    }

    //printf("year=%d,month=%d,day=%d,tt=%f\n",year,month,day,tt);

    zctx_destroy (&context);

    ephem_close();  /* remove this line for use with solsys version 2 */

    return 0;
}


