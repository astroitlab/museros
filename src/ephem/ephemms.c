#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "eph_manager.h" /* remove this line for use with solsys version 2 */
#include "novas.h"
const double PI=3.14159265358979;
const short int year = 2013;
const short int month = 12;
const short int day=22;
const short int accuracy = 0;
short int error = 0;
short int de_num = 0;
double ut1_utc, xx,yy;
const double height=1365.0;
const double latitude=42.21183333333333;
const double longitude=115.2505;
const double temperature = 10.0;
const double pressure = 1010.0;
double jd_beg,jd_end,jd_utc, jd_tt, jd_ut1, jd_tdb, delta_t,dis, ra,dec,rat,dect,dist,gmst;
on_surface geo_loc;
observer   obs_loc;
cat_entry  dummy_star;
object mercury,venus,earth,mars,jupiter,saturn,uranus,neptune,pluto,sun,moon;
double Parabola(double *r,double *t,int len,double x);
int main()
{
    clock_t t1=clock();
    printf("Usage:Reading the data of RA DEC GMST from JPL DE405.........\n");
    double BSL [44][44][3];
    double DXYZ [44][44][3];
    int i,j,k;
    double RA_SUN[25];
    double DEC_SUN[25];
    double GMST_SUN[25];
    double t[25];
    int YEAR=13;
    int MONTH=12;
    int DAY=22;
    int HOUR_S=8;
    int HOUR_E=20;
    double DELAY_NS[44]={-56,  0, 48,921, 13, 59, -3, 460, 49, 69,-675,-157,363,-65,
                 30, 42, 51, 121, -2,  73,  35, 26,  74, 35, -3,   47, -71,
                 75,343, 56, 32,313,  678,  12,-30, 48,-18, 20,   10, -1666,
                  0,  0,  0,  0};
    double MIN1;
    int  TIME_H,TIME_M,TIME_S,MS,ANT;
    char YEAR1,YEAR2,MONTH1,MONTH2,DAY1,DAY2;
    char HOUR1,HOUR2,MINU1,MINU2,SEC1,SEC2;
    unsigned short S_5A01=0x5A01;
    unsigned short S_5AEF=0x5AEF;
    unsigned short S_5A5A=0x5A5A;
    unsigned short D_NS0,D_PS0,D_NS1,D_PS1,D_NS2,D_PS2,D_NS3,D_PS3;
    double HOUR_REAL;
    double GMST,RA,DEC,H;
    double LOC_MST;
    double W[44][1000];
    double MIN2;

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

/*
   Open the JPL binary ephemeris file, here named "JPLEPH".
   Remove this block for use with solsys version 2.
*/

   if ((error = ephem_open (&jd_beg,&jd_end,&de_num)) != 0)
   {
      if (error == 1)
         printf ("JPL ephemeris file not found.\n");
       else
         printf ("Error reading JPL ephemeris file header.\n");
      return (error);
   }
    else
   {
    printf ("JPL ephemeris DE%d open. Start JD = %10.2f  End JD = %10.2f\n",
         de_num, jd_beg, jd_end);
      printf ("\n");
   }

/*
   Establish time arguments.
*/
   double hour=8.0;
   jd_utc = julian_date (year,month,day,hour);
   long MJD = (long)(julian_date (year,month,day,0.)-2400000.5);
   printf("MJD:%ld\n",MJD);
   double leapsecondt;
   get_leap_sec(MJD,0.,&leapsecondt);
   printf("Leap: %lf\n",leapsecondt);
   if (get_iers_data(MJD, &ut1_utc, &xx, &yy));
       printf("IERS: %ld %lf %lf %lf\n",MJD, ut1_utc, xx,yy);
   for(i=0;i<25;i++)
{
   jd_utc = julian_date (year,month,day,hour);
   jd_tt = jd_utc+((double)leapsecondt + 32.184) / 86400.0;
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

   if ((error = sidereal_time (jd_ut1,0.0,delta_t,0,1,accuracy, &gmst)) != 0)
   {
      printf ("Error %d from sidereal_time.", error);
      return (error);
   }
   RA_SUN[i]=15*ra;
   DEC_SUN[i]=dec;
   GMST_SUN[i]=gmst;
   t[i]=hour;
   hour=hour+0.5;
  // printf("RA_SUN[%d]=%.10f   DEC_SUN[%d] =%.10f   GMST_SUN[%d] =%.10f   t[%d]=%.10f\n",i,RA_SUN[i],i,DEC_SUN[i],i,GMST_SUN[i],i,t[i]);
}
    printf("Usage:Reading  Antenna Positions  from  your path.........\n");
    FILE *fp1;
    double ANT_POS[44][3];

    FILE *fp2;
    char str[50];

    fp1=fopen("Ant_pos.txt","r");
    for(i=0;i<40;i++)
    {
        fscanf(fp1,"%lf %lf %lf",&(ANT_POS[i][1]),&(ANT_POS[i][0]),&(ANT_POS[i][2]));
    }
    fclose(fp1);
    for(i=0;i<40;i++)
    {
        for(j=0;j<40;j++)
        {

             for(k=0;k<3;k++)
             {
                BSL[i][j][k]=ANT_POS[i][k]-ANT_POS[j][k];
             }

		{
                   DXYZ[i][j][0]=-sin(PI*latitude/180.0)*BSL[i][j][0]+cos(PI*latitude/180.0)*BSL[i][j][2];

                   DXYZ[i][j][1]=BSL[i][j][1];

                   DXYZ[i][j][2]=cos(PI*latitude/180.0)*BSL[i][j][0]+sin(PI*latitude/180.0)*BSL[i][j][2];
		}
        }
    }
    YEAR1=(int)(YEAR/10)+48;
    YEAR2=((YEAR%10)+48);
    MONTH1=(int)(MONTH/10)+48;
    MONTH2=(MONTH%10)+48;
    DAY1=(int)(DAY/10)+48;
    DAY2=DAY%10+48;
    MIN1=DELAY_NS[0];
    for(i=1;i<44;i++)
    {
        if(DELAY_NS[i]<MIN1){
            MIN1=DELAY_NS[i];
        }
    }
    for (i=0;i<44;i++)
        DELAY_NS[i]=DELAY_NS[i]-MIN1;

    sprintf(str,"SUN_20%d%d%d_%d%d.dat",YEAR,MONTH,DAY,HOUR_S,HOUR_E);

    fp2=fopen(str,"w");
    printf("Usage:Writing the data of date (year,month,day) to the file.........\n");
    printf("Usage:Computing the apparent position of  sun in 1ms through parabola interpolation.........\n");
    printf("Usage:Waiting.........\n");
    for(TIME_H=HOUR_S;TIME_H<HOUR_E;TIME_H++)
    {

            HOUR1=(int)(TIME_H/10)+48;
            HOUR2=(TIME_H%10)+48;


        for(TIME_M=0;TIME_M<60;TIME_M++)
        {
	        MINU1=(int)(TIME_M/10)+48;
                MINU2=(TIME_M%10)+48;


            for(TIME_S=0;TIME_S<60;TIME_S++)
            {
                 SEC1=(int)(TIME_S/10)+48;
                 SEC2=(TIME_S%10)+48;

                 fwrite(&S_5A01,2,1,fp2);
                 fwrite(&S_5AEF,2,1,fp2);
                 fwrite(&YEAR1,1,1,fp2);
                 fwrite(&YEAR2,1,1,fp2);
                 fwrite(&MONTH1,1,1,fp2);
                 fwrite(&MONTH2,1,1,fp2);
                 fwrite(&DAY1,1,1,fp2);
                 fwrite(&DAY2,1,1,fp2);
                 fwrite(&HOUR1,1,1,fp2);
                 fwrite(&HOUR2,1,1,fp2);
                 fwrite(&MINU1,1,1,fp2);
                 fwrite(&MINU2,1,1,fp2);
                 fwrite(&SEC1,1,1,fp2);
                 fwrite(&SEC2,1,1,fp2);
                 for(MS=0;MS<1000;MS++)
                    {
                         HOUR_REAL=TIME_H+(TIME_M/60.0)+(TIME_S/3600.0)+(MS/3600.0/1000.0);//-(67.184/3600.0);
                         RA=Parabola(RA_SUN,t,25,HOUR_REAL);
                         DEC=Parabola(DEC_SUN,t,25,HOUR_REAL);
                         GMST=Parabola(GMST_SUN,t,25,HOUR_REAL);
                         //printf("Interpolate from Bessel:in the hour=24:\n");
                         //printf("RA_SUN =%.10f\n", RA);
                         //getchar();
                         LOC_MST=(GMST+HOUR_REAL+(longitude/15.0)+(67.184/3600.0));
                         LOC_MST=LOC_MST-(int)LOC_MST+((int)LOC_MST%24);
                         H=((double)LOC_MST-RA)*((double)360.0/(24.0-(double)3.0/60.0-(double)56.0/3600.0));
                         DEC=DEC/180.0*PI;
                         H=H/180.0*PI;

                        for(ANT=0;ANT<11;ANT++)
                       {
                        W[0+ANT*4][MS]=DXYZ[0][0+ANT*4][0]*cos(DEC)*cos(H)+DXYZ[0][0+ANT*4][1]*(-cos(DEC)*sin(H))+sin(DEC)*DXYZ[0][0+ANT*4][2];
                        W[1+ANT*4][MS]=DXYZ[0][1+ANT*4][0]*cos(DEC)*cos(H)+DXYZ[0][1+ANT*4][1]*(-cos(DEC)*sin(H))+sin(DEC)*DXYZ[0][1+ANT*4][2];
                        W[2+ANT*4][MS]=DXYZ[0][2+ANT*4][0]*cos(DEC)*cos(H)+DXYZ[0][2+ANT*4][1]*(-cos(DEC)*sin(H))+sin(DEC)*DXYZ[0][2+ANT*4][2];
                        W[3+ANT*4][MS]=DXYZ[0][3+ANT*4][0]*cos(DEC)*cos(H)+DXYZ[0][3+ANT*4][1]*(-cos(DEC)*sin(H))+sin(DEC)*DXYZ[0][3+ANT*4][2];
                       }
                       MIN2=W[0][MS];
                       for (ANT=1; ANT<44; ANT++)
                           if(W[ANT][MS]<=MIN2){MIN2=W[ANT][MS];}
                       for (ANT=0; ANT<44; ANT++){
                        W[ANT][MS]=W[ANT][MS]-MIN2;
                        W[ANT][MS]=W[ANT][MS]/0.299792;
                       }
                      //printf("tt=%.10f\n",HOUR_REAL);

                    }

                for(ANT=0;ANT<11;ANT++)
                {
                    fwrite(&S_5A5A,2,1,fp2);

                    for(MS=0;MS<1000;MS++)
                    {
                        W[0+ANT*4][MS]=W[0+ANT*4][MS]+DELAY_NS[0+ANT*4];
                        W[1+ANT*4][MS]=W[1+ANT*4][MS]+DELAY_NS[1+ANT*4];
                        W[2+ANT*4][MS]=W[2+ANT*4][MS]+DELAY_NS[2+ANT*4];
                        W[3+ANT*4][MS]=W[3+ANT*4][MS]+DELAY_NS[3+ANT*4];
                        D_NS0=(short) ((W[0+ANT*4][MS]));
                        D_PS0=(short) (((W[0+ANT*4][MS])-(short)(W[0+ANT*4][MS]))*10000);
                        D_NS1=(short) (W[1+ANT*4][MS]);
                        D_PS1=(short) (((W[1+ANT*4][MS])-(short)(W[1+ANT*4][MS]))*10000);
                        D_NS2=(short) (W[2+ANT*4][MS]);
                        D_PS2=(short) (((W[2+ANT*4][MS])-(short)(W[2+ANT*4][MS]))*10000);
                        D_NS3=(short) (W[3+ANT*4][MS]);
                        D_PS3=(short) (((W[3+ANT*4][MS])-(short)(W[3+ANT*4][MS]))*10000);
                        fwrite(&D_NS0,2,1,fp2);
                        fwrite(&D_PS0,2,1,fp2);
                        fwrite(&D_NS1,2,1,fp2);
                        fwrite(&D_PS1,2,1,fp2);
                        fwrite(&D_NS2,2,1,fp2);
                        fwrite(&D_PS2,2,1,fp2);
                        fwrite(&D_NS3,2,1,fp2);
                        fwrite(&D_PS3,2,1,fp2);


                    }

                }
            }

        }

    }

    fclose(fp2);
    ephem_close();  /* remove this line for use with solsys version 2 */
    clock_t t2=clock();
    printf("Total time=%f seconds\n",(double)(t2-t1)/CLOCKS_PER_SEC);
    return (0);
    printf("Usage:the file is finished.");
}
inline double Parabola(double *r,double *t,int len,double x)
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
    #pragma omp parallel for
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
    k=(x-8.25)/0.5;
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


