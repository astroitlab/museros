#!python
#!/usr/bin/env python -tt
# encoding: utf-8
#
# Created by Holger Rapp on 2009-03-11.
# HolgerRapp@gmx.net
#

import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel

import numpy
from math import pi,cos,sin

clean_code = \
    """
    #define WIDTH 6
    #define NCGF 12
    #define HWIDTH 3
    #define STEP 4

    __device__ __constant__ float cgf[32];

    // *********************
    // MAP KERNELS
    // *********************

    __global__ void gridVis_wBM_kernel(float2 *Grd, float2 *bm, float2 *sf, int *cnt, float *d_u, float *d_v, float *d_re,
        float *d_im, int nu, float du, int gcount, int umax, int vmax, int pangle){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if(iu >= u0 && iu <= u0+umax && iv <= u0+vmax){
        for (int ivis = 0; ivis < gcount; ivis++){
          float mu = d_u[ivis];
          float mv = d_v[ivis];
          int hflag = 1;
          if (mu < 0){
            hflag = -1;
            mu = -1*mu;
            mv = -1*mv;
          }
          float uu, vv;
          uu = mu/du + u0;
          vv = mv/du + u0;
          int cnu=abs(iu-uu),cnv=abs(iv-vv);
          int ind = iv*nu+iu;
          if (cnu < HWIDTH && cnv < HWIDTH){
            float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
            Grd[ind].x +=       wgt*d_re[ivis];
            Grd[ind].y += hflag*wgt*d_im[ivis];
            cnt[ind]   += 1;
            bm [ind].x += wgt;
            sf[ind].x   = 1;
            sf[ind].y   = 1;
           }
          // deal with points&pixels close to u=0 boundary
          if (iu-u0 < HWIDTH && mu/du < HWIDTH) {
            mu = -1*mu;
            mv = -1*mv;
            uu = mu/du+u0;
            vv = mv/du+u0;
            cnu=abs(iu-uu),cnv=abs(iv-vv);
            if (cnu < HWIDTH && cnv < HWIDTH){
              float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
              Grd[ind].x +=          wgt*d_re[ivis];
              Grd[ind].y += -1*hflag*wgt*d_im[ivis];
              cnt[ind]   += 1;
              bm [ind].x += wgt;
              sf[ind].x   = 1;
              sf[ind].y   = 1;            }
          }
        }
      }
    }

    __global__ void dblGrid_kernel(float2 *Grd, int nu, int hfac){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if (iu >= 0 && iu < u0 && iv < nu && iv >= 0){
        int niu = nu-iu;
        int niv = nu-iv;
        Grd[iv*nu+iu].x =      Grd[niv*nu+niu].x;
        Grd[iv*nu+iu].y = hfac*Grd[niv*nu+niu].y;
      }
    }

    __global__ void wgtGrid_kernel(float2 *Grd, int *cnt, float briggs, int nu, int mode){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if (iu >= u0 && iu < nu && iv < nu){
        if (cnt[iv*nu+iu]!= 0){
          int ind = iv*nu+iu;
          float foo = cnt[ind];
          float wgt;
          if (mode == 1)
             wgt = 1./foo;  /*sqrt(1 + foo*foo/(briggs*briggs));*/
          else
             wgt = 1;
          Grd[ind].x = Grd[ind].x*wgt;
          Grd[ind].y = Grd[ind].y*wgt;
        }
      }
    }

    __global__ void nrmGrid_kernel(float *Grd, float nrm, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if ( iu < nu &&  iv < nu){
          Grd[iv*nu + iu] = Grd[iv*nu+iu]*nrm;
      }
    }

    __global__ void corrGrid_kernel(float2 *Grd, float *corr, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if (iu < nu && iv < nu ){
          Grd[iv*nu + iu].x = Grd[iv*nu+iu].x*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
          Grd[iv*nu + iu].y = Grd[iv*nu+iu].y*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
      }
    }

    // *********************
    // BEAM KERNELS
    // *********************
    __global__ void nrmBeam_kernel(float *bmR, float nrm, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if(iu < nu && iv < nu){
        bmR[iv*nu+iu] = nrm*bmR[iv*nu+iu];
      }
    }

    // *********************
    // MORE semi-USEFUL KERNELS
    // *********************

    __global__ void shiftGrid_kernel(float2 *Grd, float2 *nGrd, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if(iu < nu && iv < nu){
        int niu,niv,nud2 = 0.5*nu;
        if(iu < nud2) niu = nud2+iu;
          else niu = iu-nud2;
        if(iv < nud2) niv = nud2+iv;
          else niv = iv-nud2;
        nGrd[niv*nu + niu].x = Grd[iv*nu+iu].x;
        nGrd[niv*nu + niu].y = Grd[iv*nu+iu].y;
      }
    }

    __global__ void trimIm_kernel(float2 *im, float *nim, int nx, int nnx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix] = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].x;
      }
    }

    __global__ void trimIm_kernel_2(float2 *im, float2 *nim, int nx, int nnx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix].x = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].x;
        nim[iy*nnx + ix].y = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].y;
      }
    }

    __global__ void trimDisk_kernel_2(float2 *im, float *nim, int nnx, int r){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        if (( -nnx/2 + iy)*( -nnx/2 + iy)+(nnx/2 - ix)*(nnx/2-ix) > r*r)
           nim[iy*nnx + ix] = 0;
        else
           nim[iy*nnx + ix] = im[iy*nnx+ix].x;
      }
    }

    __global__ void trimDisk_kernel(float2 *im, float *nim, int nx, int nnx, int r){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        if (( -nnx/2 + iy)*( -nnx/2 + iy)+(nnx/2 - ix)*(nnx/2-ix) > r*r)
           nim[iy*nnx + ix] = 0;
        else
           nim[iy*nnx + ix] = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].x;
      }
    }

    __global__ void sub_image_kernel(float *im, float value, int r, int nx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
         if (( -nx/2 + iy)*( -nx/2 + iy)+(nx/2 - ix)*(nx/2-ix) <= int(r*r))
            im[iy*nx + ix] = im[iy*nx + ix]- value;
      }
    }

    __global__ void add_image_kernel(float *im, float value, int r, int nx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
         if (( -nx/2 + iy)*( -nx/2 + iy)+(nx/2 - ix)*(nx/2-ix) <= int(r*r))
            im[iy*nx + ix] = im[iy*nx + ix] + value;
      }
    }

    __global__ void add_flat_kernel(float *im, float value, int nx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
            im[iy*nx + ix] = im[iy*nx + ix] + value;
      }
    }


    __global__ void add_clean_kernel(float *im, float *nim, float *disk, int nx, float s1, float s2, int back, int max, int r){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
        if (( -nx/2 + iy)*( -nx/2 + iy)+(nx/2 - ix)*(nx/2-ix) > int(1.44*r*r))
            im[iy*nx + ix] = 0;
        else{
           im[iy*nx + ix] = (im[iy*nx + ix]*max*s1+nim[iy*nx + ix]*max*s2) + disk[iy*nx+ix]*back;
           im[iy*nx + ix] = (im[iy*nx + ix]*max*s1+nim[iy*nx + ix]*max*s2)/2. + disk[iy*nx+ix]*back;
           //im[iy*nx + ix] = (im[iy*nx + ix]*max*s1>0?im[iy*nx + ix]*max*s1:0+nim[iy*nx + ix]*max*s2>0?nim[iy*nx + ix]*max*s2:0)/2. + disk[iy*nx+ix]*back;
        }
        if (( -nx/2 + iy)*( -nx/2 + iy)+(nx/2 - ix)*(nx/2-ix) > int(r*r))
            if (im[iy*nx+ix] < back*0.9)
               im[iy*nx + ix] = 0;
      }
    }


    __global__ void trim_float2_kernel(float2 *im, float2 *nim, int nx, int nnx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix].x = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].x;
        nim[iy*nnx + ix].y = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].y;
      }
    }

    __global__ void trim_float_image_kernel(float *im, float *nim, int nx, int nnx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix] = im[(nx/2 - nnx/2 + iy)*nx+(nx/2 - nnx/2 + ix)];
      }
    }

    __global__ void copy_float_kernel(float *im, float *nim, int nx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
        nim[iy*nx + ix] = im[(iy)*nx+ix];
      }
    }

    __global__ void copyIm_kernel(float2 *im, float *nim, int nx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
        nim[iy*nx + ix] = im[(iy)*nx+ix].x;
      }
    }

    __global__ void copyRIm_kernel(float *im, float2 *nim, int nx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
        nim[(iy)*nx+ix].x = im[iy*nx + ix];
        nim[(iy)*nx+ix].y = 0;
      }
    }

    __global__ void sun_disk_kernel(float *Grd, float *light, int nnx, int r){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      float radius,radius1;
      int rstart;

      if(iy < nnx && ix < nnx){
        Grd[iy*nnx + ix]= 0;
        radius = (-nnx/2 + iy)*( -nnx/2 + iy)+(nnx/2 - ix)*(nnx/2-ix);
        if (radius <= 4*r*r ) {
            if (r>=100) {
                if (r==100){
                    rstart = int(sqrt(radius));
                    if (rstart>=0 && rstart < 200)
                        Grd[iy*nnx + ix] = float(light[rstart]*1.0);
                }
                else{
                    radius1 = sqrt(radius)*100./r;
                    rstart = int(radius1);
                    Grd[iy*nnx + ix] = (light[rstart]+(light[rstart+1]-light[rstart])*(radius1-int(radius1)));
                }
            }
            else
            {
                radius1 = sqrt(radius)*r/100.;
                rstart = int(radius1);
                Grd[iy*nnx + ix] = (light[rstart]+(light[rstart+1]-light[rstart])*(radius1-int(radius1)));
            }
        }
      }
    }

    __global__ void diskGrid_kernel(float2 *Grd, int nu, int radius, int light){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      int niu, niv;
      if (iu >= 0 && iu <= u0 && iv <= nu && iv >= 0){
        if (sqrt(float(iu-u0)*(iu-u0)+float(iv-u0)*(iv-u0))<=float(radius)) {
            Grd[iv*nu+iu].x = light;
            Grd[iv*nu+iu].y = 0;
            niu = nu- iu;
            niv = iv ;
            Grd[niv*nu+niu].x = light;
            Grd[niv*nu+niu].y = 0;
            niu = iu;
            niv = nu- iv;
            Grd[niv*nu+niu].x = light;
            Grd[niv*nu+niu].y = 0;
            niu = nu- iu;
            niv = nu- iv;
            Grd[niv*nu+niu].x = light;
            Grd[niv*nu+niu].y = 0;
        }
      }
    }


    __global__ void copyContour_kernel(float2 *Grd, float2 *New, float *clean,  int nx, float vis){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
        if (Grd[(iy)*nx+ix].x>=vis){
            New[iy*nx + ix].x = Grd[(iy)*nx+ix].x;
            New[iy*nx + ix].y = 0;
            clean[iy*nx+ix] += (Grd[(iy)*nx+ix].x - vis);
        }
        else
        {
            New[iy*nx + ix].x = 0.;
            New[iy*nx + ix].y = 0.;
        }
      }

    }

    """
# -------------------
# CLEAN kernels
# -------------------

find_max_kernel_source = \
    """
    // Function to compute 1D array positioncuda_gri
    #define GRID(x,y,W) ((x)+((y)*W))

    __global__ void find_max_kernel(float* dimg, int* maxid, float maxval, int W, int H, float* model)
    {
      // Identify place on grid
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      int id  = GRID(idy,idx,H);

      if (idx>-1 && idx<W && idy>-1 && idy<H) {
        // Is this greater than the current max?
        if (dimg[id]==maxval) {
          // Do an atomic replace
          // This might be #improvable#, but I think atomic operations are actually most efficient
          // in a situation like this where few (typically 1) threads will pass this conditional.
          // Note: this is a race condition!  If there are multiple instances of the max value,
          // this will end up returning one randomly
          // See atomic operation info here: http://rpm.pbone.net/index.php3/stat/45/idpl/12463013/numer/3/nazwa/atomicExch
          // See also https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
          int dummy = atomicExch(maxid,id);
        }
      }
      // Update the model
      void __syncthreads();
      if (id==maxid[0]) {
        model[id]+=dimg[id];
      }
    }
    """

sub_steer_kernel_source = \
    """
    // Function to compute 1D array position
    #define GRID(x,y,W) ((x)+((y)*W))
    // Inverse
    #define IGRIDX(x,W) ((x)%(W))
    #define IGRIDY(x,W) ((x)/(W))

    __global__ void sub_steer_kernel(float2* dimg, float* dpsf, float scaler, int nx)
    {
      // Identify place on grid
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      int id  = GRID(idy,idx,nx);

      // Stay within the bounds
      if (idx>-1 && idx<nx && idy>-1 && idy<nx) {
        if (dpsf[id]>0)
            dimg[id].x=dimg[id].x-dpsf[id]*scaler;
        dimg[id].y=0;
      };
    }
    """

sub_beam_kernel_source = \
    """
    // Function to compute 1D array position
    #define GRID(x,y,W) ((x)+((y)*W))
    // Inverse
    #define IGRIDX(x,W) ((x)%(W))
    #define IGRIDY(x,W) ((x)/(W))

    __global__ void sub_beam_kernel(float* dimg, float* dpsf, int* mid, float* cimg, float* cpsf, float scaler, int W, int H, int flag)
    {
      // Identify place on grid
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      int id  = GRID(idy,idx,H);
      // Identify position of maximum
      int midy = IGRIDX(mid[0],W);
      int midx = IGRIDY(mid[0],H);
      // Calculate position on the dirty beam
      int bidy = (idx-midx)+W;
      int bidx = (idy-midy)+H;
      int bid = GRID(bidx,bidy,2*W);

      // Stay within the bounds
      if (idx>-1 && idx<W && idy>-1 && idy<H && bidx>-1 && bidx<2*W && bidy>-1 && bidy<2*H) {
      //if (idx>-1 && idx<W && idy>-1 &&  idy<H) {
        // Subtract dirty beam from dirty map
        dimg[id]=dimg[id]-dpsf[bid]*scaler;
        // Add clean beam to clean map
        if (flag==1)
           cimg[id]=cimg[id]+cpsf[bid]*scaler;
      };
    }
    """

add_noise_kernel = ElementwiseKernel(
        "float *a, float* b, int N",
        "b[i] = a[i]+b[i]",
        "gpunoise")

histogram_kernel_source = \
    """
     #define GRID(x,y,W) ((x)+((y)*W))

    __global__ void sub_histogram_kernel (float *dimg, int W, int H, int *histo, int max, int min, int binsize)
    {
          // Identify place on grid
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          int idy = blockIdx.y * blockDim.y + threadIdx.y;
          if (idy<W && idx< H){
              int id  = GRID(idy,idx,H);
              int offset = int((dimg[id] - min)*(binsize-1.)/(max-min));
              if (offset<binsize)
                atomicAdd (&histo[offset] , 1 );
          }
    }
    """

filter_kernel_source = \
    """
    __global__ void sub_mean_average_kernel(int *in, int *out, int height, int width, int radius)
    {
        // Identify place on grid
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int id  = idy+idx*width;
        int i = 0;
        float ss = 0;
        out[id] = 0;
        if (idy < width - radius -1) {
            for (i=0;i<radius;i++)
                ss += in[id+i];
            out[id] = int(ss/radius);
        }
    }
    """

sub_cuda_cyclic_shift_kernel_source = \
    """
    __global__ void sub_cuda_cyclic_shift_kernel(float2 *in, float2 *out, int N, int x_offset, int y_offset)
    {
         int tidx = threadIdx.x +  blockIdx.x * blockDim.x;
         int tidy = threadIdx.y +  blockIdx.y * blockDim.y;
         float cosv,sinv;
         if ( tidx < N && tidy <N )
         {
             int index=tidx*N+tidy;
             float angle=2*3.1415926535*(x_offset*(tidx-N/2)+y_offset*(tidy-N/2))/N;
             sincos(angle,&sinv,&cosv);
             out[index].x = in[index].x*cosv-in[index].y*sinv;
             out[index].y = in[index].x*sinv+in[index].y*cosv;
         }
     }
     """


cycle_shift_kernel_source = \
    """
    __global__ void sub_cycle_shift_kernel(float *im, float *nim, int nx, int x_offset, int y_offset){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nx && ix < nx){
        int x = (ix+x_offset);
        int y = (iy+y_offset);
        if (x>nx)
            x = x%nx;
        if (x<0)
            x = nx+x;
        if (y>nx)
            y = y%nx;
        if (y<0)
            y = nx+y;
        if (y<nx && x < nx)
            nim[y*nx + x] = im[(iy)*nx+ix];
      }
    }
    """

sub_rotation_kernel_source = \
    """
    __global__ void sub_rotate_image_kernel(float* src, float* trg,  int imageWidth,int imageHeight, float angle, float scale)
    {
        // compute thread dimension
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        //// compute target address
        int idx = x + y * imageWidth;

        int xA = (x - imageWidth/2 );
        int yA = (y - imageHeight/2 );

        int xR = (int)floor(1.0f/scale * (xA * cos(angle) - yA * sin(angle)));
        int yR = (int)floor(1.0f/scale * (xA * sin(angle) + yA * cos(angle)));

        float src_x = xR + imageWidth/2;
        float src_y = yR + imageHeight/2;

         if ( src_x >= 0.0f && src_x < imageWidth && src_y >= 0.0f && src_y < imageHeight) {
            // BI - LINEAR INTERPOLATION
            float src_x0 = (float)(int)(src_x);
            float src_x1 = (src_x0+1);
            float src_y0 = (float)(int)(src_y);
            float src_y1 = (src_y0+1);

            float sx = (src_x-src_x0);
            float sy = (src_y-src_y0);


            int idx_src00 = min(max(0.0f,src_x0   + src_y0 * imageWidth),imageWidth*imageHeight-1.0f);
            int idx_src10 = min(max(0.0f,src_x1   + src_y0 * imageWidth),imageWidth*imageHeight-1.0f);
            int idx_src01 = min(max(0.0f,src_x0   + src_y1 * imageWidth),imageWidth*imageHeight-1.0f);
            int idx_src11 = min(max(0.0f,src_x1   + src_y1 * imageWidth),imageWidth*imageHeight-1.0f);

            trg[idx] = 0.0f;

            trg[idx]  = (1.0f-sx)*(1.0f-sy)*src[idx_src00];
            trg[idx] += (     sx)*(1.0f-sy)*src[idx_src10];
            trg[idx] += (1.0f-sx)*(     sy)*src[idx_src01];
            trg[idx] += (     sx)*(     sy)*src[idx_src11];
         } else {
            trg[idx] = 0.0f;
         }
    }
    """

dot_mul_kernel_source = \
    """
    __global__ void sub_dot_mul_kernel(float2 *A, float2 *B, float2 *C, int width, int height)
    {
        // Identify place on grid
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int id  = idy+idx*width;

        C[id].x = A[id].x*B[id].x - A[id].y*B[id].y ; //cuCaddf(Csub,cuCmulf(As[ty][k],Bs[k][tx]));
        C[id].y = -(A[id].y*B[id].x + A[id].x*B[id].y );
    }
    """

matrix_mul_kernel_source = \
    """
    //#include <cuComplex.h>
    #define MATRIX_SIZE 1024
    #define BLOCK_SIZE 8

    __global__ void sub_matrix_mul_kernel(float2 *A, float2 *B, float2 *C, int wA, int wB)
    {
          //const int wA = MATRIX_SIZE;
          //const int wB = MATRIX_SIZE;

          // Block index
          int bx = blockIdx.x;
          int by = blockIdx.y;

          // Thread index
          int tx = threadIdx.x;
          int ty = threadIdx.y;

          // Index of the first sub-matrix of A processed by the block
          int aBegin = wA * BLOCK_SIZE * by;
          // Index of the last sub-matrix of A processed by the block
          int aEnd   = aBegin + wA - 1;
          // Step size used to iterate through the sub-matrices of A
          int aStep = BLOCK_SIZE;

          // Index of the first sub-matrix of B processed by the block
          int bBegin = BLOCK_SIZE * bx;
          // Step size used to iterate through the sub-matrcies of B
          int bStep = BLOCK_SIZE * wB;

          // The element of the block sub-matrix that is computed by the thread
          float2 Csub;
          Csub.x = 0; Csub.y =0; //= make_cuFloatComplex(0,0);
          // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
          for (int a = aBegin, b = bBegin; a <= aEnd;    a += aStep, b += bStep)
          {
               // Shared memory for the sub-matrix of A
               __shared__ float2 As[BLOCK_SIZE][BLOCK_SIZE];
               // Shared memory for the sub-matrix of B
               __shared__ float2 Bs[BLOCK_SIZE][BLOCK_SIZE];

               // Load the matrices from global memory to shared memory;
               // each thread loads one element of each matrix
               As[ty][tx].x = A[a + wA*ty + tx].x; As[ty][tx].y = A[a + wA*ty + tx].y;
               Bs[ty][tx].x = B[b + wB*ty + tx].x; Bs[ty][tx].y = B[b + wB*ty + tx].y;

               // Synchronize to make sure the matrices are loaded
               __syncthreads();

               // Multiply the two matrcies together
               // each thread computes one element of the block sub-matrix
               for(int k = 0; k < BLOCK_SIZE; ++k)
               {
                    Csub.x = Csub.x + (As[ty][k].x*Bs[k][tx].x - As[ty][k].y*Bs[k][tx].y ); //cuCaddf(Csub,cuCmulf(As[ty][k],Bs[k][tx]));
                    Csub.y = Csub.y + (As[ty][k].y*Bs[k][tx].x + As[ty][k].x*Bs[k][tx].y );
               }

               // Synchronize to make sure that the preceding computation
               // is done before loading two new sub-matrices of A and B in the next iteration
               __syncthreads();
         }

         // Write the block sub-matrix to global memory
         // each thread writes one element
         int cc = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
         C[cc + wB*ty + tx].x = Csub.x;
         C[cc + wB*ty + tx].y = Csub.y;
    }
    """

_rotation_kernel_source = """
__global__ void rotateImage_Kernel(float2* trg, const float2* src, int imageWidth,int imageHeight, float angle, float scale)
{
    // compute thread dimension
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //// compute target address
    int idx = x + y * imageWidth;

    int xA = (x - imageWidth/2 );
    int yA = (y - imageHeight/2 );

    int xR = (int)floor(1.0f/scale * (xA * cos(angle) - yA * sin(angle)));
    int yR = (int)floor(1.0f/scale * (xA * sin(angle) + yA * cos(angle)));

    float src_x = xR + imageWidth/2;
    float src_y = yR + imageHeight/2;

     if ( src_x >= 0.0f && src_x < imageWidth && src_y >= 0.0f && src_y < imageHeight) {
        // BI - LINEAR INTERPOLATION
        float src_x0 = (float)(int)(src_x);
        float src_x1 = (src_x0+1);
        float src_y0 = (float)(int)(src_y);
        float src_y1 = (src_y0+1);

        float sx = (src_x-src_x0);
        float sy = (src_y-src_y0);


        int idx_src00 = min(max(0.0f,src_x0   + src_y0 * imageWidth),imageWidth*imageHeight-1.0f);
        int idx_src10 = min(max(0.0f,src_x1   + src_y0 * imageWidth),imageWidth*imageHeight-1.0f);
        int idx_src01 = min(max(0.0f,src_x0   + src_y1 * imageWidth),imageWidth*imageHeight-1.0f);
        int idx_src11 = min(max(0.0f,src_x1   + src_y1 * imageWidth),imageWidth*imageHeight-1.0f);

        trg[idx].y = 0.0f;

        trg[idx].x  = (1.0f-sx)*(1.0f-sy)*src[idx_src00].x;
        trg[idx].x += (     sx)*(1.0f-sy)*src[idx_src10].x;
        trg[idx].x += (1.0f-sx)*(     sy)*src[idx_src01].x;
        trg[idx].x += (     sx)*(     sy)*src[idx_src11].x;
     } else {
        trg[idx].x = 0.0f;
        trg[idx].y = 0.0f;
     }

    DEVICE_METHODE_LAST_COMMAND;

}


void translateImage_Kernel(float2* trg, const float2* src, const unsigned int imageWidth, const unsigned int imageHeight, const float tX, const float tY)
{
    // compute thread dimension
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //// compute target address
    const unsigned int idx = x + y * imageWidth;

    const int xB = ((int)x + (int)tX );
    const int yB = ((int)y + (int)tY );

    if ( xB >= 0 && xB < imageWidth && yB >= 0 && yB < imageHeight) {
        trg[idx] = src[xB + yB * imageWidth];
    } else {
        trg[idx].x = 0.0f;
        trg[idx].y = 0.0f;
    }

    DEVICE_METHODE_LAST_COMMAND;

}
"""
