#define WIDTH 6
#define NCGF 12
#define HWIDTH 3
#define STEP 4



__kernel void gridVis_wBM_kernel(__global float *cgf,__global float2 *Grd, __global float2 *bm,__global float2 *sf,__global int *cnt,__global float *d_u,__global float *d_v,__global float *d_re,__global float *d_im,int nu, float du,int gcount,int umax,int vmax,int pangle){
     
      int iu=get_global_id(0);
      int iv=get_global_id(1);
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

          int iuuu=iu-uu;
          int ivvv=iv-vv;
          int cnu=abs(iuuu);
          int cnv=abs(ivvv);
          int ind = iv*nu+iu;
          if (cnu < HWIDTH && cnv < HWIDTH){
          float roundcnu=4.6*cnu+NCGF-0.5;
          float roundcnv=4.6*cnv+NCGF-0.5;
          int round1=round(roundcnu);
          int round2=round(roundcnv);
            float wgt = cgf[round1]*cgf[round2];
            Grd[ind].x +=       wgt*d_re[ivis];
            Grd[ind].y += hflag*wgt*d_im[ivis];
            cnt[ind]   += 1;
            bm [ind].x += wgt;
            sf[ind].x   = 1;
            sf[ind].y   = 1;
           }
         
          if (iu-u0 < HWIDTH && mu/du < HWIDTH) {
            mu = -1*mu;
            mv = -1*mv;
            uu = mu/du+u0;
            vv = mv/du+u0;
            int iuuu=iu-uu;
            int ivvv=iv-vv;
            int cnu=abs(iuuu);
            int cnv=abs(ivvv);

            if (cnu < HWIDTH && cnv < HWIDTH){
            float roundcnu=4.6*cnu+NCGF-0.5;
            float roundcnv=4.6*cnv+NCGF-0.5;
            int round1=round(roundcnu);
            int round2=round(roundcnv);
          
            float wgt = cgf[round1]*cgf[round2];
              Grd[ind].x +=          wgt*d_re[ivis];
              Grd[ind].y += -1*hflag*wgt*d_im[ivis];
              cnt[ind]   += 1;
              bm[ind].x += wgt;
              sf[ind].x   = 1;
              sf[ind].y   = 1;            }
          }
        }
      }
    }


__kernel void dblGrid_kernel(__global float2 *Grd, int nu, int hfac){
      int iu=get_global_id(0);
      int iv=get_global_id(1);
      int u0 = 0.5*nu;
      if (iu >= 0 && iu < u0 && iv < nu && iv >= 0){
        int niu = nu-iu;
        int niv = nu-iv;
        Grd[iv*nu+iu].x =      Grd[niv*nu+niu].x;
        Grd[iv*nu+iu].y = hfac*Grd[niv*nu+niu].y;
      }
    }
	

__kernel void wgtGrid_kernel(__global float2 *Grd,__global int *cnt, float briggs, int nu){
      int iu=get_global_id(0);
      int iv=get_global_id(1);
      int u0 = 0.5*nu;
      if (iu >= u0 && iu < nu && iv < nu){
        if (cnt[iv*nu+iu]!= 0){
          int ind = iv*nu+iu;
          float foo = cnt[ind];
          float wgt = 1./sqrt(1 + foo*foo/(briggs*briggs));
          Grd[ind].x = Grd[ind].x*wgt;
          Grd[ind].y = Grd[ind].y*wgt;
        }
      }
    }
   

__kernel void nrmGrid_kernel(__global float *Grd, float nrm, int nu){
      int iu=get_global_id(0);
      int iv=get_global_id(1);
      if ( iu < nu &&  iv < nu){
          Grd[iv*nu + iu] = Grd[iv*nu+iu]*nrm;
      }
    }


__kernel void corrGrid_kernel(__global float2 *Grd,__global float *corr, int nu){
      int iu=get_global_id(0);
      int iv=get_global_id(1);
      if (iu < nu && iv < nu ){
          Grd[iv*nu + iu].x = Grd[iv*nu+iu].x*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
          Grd[iv*nu + iu].y = Grd[iv*nu+iu].y*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
      }
    }


__kernel void nrmBeam_kernel(__global float *bmR, float nrm, int nu){
      int iu=get_global_id(0);
      int iv=get_global_id(1);
      if(iu < nu && iv < nu){
        bmR[iv*nu+iu] = nrm*bmR[iv*nu+iu];
      }
    }


__kernel void shiftGrid_kernel(__global float2 *Grd,__global float2 *nGrd, int nu){
      int iu=get_global_id(0);
      int iv=get_global_id(1);
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


__kernel void trimIm_kernel(__global float2 *im,__global float *nim, int nx, int nnx){
      int ix=get_global_id(0);
      int iy=get_global_id(1);
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix] = im[(nx/2 - nnx/2 + iy)*nx+(nx/2-nnx/2)+ix].x;
      }
    }


__kernel void trim_float_image_kernel(__global float *im, __global float *nim, int nx, int nnx){
      int ix=get_global_id(0);
      int iy=get_global_id(1);
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix] = im[(nx/2 - nnx/2 + iy)*nx+(nx/2 - nnx/2 + ix)];
      }
    }


__kernel void copyIm_kernel(__global float2 *im,__global float *nim, int nx){
      int ix=get_global_id(0);
      int iy=get_global_id(1);
      if(iy < nx && ix < nx){
        nim[iy*nx + ix] = im[(iy)*nx+ix].x;
      }
    }


__kernel void copyRIm_kernel(__global float *im, __global float2 *nim, int nx){
      int ix=get_global_id(0);
      int iy=get_global_id(1);
      if(iy < nx && ix < nx){
        nim[(iy)*nx+ix].x = im[iy*nx + ix];
        nim[(iy)*nx+ix].y = 0;
      }
    }


 __kernel void diskGrid_kernel( __global float2 *Grd, int nu, int radius, int light){
    	
		int iu = get_global_id(0);
    	int iv = get_global_id(1);
      int u0 = 0.5*nu;
      int niu, niv;
      if (iu >= 0 && iu <= u0 && iv <= nu && iv >= 0){
	    float temp=(iu-u0)*(iu-u0)+(iv-u0)*(iv-u0);
		float sqrt1=sqrt(temp);
		if (sqrt1<=radius) {
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



	#define GRID(x,y,W) ((x)+((y)*W))
	__kernel void find_max_kernel(__global float* dimg,__global int* maxid,float maxval, int W, int H, __global float* model)
    {
     
     		int idx = get_global_id(0);
    		int idy = get_global_id(1);
			int id  = GRID(idy,idx,H);

      
		if (idx>-1 && idx<W && idy>-1 && idy<H) {
			
			if (dimg[id]==maxval) {
           	     int dummy = atomic_xchg(maxid,id);
        }
      }
     
       barrier(CLK_LOCAL_MEM_FENCE);
       if (id==maxid[0]) {
        model[id]+=dimg[id];
      }
    }

    #define IGRIDX(x,W) ((x)%(W))
    #define IGRIDY(x,W) ((x)/(W))

    __kernel void sub_beam_kernel(__global  float* dimg, __global  float* dpsf,__global  int* mid,__global  float* cimg,__global  float* cpsf,float scaler,int W, int H, int flag)
    {
    
    	 int idx = get_global_id(0);
   		 int idy = get_global_id(1);
         int id  = GRID(idy,idx,H);
     
	
         int midy = IGRIDX(mid[0],W);
         int midx = IGRIDY(mid[0],H);
     
	
         int bidy = (idx-midx)+W;
         int bidx = (idy-midy)+H;
         int bid = GRID(bidx,bidy,2*W);

     
      if (idx>-1 && idx<W && idy>-1 && idy<H && bidx>-1 && bidx<2*W && bidy>-1 && bidy<2*H) {
		
        dimg[id]=dimg[id]-dpsf[bid]*scaler;
      
        if (flag==1)
           cimg[id]=cimg[id]+cpsf[bid]*scaler;
      }
    }


#define GRID(x,y,W) ((x)+((y)*W))
__kernel void sub_histogram_kernel (__global float *dimg,int W, int H,__global int *histo,int max, int min,	int binsize)
    {
         
				int idx = get_global_id(0);
        		int idy = get_global_id(1);
				
          if (idy<W && idx< H){
              int id  = GRID(idy,idx,H);
              int offset = (dimg[id] - min)*(binsize-1.)/(max-min);
              if (offset<binsize)
                atomic_add (&histo[offset] , 1 );
          }
    }


__kernel void sub_mean_average_kernel(__global  int *in, __global  int *out, int height, int width, int radius)
    {
       
      		int idx = get_global_id(0);
      		int idy = get_global_id(1);
			int id  = idy+idx*width;
			int i = 0;
        
		float ss = 0;
        out[id] = 0;
        if (idy < width - radius -1) {
            for (i=0;i<radius;i++)
                 ss += in[id+i];
           	 int s = ss/radius;
		  out[id] = s;
	}
	}
	

__kernel void sub_cuda_cyclic_shift_kernel(__global float2 *in,__global float2 *out, int N,	int x_offset, int y_offset)
    {
        int tidx = get_global_id(0);
       	int tidy = get_global_id(1);
        
		float cosv,sinv;
         if ( tidx < N && tidy <N )
         {
            int index=tidx*N+tidy;
            float angle=2*3.1415926535*(x_offset*(tidx-N/2)+y_offset*(tidy-N/2))/N;
		    sinv =  sin(angle);
            cosv =  cos(angle);
            out[index].x = in[index].x*cosv-in[index].y*sinv;
            out[index].y = in[index].x*sinv+in[index].y*cosv;
         }
     }


__kernel void sub_cycle_shift_kernel( __global float *im, __global float *nim, int nx, int x_offset, int y_offset){
     	int ix = get_global_id(0);
    	int iy = get_global_id(1);
			
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


__kernel void sub_rotate_image_kernel(__global float* src,__global float* trg, int imageWidth,int imageHeight, 	float angle,float scale)
    {
       
			int x = get_global_id(0);
    		int y = get_global_id(1);
    
       
			int idx = x + y * imageWidth;
			int xA = (x - imageWidth/2 );
			int yA = (y - imageHeight/2 );

            int xR = (int)floor(1.0f/scale * (xA * cos(angle) - yA * sin(angle)));
			int yR = (int)floor(1.0f/scale * (xA * sin(angle) + yA * cos(angle)));

			float src_x = xR + imageWidth/2;
			float src_y = yR + imageHeight/2;

        if ( src_x >= 0.0f && src_x < imageWidth && src_y >= 0.0f && src_y < imageHeight) {
          
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


__kernel void sub_dot_mul_kernel(__global const float2 *A, __global const float2 *B, __global const float2 *C, int width, int height){
   
		int idx = get_global_id(0);
		int idy = get_global_id(1);
        int id  = idy+idx*width;

 
        C[id].x = A[id].x*B[id].x - A[id].y*B[id].y ;              
        C[id].y = A[id].y*B[id].x + A[id].x*B[id].y ;
	}
	


#define MATRIX_SIZE 1024
#define BLOCK_SIZE 8
__kernel void sub_matrix_mul_kernel(__global const float2 *A, __global const float2 *B, __global const float2 *C, int wA, int wB)
    {
        
       	    int bx = get_local_size(0);
   			int by = get_local_size(1);

         	int tx = get_local_id(0);
        	int ty = get_local_id(1);

			int aBegin = wA * BLOCK_SIZE * by;
       
			int aEnd   = aBegin + wA - 1;
       
			int aStep = BLOCK_SIZE;

       
			int bBegin = BLOCK_SIZE * bx;
        
			int bStep = BLOCK_SIZE * wB;

       
			float2 Csub;
			Csub.x = 0; Csub.y =0; //= make_cuFloatComplex(0,0);
       
			for (int a = aBegin, b = bBegin; a <= aEnd;    a += aStep, b += bStep)
          {
            
               __local float2 As[BLOCK_SIZE][BLOCK_SIZE];
           
               __local float2 Bs[BLOCK_SIZE][BLOCK_SIZE];

          
               As[ty][tx].x = A[a + wA*ty + tx].x; As[ty][tx].y = A[a + wA*ty + tx].y;
               Bs[ty][tx].x = B[b + wB*ty + tx].x; Bs[ty][tx].y = B[b + wB*ty + tx].y;

            
               barrier(CLK_LOCAL_MEM_FENCE);

         
               for(int k = 0; k < BLOCK_SIZE; ++k)
               {
                    Csub.x = Csub.x + (As[ty][k].x*Bs[k][tx].x - As[ty][k].y*Bs[k][tx].y ); //cuCaddf(Csub,cuCmulf(As[ty][k],Bs[k][tx]));
                    Csub.y = Csub.y + (As[ty][k].y*Bs[k][tx].x + As[ty][k].x*Bs[k][tx].y );
               }

           
               barrier(CLK_LOCAL_MEM_FENCE);
         }

       
         int cc = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
         C[cc + wB*ty + tx].x = Csub.x;
         C[cc + wB*ty + tx].y = Csub.y;
    }


__kernel void rotateImage_Kernel(__global float2* trg, __global const float2* src, 
	int imageWidth,int imageHeight, float angle, float scale)
	{    
   
		int x = get_global_id(0);
   		int y = get_global_id(1);

    
	int idx = x + y * imageWidth;
    int xA = (x - imageWidth/2 );
    int yA = (y - imageHeight/2 );

    int xR = (int)floor(1.0f/scale * (xA * cos(angle) - yA * sin(angle)));
    int yR = (int)floor(1.0f/scale * (xA * sin(angle) + yA * cos(angle)));

    float src_x = xR + imageWidth/2;
    float src_y = yR + imageHeight/2;

     if ( src_x >= 0.0f && src_x < imageWidth && src_y >= 0.0f && src_y < imageHeight) {
      
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

	
	}
			
__kernel void translateImage_Kernel(__global float2* trg, __global const float2* src, 
	const unsigned int imageWidth, const unsigned int imageHeight, const float tX, const float tY)
	{ 

		const unsigned int x = get_global_id(0);
 		const unsigned int y = get_global_id(1);
		
   
    const unsigned int idx = x + y * imageWidth;

    const int xB = ((int)x + (int)tX );
    const int yB = ((int)y + (int)tY );

    if ( xB >= 0 && xB < imageWidth && yB >= 0 && yB < imageHeight) {
        trg[idx] = src[xB + yB * imageWidth];
    } else {
        trg[idx].x = 0.0f;
        trg[idx].y = 0.0f;
    }
	
	}

	
	

