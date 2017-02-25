#coding=utf-8
#**********************************************************
import math
import numpy as np
import matplotlib
import datetime

from matplotlib import rc
import matplotlib.font_manager as fmanger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

plt.switch_backend('agg')

class MuserDraw:

    def xLables(self,xarry):
        new_x = []
        for i in range(len(xarry)):
            if xarry[i]<0:
                temp_x = "$-"+str(int(math.fabs(xarry[i])/60))+"^h"+str(int(math.fabs(xarry[i])%60)).zfill(2)+"^m$"
            else:
                temp_x = "$"+str(int(xarry[i]/60))+"^h"+str(int(xarry[i]%60)).zfill(2)+"^m$"
            new_x.append(temp_x)
        return new_x

    def yLables(self,yarry):
        new_y = []
        for i in range(len(yarry)):
            if yarry[i]<0:
                temp_y = "$-"+str(int(math.fabs(yarry[i])/60))+'^\circ'+str(int(math.fabs(yarry[i])%60)).zfill(2)+"'$"
            else:
                temp_y = "$"+str(int(yarry[i]/60))+'^\circ'+str(int(yarry[i]%60)).zfill(2)+"'$"
            new_y.append(temp_y)
        return new_y

    #reflacation (a,b) to (0,1)
    def reflaction(self,start,end):
        k = 1.0/(end-start)
        t = start*1./(start-end)
        return (k,t)

    def line_fun(self,param,inputs):
        return  param[0]*inputs+param[1]

    def convert_to_degree(self,start,end):
        if start >end:
            start = (int(start),math.floor((start-int(start))*60))
            end   =  (int(end),math.floor((end-int(end))*60))
        else:
            start = (int(start),math.ceil((start-int(start))*60))
            end   =  (int(end),math.ceil((end-int(end))*60))
        return start,end

    def define_gap_axs(self,start,end,gap_num):
        # start param celiling
        new_start,new_end = self.convert_to_degree(start, end)
        temp_start = new_start[0]*60+new_start[1]
        temp_end =new_end[0]*60+new_end[1]
        temp_gap = math.fabs(temp_end -temp_start)
        gap = math.ceil(1.*temp_gap/gap_num)
        new_axs = []
        for i in range(gap_num):
            if(new_start>new_end):
                temp = int(temp_start-gap*i)
                if temp>temp_end:
                    new_axs.append(temp)
            else:
                temp = int(temp_start+gap*i)
                if temp<temp_end:
                    new_axs.append(temp)
        return new_axs


    def degree_to_num(self,deg_array):

        num_array = []
        for i in range(len(deg_array)):
            if(deg_array[i]<0):
                temp = 1.*math.fabs(deg_array[i])/60.0
                temp = temp*(-1.0)
            else:
                temp = deg_array[i]/60.0
            num_array.append(temp)
        return num_array



    def conver_data(self,start,end,axs_array):
        param = self.reflaction(start, end)
        for i in range(len(axs_array)):
            axs_array[i] = self.line_fun(param, axs_array[i])
        return axs_array

    def draw_one(self,filename, title, fov, data, xleft=0, xright=0, ybottom=0, ytop=0, radius=0, axis=True, axistype = 0):
        if axistype ==0:
            vra = [np.percentile(data, 1), np.percentile(data, 100)]
            fig=plt.figure(figsize=(8,6), dpi=800, facecolor="white")

            font = {'family': 'sans-serif',
                    'weight': 'normal',
                    'size': 11,
                    }

            rc('text', usetex=False)
            rc('font', **font)
            norm = matplotlib.colors.Normalize(vmin=vra[0], vmax=vra[1])
            axs = plt.subplot(111)
            cmap = matplotlib.cm.gist_heat

            axs.set_title(title, fontdict=font)

            im = axs.imshow(data, vmin=vra[0], vmax=vra[1], cmap=cmap, origin='upper', norm=norm)
            plt.colorbar(im)

            #print "xright", xleft, xright, ybottom, ytop

            gap_x = self.define_gap_axs(xleft, xright, 10)
            gap_y = self.define_gap_axs(ybottom, ytop, 10)
            num_x = self.degree_to_num(gap_x)
            num_y = self.degree_to_num(gap_y)

            # x_lables = self.xLables(gap_x)
            # y_lables = self.yLables(gap_y)

            font = {'family': 'sans-serif',
                    'weight': 'normal',
                    'size': 14,
                    }

            ax = plt.gca()
            # ax.set_xticklabels(x_lables, fontdict=font)
            # ax.set_yticklabels(y_lables, fontdict=font)

            # axs.set_ylabel('J2000 Declination', fontdict=font)
            # axs.set_xlabel('J2000 Right Ascension', fontdict=font)

            plt.savefig(filename)

            plt.close()

        else:
            # if not axis:
            #     fig = plt.figure(figsize=(6,6), dpi=300, facecolor="white")
            # else:
            fig = plt.figure(figsize=(8,6), dpi=800, facecolor="white")
            axes = plt.subplot(111)
            axes.cla()

            sizeOfFont = 9
            fontProperties = {'family':'sans-serif',#'sans-serif':['Helvetica'],
                              'weight' : 'normal', 'size' : sizeOfFont}
            ticks_font = fmanger.FontProperties(family='Helvetica', style='normal',
                                                     size=sizeOfFont, weight='normal', stretch='normal')
            rc('text', usetex=False)
            rc('font',**fontProperties)
            gap_x = self.define_gap_axs(xleft, xright, 10)
            gap_y = self.define_gap_axs(ybottom, ytop, 10)
            num_x = self.degree_to_num(gap_x)
            num_y = self.degree_to_num(gap_y)

            x = self.conver_data(xleft,xright,num_x)
            y = self.conver_data( ybottom, ytop, num_y)

            # x_lables = self.xLables(gap_x)
            # y_lables = self.yLables(gap_y)
            cbary = np.linspace(0,1,5)

            extent=(0,1,0,1)
            cmap = matplotlib.cm.gist_heat

            vra = [np.percentile(data, 1), np.percentile(data, 100)]
            norm = matplotlib.colors.Normalize(vmin=vra[0], vmax=vra[1])
            if not axis:
                # gci=plt.imshow(np.flipud(data), extent=extent, origin='lower',cmap=cmap, vmin=vra[0], vmax=vra[1])
                gci=plt.imshow(data, extent=extent, origin='upper',cmap=cmap, norm=norm)
                plt.axis('off')
                gci.axes.get_xaxis().set_visible(False)
                gci.axes.get_yaxis().set_visible(False)
            else:
                gci=plt.imshow(data, extent=extent, origin='upper',cmap=cmap, norm=norm)
                # gci=plt.imshow(np.flipud(data), extent = extent,  origin='lower',cmap=cmap, vmin=vra[0], vmax=vra[1])

                cbar = plt.colorbar(gci)
            # cbar.set_ticks(np.linspace(vra[0],vra[1],10))
            # cbar.set_ticklabels(cbary,fontProperties)

            if axis:
                ax=plt.gca()
                # ax.set_xticklabels(ax.get_xticks(), fontProperties)
                # ax.set_yticklabels(ax.get_yticks(), fontProperties)


                ax.set_xticks(x)
                # ax.set_xticklabels(x_lables,fontProperties)
                ax.set_yticks(y)
                # ax.set_yticklabels(y_lables,fontProperties)

                # ax.set_ylabel('J2000 Declination')
                # ax.set_xlabel('J2000 Right Ascension')


                plt.title(title)
            figname="sw1"+'.png'
            # ax1 = fig.add_subplot(111)
            # ax1.add_patch(
            #     patches.Circle(
            #     (0.5, 0.5),   # (x,y)
            #     radius,          # radius
            #     fill = False,
            #     edgecolor="white"
            #     )
            # )
            if not axis:
                plt.savefig(filename, bbox_inches='tight', pad_inches = 0)
            else:
                plt.savefig(filename)

            plt.show()

# if __name__ == '__main__':
#     muser = MuserDraw()
#     temp = gData.Data(100,100)
#     data = temp.generate_data()
#     muser.draw_one("sss", "sss",data,  1.444,0.425333,-0.1, 1.1)
#     muser.draw_one("sss", "sss",data,  0.425333,1.444,-0.1, 1.1)
#
#
#     #print muser.line_fun(muser.reflaction(2.1, 3.25), 2.5)
#     #print muser.convert_x(muser.conver_data(2.1, 3.25, np.linspace(0,1,10)))
