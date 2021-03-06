# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.10
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.



from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_calplot', [dirname(__file__)])
        except ImportError:
            import _calplot
            return _calplot
        if fp is not None:
            try:
                _mod = imp.load_module('_calplot', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _calplot = swig_import_helper()
    del swig_import_helper
else:
    import _calplot
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class calplot(_object):
    """Proxy of C++ casac::calplot class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, calplot, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, calplot, name)
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> calplot"""
        this = _calplot.new_calplot()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _calplot.delete_calplot
    __del__ = lambda self : None;
    def close(self):
        """
        close(self) -> bool

        Summary
        	*** n Close the current calibration table. n ***

        Description
        	
        Close the current calibration table.

        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_close(self)

    def done(self):
        """
        done(self) -> bool

        Summary
        	*** n Close the current calibration table. n ***

        Description
        	
        End the calplot tool.

        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_done(self)

    def next(self):
        """
        next(self) -> bool

        Summary
        	*** n Continue plotting (iteration mode).n ***

        Description
        	
        Continue plotting by stepping through the iteration axes.

        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_next(self)

    def open(self, *args, **kwargs):
        """
        open(self, caltable=string("")) -> bool

        Summary
        	*** n Open a calibration table for use. n
        Supported calibration types: n
           G = gainn
           B = bandpassn
           M = baseline-based gainn
           MF= baseline-based bandpassn
           P = parallactic anglen
           T = tropospheren
           K = baseline-based fringe fittingn *** 


        Description
        	
        Open a calibration table for use.


        Input Parameters:
        	caltable	 Calibration Table name 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_open(self, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        plot(self, xaxis=string("time"), yaxis=string("time")) -> bool

        Summary
        	*** n Plot calibration solutions using user inputs from plotoptions, setdata n
        Options: plottype - AMP, PHASE, REAL, IMAG, SNRn ***

        Description
        	
        Draw plots specified via user inputs. If multiplot is turned on and an iteration axis is selected, the {	t next} method can be used to step through the data.
        specification of the what of the calibration solution to plot is necessary, default is PHASE.
        Supported solution type (AMP, PHASE, RLPHASE, XYPHASE, DELAY, DELAYRATE)


        Input Parameters:
        	xaxis		 Value to plot on the X axis ('time','chan','amp','phase','real','imag','snr') time 
        	yaxis		 Value to plot on the Y axis ('amp','phase','real','imag','snr') time 
        	
        Example:
        	

        # plot some phase calibration solution 
          cp.open('caldata.gcal')
          cp.plot('PHASE')


        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_plot(self, *args, **kwargs)

    def savefig(self, *args, **kwargs):
        """
        savefig(self, filename=string(""), dpi=-1, orientation=string(""), papertype=string(""), facecolor=string(""), 
            edgecolor=string("")) -> bool

        Summary
        	
             Save the currently plotted image.
           

        Description
        	
            Store the contents of the plot window in a file.  The file format (type)
            is based on the file name, ie. the file extension given determines the
            format the file is saved as. The accepted formats are eps,
            ps, png, pdf, and svg.

            Internally, this function uses the matplotlib pl.savefig function.

            Note that if a full path is not given that the files will be saved in
            the current working directory.
           

        Input Parameters:
        	filename	 Name the plot image is to be saved to. 
        	dpi		 Number of dots per inch (resolution) to save the image at. -1 
        	orientation	 Either landscape or portrait. Supported by the postscript format only. 
        	papertype	 Valid values are: letter, legal, exective, ledger, a0-a10 and b0-b10. This option is supported byt the postscript format only. 
        	facecolor	 Color of space between the plot and the edge of the square. Valid values are the same as those accepted by the plotcolor option. 
        	edgecolor	 Color of the outer edge. Valid values are the same as those accepted by the plotcolor option. 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_savefig(self, *args, **kwargs)

    def plotoptions(self, *args, **kwargs):
        """
        plotoptions(self, subplot=111, overplot=False, iteration=string(""), plotrange=initialize_vector(1, (double)0.0), 
            showflags=False, plotsymbol=string("."), plotcolor=string("green"), 
            markersize=5.0, fontsize=10.0) -> bool

        Summary
        	*** n Specify list of plot options. n
        Options: multiplot = true; will use the nxpanels,nypanels setting for displayn***

        Description
        	 
        Specify a list of plot options including number of panels in x and y, iteration axis (if any),
        and if it is a multiplot.


        Input Parameters:
        	subplot		 matplotlib style panel number, e.g 221 means 4 panels 2x2 and plotting on the first panel 111 
        	overplot	 Overplot the next plot. false 
        	iteration	 Iterate plots on antenna, spw, field, and/or time 
        	plotrange	 integer for the plot symbol. 0.0 
        	showflags	 Create multiple pages of plots (per antenna). false 
        	plotsymbol	 The plot symbol to use . 
        	plotcolor	 integer for the plot symbol. green 
        	markersize	 integer for the plot symbol. 5.0 
        	fontsize	 integer for the plot symbol. 10.0 
        	
        Example:
        	

        # create a calplot tool and set the options for subsequent plotting.
          cp := calplot();
          cp.setoptions(nxpanels=3,nypanels=3,iteraxis='antenna',multiplot=T);


        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_plotoptions(self, *args, **kwargs)

    def markflags(self, *args, **kwargs):
        """
        markflags(self, panel=0, region=initialize_vector(1, (double)0.0)) -> bool

        Summary
        	*** n Mark a rectangular region to flag. Click left mouse button, drag and release to mark the region. Multiple regions can be marked. Hit ESC to clearregions (not currently enabled).n *** 

        Description
        	 
        This function is to be called every time a box is to be drawn to flag data


        Input Parameters:
        	panel		 whihc panel to flag on, in case there is more than 1 0 
        	region		 [xmin,ymin,xmax,ymax] bounding box 0.0 
        	
        Example:
        	

        # plot some calibration solution and do some flagging
          cp.open('caldata.gcal')
          cp.plot()
          cp.markflags()
          cp.flagdata()


        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_markflags(self, *args, **kwargs)

    def flagdata(self):
        """
        flagdata(self) -> bool

        Summary
        	*** n Flag Data for selected flag regions. Flags are not writtento disk unless diskwrite=true (or diskwrite=1) is set.n ***

        Description
        	
        Mark the solutions that has been marked by previous calls of markflags as bad in the caltable.

        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_flagdata(self)

    def locatedata(self):
        """
        locatedata(self) -> bool

        Summary
        	*** n Locate and print info about Data for selected flag regions.n ***

        Description
        	
        Display information about  the solutions that has been marked by previous calls of markflags.

        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_locatedata(self)

    def selectcal(self, *args, **kwargs):
        """
        selectcal(self, antenna=initialize_variant(""), field=initialize_variant(""), spw=initialize_variant(""), 
            time=initialize_variant(""), poln=string("")) -> bool

        Summary
        	*** n Select a subset of the data for plotting and specify calibration type n ***

        Description
        	 
        This method enables plotting of a subset of the caltable.  


        Input Parameters:
        	antenna		 Select on antennas 
        	field		 Select on field 
        	spw		 Select on spectral window 
        	time		 Select on time 
        	poln		 Polarization to plot ('', 'RL', 'R', 'L', 'XY', 'X', 'Y', '/') 
        	
        Example:
        	

          cp.open('caltable.gcal');
          cp.setselect(field='1331+305')


        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_selectcal(self, *args, **kwargs)

    def stopiter(self, rmplotter=False):
        """
        stopiter(self, rmplotter=False) -> bool

        Summary
        	*** n Stop plot iterations. n ***

        Description
        	 
        To be called at the end of the plot iterations, or in between if desired.


        Input Parameters:
        	rmplotter	 Indicates of the plot window should be removed (true)from the display or left (false) false 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_stopiter(self, rmplotter)

    def clearplot(self, subplot=000):
        """
        clearplot(self, subplot=000) -> bool

        Summary
        	Clear the plotting window or a particular
           panel. 


        Description
        	 
           Clear the plotting window. Either clear the whole window (default) 
           or a particular panel (specified by the subplot parameter). 
         ***



        Input Parameters:
        	subplot		 Three digits number: first digit for nrows, second for ncols, last for pannel number. 000 
        	
        --------------------------------------------------------------------------------
        	      
        """
        return _calplot.calplot_clearplot(self, subplot)

calplot_swigregister = _calplot.calplot_swigregister
calplot_swigregister(calplot)

# This file is compatible with both classic and new-style classes.


