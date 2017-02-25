# cat: calibration, utility, imaging, manipulation, analysis, simulation, visualization
from odict import odict
mytasks = odict()
tasksum = odict()
taskcat = odict()
taskvis = odict()
tasksum['autoclean'] = 'CLEAN an image with automatically-chosen clean regions.'
taskcat['autoclean'] = 'imaging'
taskvis['autoclean'] = 'hidden'
tasksum['calstat'] = 'Displays statistical information on a calibration table'
taskcat['calstat'] = 'calibration'
taskvis['calstat'] = ''
tasksum['caltabconvert'] = 'Convert old-style caltables into new-style caltables.'
taskcat['caltabconvert'] = 'utility'
taskvis['caltabconvert'] = ''
tasksum['clean'] = 'Invert and deconvolve images with selected algorithm'
taskcat['clean'] = 'imaging'
taskvis['clean'] = ''
tasksum['clean_cl'] = 'Invert and deconvolve images with selected algorithm'
taskcat['clean_cl'] = 'imaging'
taskvis['clean_cl'] = ''

tasksum['ephemeris'] = 'Calculate apparent position of planet'
taskcat['ephemeris'] = 'information'
taskvis['ephemeris'] = ''

tasksum['exportuvfits'] = 'Convert a MUSER observationial data to a UVFITS file'
taskcat['exportuvfits'] = 'import/export'
taskvis['exportuvfits'] = ''
tasksum['exportphase'] = 'Convert a MUSER phase data set to a UVFITS file:'
taskcat['exportphase'] = 'import/export'
taskvis['exportphase'] = ''
tasksum['flagcmd'] = 'Flagging task based on batches of flag-commands'
taskcat['flagcmd'] = 'data editing'
taskvis['flagcmd'] = ''
tasksum['flagdata'] = 'All-purpose flagging task based on data-selections and flagging modes/algorithms.'
taskcat['flagdata'] = 'editing'
taskvis['flagdata'] = ''
tasksum['flagmanager'] = 'Enable list, save, restore, delete and rename flag version files.'
taskcat['flagmanager'] = 'editing'
taskvis['flagmanager'] = ''
tasksum['fluxscale'] = 'Bootstrap the flux density scale from standard calibrators'
taskcat['fluxscale'] = 'calibration'
taskvis['fluxscale'] = ''
tasksum['ft'] = 'Insert a source model  a visibility set:'
taskcat['ft'] = 'imaging, calibration'
taskvis['ft'] = ''
tasksum['gaincal'] = 'Determine temporal gains from calibrator observations'
taskcat['gaincal'] = 'calibration'
taskvis['gaincal'] = ''
tasksum['gencal'] = 'Specify Calibration Values of Various Types'
taskcat['gencal'] = 'calibration'
taskvis['gencal'] = ''

tasksum['headdata'] = 'List, get and data header parameters'
taskcat['headdata'] = 'analysis, information, manipulation'
taskvis['headdata'] = ''

tasksum['immath'] = 'Perform math operations on images'
taskcat['immath'] = 'analysis'
taskvis['immath'] = ''
tasksum['immoments'] = 'Compute moments from an image'
taskcat['immoments'] = 'analysis'
taskvis['immoments'] = ''
tasksum['impbcor'] = 'Construct a primary beam corrected image from an image and a primary beam pattern.'
taskcat['impbcor'] = 'analysis'
taskvis['impbcor'] = ''

tasksum['importfits'] = 'Convert an image FITS file into a CASA image'
taskcat['importfits'] = 'import/export'
taskvis['importfits'] = ''

tasksum['importfitsidi'] = 'Convert a FITS-IDI file to a CASA visibility data set'
taskcat['importfitsidi'] = 'import/export'
taskvis['importfitsidi'] = ''

tasksum['importuvfits'] = 'Convert a UVFITS file to a CASA visibility data set'
taskcat['importuvfits'] = 'import/export'
taskvis['importuvfits'] = ''

tasksum['listcal'] = 'List antenna gain solutions'
taskcat['listcal'] = 'information, calibration'
taskvis['listcal'] = ''

tasksum['listhistory'] = 'List the processing history of a dataset:'
taskcat['listhistory'] = 'information'
taskvis['listhistory'] = ''

tasksum['listfits'] = 'List the HDU and typical data rows of a fits file:'
taskcat['listfits'] = 'information'
taskvis['listfits'] = ''

tasksum['listuvfits'] = 'List the HDU and typical data rows of a uvfits file:'
taskcat['listuvfits'] = 'information'
taskvis['listuvfits'] = ''

tasksum['listdata'] = 'List the summary of a data set in the logger or in a file'
taskcat['listdata'] = 'information'
taskvis['listdata'] = ''

tasksum['listpartition'] = 'List the summary of a multi-MS data set in the logger or in a file'
taskcat['listpartition'] = 'information'
taskvis['listpartition'] = ''
tasksum['listsdm'] = 'Lists observation information present in an SDM directory.'
taskcat['listsdm'] = 'information'
taskvis['listsdm'] = 'experimental'
tasksum['listvis'] = 'List measurement set visibilities.'
taskcat['listvis'] = 'information, analysis'
taskvis['listvis'] = ''

tasksum['plotants'] = 'Plot the antenna distribution in the local reference frame:'
taskcat['plotants'] = 'visualization, calibration'
taskvis['plotants'] = ''
tasksum['plotbandpass'] = 'Makes detailed plots of Tsys and bandpass solutions.'
taskcat['plotbandpass'] = 'visualization, calibration'
taskvis['plotbandpass'] = ''
tasksum['plotcal'] = 'An all-purpose plotter for calibration results '
taskcat['plotcal'] = 'visualization, calibration'
taskvis['plotcal'] = ''
tasksum['plotms'] = 'A plotter/interactive flagger for visibility data.'
taskcat['plotms'] = 'visualization, information,editing, manipulation, utility'
taskvis['plotms'] = ''
tasksum['plotuv'] = 'Plot the baseline distribution'
taskcat['plotuv'] = 'visualization,information'
taskvis['plotuv'] = ''
tasksum['plotweather'] = 'Plot elements of the weather table; estimate opacity.'
taskcat['plotweather'] = 'visualization'
taskvis['plotweather'] = 'experimental'
tasksum['partition'] = 'Task to produce Multi-MSs using parallelism'
taskcat['partition'] = 'manipulation'
taskvis['partition'] = ''
tasksum['polcal'] = 'Determine instrumental polarization calibrations'
taskcat['polcal'] = 'calibration'
taskvis['polcal'] = ''
tasksum['predictcomp'] = 'Make a component list for a known calibrator'
taskcat['predictcomp'] = 'modeling, calibration'
taskvis['predictcomp'] = ''
tasksum['impv'] = 'Construct a position-velocity image by choosing two points in the direction plane.'
taskcat['impv'] = 'analysis'
taskvis['impv'] = ''
tasksum['rmfit'] = 'Calculate rotation measure.'
taskcat['rmfit'] = 'analysis'
taskvis['rmfit'] = ''
tasksum['rmtables'] = 'Remove tables cleanly, use this instead of rm -rf'
taskcat['rmtables'] = 'utility'
taskvis['rmtables'] = ''
tasksum['simobserve'] = 'visibility simulation task'
taskcat['simobserve'] = 'simulation'
taskvis['simobserve'] = ''
tasksum['simanalyze'] = 'image and analyze measurement sets created with simobserve'
taskcat['simanalyze'] = 'simulation'
taskvis['simanalyze'] = ''
tasksum['viewer'] = 'View an image or visibility data set'
taskcat['viewer'] = 'visualization'
taskvis['viewer'] = ''

tasksum['vishead'] = 'List, summary, get, and put metadata in a measurement set'
taskcat['vishead'] = 'information, manipulation'
taskvis['vishead'] = ''
tasksum['visstat'] = 'Displays statistical information from a Measurement Set, or from a Multi-MS'
taskcat['visstat'] = 'information'
taskvis['visstat'] = ''
allcat = {}
thecats =['import/export', 'information', 'editing', 'manipulation', 'calibration', 'modeling', 'imaging', 'analysis', 'visualization', 'simulation', 'utility', 'singledish']
for category in thecats :
   allcat[category] = []
   experimental = []
   deprecated = []
   for key in taskcat.keys() :
      if taskcat[key].count(category) > 0 :
         if taskvis[key].count('hidden') > 0 :
            continue
         else :
            if taskvis[key].count('experimental') > 0 :
               experimental.append('('+key+')')
            else :
               if taskvis[key].count('deprecated') > 0 :
                  deprecated.append('{'+key+'}')
               else :
                  allcat[category].append(key)
   if category == 'utility' :
      allcat[category].append('taskhelp')
      allcat[category].append('tasklist')
      allcat[category].append('toolhelp')
      allcat[category].append('startup')
      allcat[category].append('help par.parametername')
      allcat[category].append('help taskname')
   allcat[category].sort()
   experimental.sort()
   deprecated.sort()
   for key in experimental :
      allcat[category].append(key)
   for key in deprecated :
      allcat[category].append(key)

