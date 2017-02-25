# MUSEROS - src
set(CMAKE_BUILD_TYPE Debug)

# Two subdirectories included

ADD_SUBDIRECTORY(ephem)
#ADD_SUBDIRECTORY(fits)
ADD_SUBDIRECTORY(fusion)
ADD_SUBDIRECTORY(python)
ADD_SUBDIRECTORY(resource)

# All Python files should be copied to DESTINATION/scripts
