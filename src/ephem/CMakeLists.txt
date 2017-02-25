# CSRHOS - ephemris subdirectory
# _novas.py _novas.so
# Other example files

#set( CMAKE_C_FLAGS " -O2 -fPIC -flat_namespace -arch x86_64")
#set( CMAKE_MODULE_LINK_FLAGS " -O2 -fPIC -flat_namespace -arch x86_64")

SET(LIBNOVAS_SRC novascon.c readeph0.c solsys1.c nutation.c eph_manager.c slalib.c)

ADD_LIBRARY(novas STATIC novas.c ${LIBNOVAS_SRC})
ADD_EXECUTABLE(example example.c)
TARGET_LINK_LIBRARIES(example novas m) 
ADD_EXECUTABLE(ephemms ephemms.c)
TARGET_LINK_LIBRARIES(ephemms novas m)

#ADD_EXECUTABLE(positionServer positionServer.c)
#TARGET_LINK_LIBRARIES(positionServer novas m czmq)

#ADD_EXECUTABLE(positionClient positionClient.c)
#TARGET_LINK_LIBRARIES(positionClient novas m czmq)

SET_SOURCE_FILES_PROPERTIES(novas.i PROPERTIES C ON)
SET_SOURCE_FILES_PROPERTIES(novas.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(novas python novas.i novas.c novas.h ${LIBNOVAS_SRC})
IF("${CMAKE_SYSTEM}" MATCHES "Darwin")
  SWIG_LINK_LIBRARIES(novas ${PYTHON_LIBRARIES} ${MATH_LIBRARY})
ENDIF()
IF ("${CMAKE_SYSTEM}" MATCHES "Linux")
  SWIG_LINK_LIBRARIES(novas ${PYTONN_LIBRARIES} ${MATH_LIBRARY})
ENDIF()

include_directories (${CMAKE_CURRENT_SOURCE_DIR})
#Message(${CMAKE_BINARY_DIR})

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/novas.py
	      DESTINATION python/pymuser)
#ADD_LIBRARY(_novas MODULE ${LIBNOVAS_SRC})
#set_target_properties(_novas PROPERTIES PREFIX "")

#install(TARGETS novas ARCHIVE DESTINATION python/package)
#install(FILES eph_manager.h  DESTINATION include)

INSTALL (TARGETS _novas
	 LIBRARY DESTINATION python/pymuser
	 ARCHIVE DESTINATION python/pymuser)

#INSTALL(TARGETS positionServer DESTINATION bin)
#INSTALL(TARGETS positionClient DESTINATION bin)
INSTALL(TARGETS ephemms DESTINATION bin)
