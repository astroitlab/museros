#!/usr/bin/env python

import musercliutils
import pickle
import os.path
import os
import shutil
import time
import socket
import sys




def get_user_name():
    """Return the real user name based on the environ var"""
    # FIXME needs better implementation
    if os.environ.has_key("USERNAME"):
        return os.environ["USERNAME"]
    elif os.environ.has_key("SUDO_USER"):
        return os.environ["SUDO_USER"]
    else:
        return None

get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_enable_password_STRING')
get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_password_STRING')

get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_observatory_name_STRING')
get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_observatory_longitude_FLOAT')
get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_observatory_latitude_FLOAT')
get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_observatory_altitude_FLOAT')
get_cli().set_privilege(privileges.CONFIG, privileges.CONFIGURE, 'ccli_hostname_STRING')

def ccli_observatory_name_STRING(pw):
    """Modify Astropnomical observatory information
    Observatory name
    The UNENCRYPTED (cleartext) name
    """
    get_cli().set_obsname(pw)

def ccli_observatory_longitude_FLOAT(pw):
    """Modify Astropnomical observatory information
    The altitude of observatory
    The value of latitude
    """
    get_cli().set_obslongitude(pw)

def ccli_observatory_latitude_FLOAT(pw):
    """Modify Astropnomical observatory information
    The latitude of observatory
    The value of latitude
    """
    get_cli().set_obslatitude(pw)

def ccli_observatory_altitude_FLOAT(pw):
    """Modify Astropnomical observatory information
    The altitude of observatory
    The value of altitude
    """
    get_cli().set_obsaltitude(pw)

def ccli_hostname_STRING(pw):
    """Modify host name
    The UNENCRYPTED (cleartext) hostname
    """
    get_cli().set_hostname(pw)
    get_cli().set_prompt()

def ccli_enable_password_STRING(pw):
    """Modify enable password parameters
    Assign the privileged level password
    The UNENCRYPTED (cleartext) 'enable' password
    """
    #print "Changing system login password"
    #csrhcliutils.InteractiveCommand("passwd %s" % get_user_name())
    get_cli().get_cliFunctionsManager().set_enablepassword(pw)


def ccli_password_STRING(pw):
    """Modify password parameters
    The UNENCRYPTED (cleartext) 'enable' password
    """
    #print "Changing system login password"
    get_cli().get_cliFunctionsManager().set_password(pw)


def validate_change_mode(mode):
    """If the is not a password defined return false,
    in other cases, prompt the user for password and return if
    the introduced passwd is correct. If password defined is
    empty don't ask the user for pass"""
    #FIXME
    try:
        return True

    except:
        # If we have any problem allways return that the password
        # is not ok.
        return False


#---------------------------------------------------------------------------------
# Functions for change privileges:
#   enable
#   disable
#   end
#   exit

get_cli().set_privilege(privileges.ENABLE, privileges.ENABLED, 'ccli_disable')
get_cli().set_privilege(privileges.NONE, privileges.NORMAL, 'ccli_enable')
get_cli().set_privilege(privileges.ENABLE, privileges.ENABLED, 'ccli_coordinate')

#-------------------------------------------------------------------------------
# Functions for change modes (NORMAL, CONFIGURE):
#   configure
#   normal
get_cli().set_privilege(privileges.ENABLE, privileges.ENABLED, 'ccli_configure_terminal')
get_cli().set_privilege(privileges.CONFIG, privileges.COMMON, 'ccli_end')
get_cli().set_privilege(privileges.ENABLE, privileges.COMMON, 'ccli_write')
get_cli().set_privilege(privileges.ENABLE, privileges.COMMON, 'ccli_show_runningconfig')

def confinfo():
    info = []

    info.append('!# Csrh command line interface configuration file. Generate automatically.')
    info.append('!# Version 2013.')

    info.append('\n')
    info.append('hostname %s' % get_cli().get_hostname())
    info.append('!')
    if  get_cli().get_cliFunctionsManager().get_enpass() != '':
        info.append('enable password %s' % get_cli().get_cliFunctionsManager().get_enpass())
        info.append('!')
    if  get_cli().get_cliFunctionsManager().get_pass() != '':
        info.append('password %s' % get_cli().get_cliFunctionsManager().get_pass())
        info.append('!')

    info.append('observatory name %s' % get_cli().get_obsname())
    info.append('observatory longitude %s' % get_cli().get_obslongitude())
    info.append('observatory latitude %s' % get_cli().get_obslatitude())
    info.append('observatory altitude %s' % get_cli().get_obsaltitude())
    return info



def ccli_show_runningconfig():
    """Show system or application information.
    Current running configuration file
    """
    for x in confinfo():
        print x
    print
    print


def ccli_write():
    """Write configuration file to disk..
    """
    print 'Building configuration...'
    f = file (get_cli().get_configfile()+'.tmp','w')
    n =0
    for x in confinfo():
        f.write(x+'\n')
        n=n+1
    f.close()
    shutil.copyfile(get_cli().get_configfile()+'.tmp',get_cli().get_configfile())
    os.remove(get_cli().get_configfile()+'.tmp')

    print n,' line(s) saved'
    print '[OK]'


def ccli_ephemeris():
    """Turn on ephemeris mode command.
    """
    get_cli().change_priv(privileges.ENABLE)
    get_cli().change_mode(privileges.EPHEMERIS)


def ccli_enable():
    """Turn on privileged mode command.
    """
    if get_cli().get_cliFunctionsManager().has_enable_password() == True:
        if get_cli().get_cliFunctionsManager().check_enable_password() == True:
            get_cli().change_priv(privileges.ENABLE)
            get_cli().change_mode(privileges.ENABLED)


def ccli_disable():
    """Turn off privileged mode command
    """
    get_cli().change_priv(privileges.NONE)
    get_cli().change_mode(privileges.NORMAL)

# Usefull for users with cisco cli addiction
def ccli_configure_terminal():
    """Change mode to configure mode.
    Configure using terminal/console.
    """
    get_cli().change_priv(privileges.CONFIG)
    get_cli().change_mode(privileges.CONFIGURE)

def ccli_end():
    """Change mode to privileged mode.
    """
    if get_cli().get_mode() >= privileges.CONFIGURE:
        get_cli().change_priv(privileges.ENABLE)
        get_cli().change_mode(privileges.ENABLED)

#-------------------------------------------------------------------------------
def ccli_exit():
    """Exit current mode and down to previous mode.
    """
    if get_cli().get_mode() == privileges.NORMAL or get_cli().get_mode() == privileges.ENABLED:
        get_cli().quit()
    elif get_cli().get_mode() > privileges.CONFIGURE:
        get_cli().change_priv(privileges.ENABLE)
        get_cli().change_mode(privileges.ENABLED)
    elif get_cli().get_mode() == privileges.CONFIGURE:
        get_cli().change_priv(privileges.ENABLE)
        get_cli().change_mode(privileges.ENABLED)
    #else:
    #    get_cli().change_mode(privileges.NORMAL)

def ccli_show_version():
    """Show system or application information.
    System version
    """
    import platform


    print 'Hardware platform information:'
    print ' '

    print 'system   :', platform.system()
    print 'node     :', platform.node()
    print 'release  :', platform.release()
    print 'version  :', platform.version()
    print 'machine  :', platform.machine()
    print 'processor:', platform.processor()

    print ' '
    print 'System information:'

    print 'Normal :', platform.platform()
    print 'Aliased:', platform.platform(aliased=True)
    print 'Terse  :', platform.platform(terse=True)

def ccli_split():
      print "----------------------------------------------------"
