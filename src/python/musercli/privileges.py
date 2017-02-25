#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Alea-Soluciones (Bifer) 2007, 2008, 2009
# Licensed under GPL v3 or later, see COPYING for the whole text

'''
  $Id:$
  $URL:$
  Alea-Soluciones (Bifer) (c) 2007
  Created: eferro - 29/7/2007

'''

import musercliutils
import pickle
import os.path
import os
import time
import socket
import re


NONE, ENABLE, CONFIG = 0,1,2

#3333###LOGIN, NORMAL, CONFIGURE, LINE_VTY = 'login','normal', 'configure', 'line_vty'
COMMON, NORMAL, ENABLED, CONFIGURE, EPHEMERIS = -1,0,1,2,3

class InvalidPassword(Exception):
    """Invalid password"""
    pass

class Privileges:
    def __init__(self, min_privileges, mode):
        self.__min_privileges = min_privileges
        self.__mode = mode

class CliFunctionsManager:
    def __init__(self, name, refusedfunctions):
        self.__context = []
        self.__name = name
        self.__refusedfunctions = refusedfunctions
        self.__prompts = {
            0: '>',
            1: '#',
            2: '(config)#',
            3: '(ephemeris)#'
            }
        self.__priv_names = {
            0: 'normal',
            1: 'enable',
            2: 'config',
            3: 'ephemeris',
            }
        self.__config_file = "musercli.conf"
        self.__pass_file = "~/.%s_passwd" %  self.__name
        self.__privileges_file = "~/.%s_privileges"  %  self.__name
        self.__priv = 0
        self.__mode = 0
        self.__valid_modes = [-1,0,1,2,3,4]

        self.__password = 'muser'
        self.__enablepassword = 'muser'

        # try to read password file (if exists)

        self.__functions_access = {}
        self.__functions = {}
        

    def init_privileges(self):
        pass

    def validate_conf(self):
        errors = [f for f in self.__functions_access.keys() if f not in self.__functions.keys()]
        for e in errors:
            musercliutils.Log.info("Access defined for an in existent symbol: %s" % e)
            del self.__functions_access[e]


    def execute(self, function, args):
        str_function = '_'.join(function)
        f = self.__functions[str_function]

        # args contains the list of words typed by the user
        # we should extract the list of parameters values.
        # These values are the words corresponding to the
        # types in the function name (uppercase words)
        parameters = []
        for pos in range(len(function)):
            if function[pos] == function[pos].upper():
                parameters.append(args[pos])
        # finally, call the function with the parameter list
        f(*parameters)

    def get_function(self, func_name):
        str_function = '_'.join(func_name)
        return self.__functions[str_function]

    def validate(self, function):
        try:

            min_priv, mode = self.__functions_access[function]
            # We cosider that a function is accesible if:
            # - We have a privilege level equal or grater
            #   than min priv level of funct
            # - We have configure mode (acces to normal and
            #   configure functs) or we have normal mode and
            #   the funct is defined for normal mode
            if  self.__priv >= min_priv and \
                    (mode == COMMON or self.__mode == mode):
                return True
        except KeyError:
            return False
        return False

    def valid_function(self, function_name):
        
        # At this moment, we only validate if the function
        # is configured to be refused ([refusedfunctions] section
        # at conf file
        for expr in self.__refusedfunctions:
            # The reg expr, can have spaces as word separator, so
            # we remove spaces at the begining and at the end, and
            # replace the internal separators with '_' that is the
            # real separator for functs names
            expr = expr.strip()
            expr = "_".join(expr.split())

            if expr != '' and re.match(expr, function_name):
                musercliutils.Log.debug("'%s' refused function (conf '%s')" % (function_name, expr))
                return False
        return True


    def append(self, function_name, func):        
        # Remove initial word, from functionname (bcli, dynbcli, etc)
        function_name = '_'.join(function_name.split('_')[1:])

        if not self.valid_function(function_name):
            musercliutils.Log.debug("'%s' refused function" % (function_name))
            return

        self.__functions[function_name] = func
        if not self.__functions_access.has_key(function_name):
            self.__functions_access[function_name] = [NONE, COMMON]
    
    def remove(self, function_name):        
        # Remove initial word, from functionname (bcli, dynbcli, etc)
        function_name = '_'.join(function_name.split('_')[1:])
        del self.__functions[function_name]
        del self.__functions_access[function_name]

    def set_function_privileges(self,  min_priv, mode, function):
        # Remove initial word, from functionname (bcli, dynbcli, etc)
        function = '_'.join(function.split('_')[1:])
        self.__functions_access[function] = [min_priv, mode]
        
    def set_privileges(self, priv):
        self.__priv = priv
        return True
    
    def get_privileges(self): return self.__priv
    def set_mode(self, mode):
        self.__mode = mode
        return True

    def get_mode(self): return self.__mode

    def get_functions(self):
        funct = self.__functions_access.keys()
        funct.sort()
        return funct

    def get_function_info(self, function):
        return self.__functions_access[function]

    def get_active_functions(self):
        funct = []
        for f in self.__functions_access.keys():
            if self.validate(f):
                funct.append(f.split('_'))
        funct.sort()

        return funct

    #------------------------------------------------
    def push_context(self, context):
        self.__context.append(context)        

    def pop_context(self):
        try:
            return self.__context.pop()
        except IndexError:
            # Poping from empty stack
            return ""

    def context(self):
        return " ".join(self.__context)
    #------------------------------------------------        


    def context_info(self):
        context = self.context()
        if context == "":
            context = self.__mode

        return "%s" % (
                                 self.__prompts[self.__mode])

        #return "%d [%s] (%s)%s" % (self.__priv,
        #                         self.__priv_names[self.__priv],
        #                         context,
        #                         self.__prompts[self.__priv])


    def change_priv(self, priv):
        self.__change_priv(priv)

    def change_mode(self, mode):
        self.__change_mode(mode)
        
    def __change_mode(self, dest_mode):
        # FIXME refactor __change_priv and __change_mode
        # LOGIN, Enable, Config
        self.set_mode(dest_mode)
        return

    def __change_priv(self, dest_priv):
        # FIXME refactor __change_priv and __change_mode
        #if dest_priv == 0:
        self.set_privileges(dest_priv)
        return

    def get_pass(self):
        return self.__password

    def get_enpass(self):
        return self.__enablepassword

    def set_password(self,pw):
        pw = pw.strip()
        self.__password = pw

    def set_enablepassword(self,pw):
        pw = pw.strip()
        self.__enablepassword = pw

    def has_password(self):
        if self.__password == None:
            return False
        else:
            return True

    def has_enable_password(self):
        if self.__enablepassword == None:
            return False
        else:
            return True

    def check_password(self):
        return self.__get_password()


    def __get_password(self):
        '''Prompt the user for the password and return it'''
        # We import the module only if we really need a password entry
        import getpass
        # FIX-ME
        # We can't use the prompt of the getpass because if we have a output
        # active the prompt appear after the user enter the password
        if self.__password == getpass.getpass('Password: '):
            return True
        else:
            return False

    def check_enable_password(self):
        return self.__get_enable_password()

    def __get_enable_password(self):
        '''Prompt the user for the password and return it'''
        # We import the module only if we really need a password entry
        import getpass
        # FIX-ME
        # We can't use the prompt of the getpass because if we have a output
        # active the prompt appear after the user enter the password
        #print "Password: "
        login = 0
        while login<=2:
            if self.__enablepassword== getpass.getpass('Password:'):
                return True
            else:
                login += 1
        return False

    def change_initial_privilege(self, privilege):
        """Save initial privilege for this user"""
        file = open(self.__get_privilegesfile_name(), "wb")
        pickle.dump(privilege, file)


    def __get_temp_passwords(self):
        """Dummie Algorithm for generate temporal passwords"""
        p = []
        seed1 = int(str(time.time()).split('.')[0])/10000
        seed2 = seed1 - 1
        seed3 = seed1 + 1
        host_seed = socket.gethostname()
        h_seed1 = str(ord(host_seed[0]))
        h_seed2 = str(ord(host_seed[1]))
        p.append(h_seed1 + str(seed1 % 1000) + h_seed2)
        p.append(h_seed1 + str(seed2 % 1000) + h_seed2)
        p.append(h_seed1 + str(seed3 % 1000) + h_seed2)
        return p

