#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Alea-Soluciones (Csrh) 2007, 2008, 2009
# Licensed under GPL v3 or later, see COPYING for the whole text

'''
  $Id:$
  $URL:$
  Alea-Soluciones (Csrh) (c) 2007
  Created: eferro - 31/7/2007
'''


import musercliutils


def get_cmds():
    '''Return the command suported in the system. We consider a command (or
    a command family) the first word of a function'''
    functions = get_cli().get_functions()
    return sorted(list(set([a[0] for a in functions])))

def get_active_cmds():
    '''Return the command enable in the actual mode'''
    functions = get_cli().get_active_functions()
    return sorted(list(set([a[0] for a in functions])))

def get_functions(cmd):
    '''Return all the enabled functions that correspond to the comand indicated'''
    matchs = [a for a in get_cli().get_active_functions() if a[0] == cmd]
    return matchs


def ccli_help():
    """Commands for interactive help.
    """    
    #get_cli().basic_help()
    he =  """
    Help may be requested at any point in a command by entering
    a question mark '?'.  If nothing matches, the help list will
    be empty and you must backup until entering a '?' shows the
    available options.
    Two styles of help are provided:
    1. Full help is available when you are ready to enter a
       command argument (e.g. 'show ?') and describes each possible
       argument.
    2. Partial help is provided when an abbreviated argument is entered
       and you want to know what arguments match the input
       (e.g. 'show pr?'.)
    """
    print he



def ccli_help_all():
    """Show basic help for all available commands.
    All commands
    """
    cmd_list = []
    for cmd in get_cli().get_active_functions():
        str = ''
        str = str + get_cli().get_normalize_func_name(cmd)
        str = str + ' : ' + get_function_help(cmd).split('\n')[0]
        cmd_list.append(str)    
    cmd_list.sort()
    for cmd in cmd_list: print cmd
    


def get_function_name(function):
    '''Return the real simbol name of a function from the function sintax in the bcli internal format'''
    return 'ccli_' + ('_'.join(function))

def get_function_help(function):
    '''Return all the doc of a function.
If the function have no doc, it returns "Not documented yet"
'''
    f = get_cli().get_function(function)
    doc = f.__doc__
    if doc == None:
        doc = "Not documented yet\n"
    return doc    

def get_basic_help(cmd):
    str = ""
    for func in get_functions(cmd):
        str = str + get_cli().get_normalize_func_name(func)
        str = str + " => " + get_function_help(func).split('\n')[0] + '\n'
    return str

def get_extended_help(cmd):
    str = ""
    for func in get_functions(cmd):
        str = str + get_cli().get_normalize_func_name(func)
        str = str + " => " + get_function_help(func) + '\n'
    return str
