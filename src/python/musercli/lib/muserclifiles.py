#!/usr/bin/env python

import musertypes
import musercliutils
import re
import string
import os
import privileges


def __shell_helper(shellcmdstr, interactive, func_params_names, *args):
    try:
        d = {}
        for pos in range(len(func_params_names)):
            d[func_params_names[pos]] = args[pos]

        s = string.Template(shellcmdstr)
        s = s.substitute(d)
        

        # TODO if s contains ${<var>}, 
        # not all the vars were substituted so
        # raise error
        if interactive:
            musercliutils.InteractiveCommand(s)
        else:
            musercliutils.BatchCommand(s)

    except Exception,ex:
        print "__shell_helper. Error", ex
        print "__shell_helper. Tpe", type(ex)

    


class CommandDefiner():
    def __init__(self, def_dict):
        self.__templates = {}
        # Cmd definition map
        self.__def_dict = def_dict

        self.__templates['shell'] = """
def ${func_name}(*args):
     \"\"\"${doc}
     \"\"\"
     __shell_helper(\"${shell}\", ${interactive}, ${func_params_names}, *args)


get_cli().set_privilege(privileges.${priv},
                        privileges.${mode},
                        '${func_name}')

"""
        self.__normalice_interactive()

        self.__process_cmdstr()

        try:
            s = string.Template(self.__templates['shell'])
            code = s.substitute(self.__def_dict)

            exec(code, globals(), globals())
        except Exception,ex:
            print "ERROR", ex
            print type(ex)


    def __normalice_interactive(self):
        """Transform interactive map entry for the corresponding boolean literal"""
        try:
            interactive_value = self.__def_dict['interactive'].lower()
        except KeyError:
            interactive_value = 'no'
        
        if interactive_value == 'no': 
            self.__def_dict['interactive'] = 'False'
        elif interactive_value == 'yes': 
            self.__def_dict['interactive'] = 'True'        
        else:
            raise ValueError("Value '%s' is not valid. Use yes/no" % interactive_value)



    def __process_cmdstr(self):        
        self.__func_params = []
        self.__func_params_names = []        
        func_name = "ccli"
        pos = 0
        for w in self.__def_dict['cmd'].split():
            if w.find(':') != -1:
                # parameter definition
                pname, ptype = w.split(':')
                ptype = ptype.upper()
                self.__func_params.append([pname, ptype, pos])
                self.__func_params_names.append(pname)
                func_name = func_name + '_' + ptype
            else:
                # keyword
                func_name = func_name + '_' + w
            pos = pos + 1

        self.__func_name = func_name
        self.__def_dict['func_name'] = self.__func_name
        self.__def_dict['func_params'] = self.__func_params
        self.__def_dict['func_params_names'] = self.__func_params_names


class ccliFileParser():
    def __init__(self, filename):
        self.__filename = filename
        self.__need_attribs = ['cmd', 'shell']
        self.__legal_attribs = ['cmd', 'shell', 'doc', 'priv', 'mode']
        self.__def_map = {}        
        
    def __add_cmd_attrib(self, var, value):
        # check already defined var or not legar varname
        if self.__def_map.has_key(var) or var not in self.__legal_attribs:
            raise ValueError("Var '%s' already defined or not a valid name")

        self.__need_attribs = ['cmd', 'shell']
        self.__legal_attribs = ['cmd', 'shell', 'doc', 'priv', 'mode', 'interactive']

        self.__def_map[var] = value

    def __empty_definition(self):
        return self.__def_map == {}



    def __define_command(self):
        # TODO check that we have all the needed attribs

        if not self.__def_map.has_key('doc'):
            self.__def_map['doc'] = "Not documented yet\n"
        if not self.__def_map.has_key('priv'): 
            self.__def_map['priv'] = 'NONE'
        if not self.__def_map.has_key('mode'): 
            self.__def_map['mode'] = 'NORMAL'
        if not self.__def_map.has_key('interactive'):
            self.__def_map['interactive'] = 'no'


        CommandDefiner(self.__def_map)

        # TODO process other attribs
        
        # init temporal definition data
        self.__def_map = {}

    def remove_comments(self, l):
        first_comment = l.find('#')
        if first_comment == 0: return ''
        if first_comment == -1: return l

        l = l.replace('\#', chr(0x00))
        if l.find('#') != -1:
            l = l[:l.find('#')]
        l = l.replace(chr(0x00), '#')
        return l

    def load(self):
        #print "Loading ccli file '%s'" % self.__filename
        try:
            def_found = False
            for l in open(self.__filename).readlines():
                l = l.strip()
                l = self.remove_comments(l)
                if l == '': continue
                if re.match(r'\s*\[\s*\w+\s*\]\s*', l):
                    if def_found:
                        self.__define_command()
                    else:                       
                        def_found = True                        
                else:
                    m = re.search(r'\s*(?P<var>\b\w+\b)\s*=(?P<value>.+)\s*', l)
                    if m:
                        var = m.group('var').strip().lower()
                        value = m.group('value').strip()
                        self.__add_cmd_attrib(var, value)
                    else:
                        print "Invalid musercli file line: '%s'" % (l)
                        raise ValueError("Not valid musercli file line '%s'" % (l))

            # Process the last command definition
            if not self.__empty_definition():
                self.__define_command()

        except IOError, ex:
            msg = "Can't read %s musercli file" % self.__filename
            print msg
            raise IOError("Can't read %s musercli file" % self.__filename)



# Read/Parse all the '.ccli' extensions
for path in get_cli().get_extension_paths():
    for root, dirs, files in os.walk(path):
        for l in [os.path.join(root, f) for f in files if f.endswith('.ccli')]:
            ccliFileParser(l).load()
