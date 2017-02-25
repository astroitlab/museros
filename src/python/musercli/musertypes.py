#!/usr/bin/env python

import socket
import re
import muserip
import os
import musercliutils
import glob

global LIBPATH, DATAPATH, SCRIPTSPATH

class MuserShellTypeManager:
    def __init__(self):
        self.types = {
            'STRING': CsrhShellString(),
            'BOOL': CsrhShellBool(),
            'STATE': CsrhShellState(),
            'INT': CsrhShellInt(),
            'FLOAT': CsrhShellFloat(),
            'NET': CsrhShellNet(),
            'NETMASK': CsrhShellNetMask(),
            'NETIF': CsrhShellNetIf(),
            'MAC': CsrhShellMac(),
            'IP': CsrhShellIp(),
            'CIDR': CsrhShellCidr(),
            'HOST': CsrhShellHost(),
            'IPHOST': CsrhShellIpHost(),
            'WEBURL': CsrhShellWebUrl(),
            'FILE': CsrhShellFile(),
            'CMD': CsrhShellCmd(),
            'KERNELVAR': CsrhShellKernelVar(),
            'HEXNUMBER': CsrhShellHexNumber(),
            'DATE':CsrhShellDate(),
            'TIME':CsrhShellTime(),
            'PLANET':CsrhShellPlanet(),
            'RA':CsrhShellRA(),
            'DEC':CsrhShellDEC(),
        }
    
    def add_type(self, type_name, type_class_instance):
        self.types[type_name] = type_class_instance

    def get_types(self):
        return self.types.keys()
        
    def get_type(self, type_name):
        return self.types[type_name]
    
    def set_cli(self, cli):
        for type in self.types.keys():
            self.types[type].set_cli(cli)
            
    def validate_value(self, type, value):
        return self.types[type].validate_value(value)
    def normalize(self, type, value):
        return self.types[type].normalize(value)
    def values(self, type, incomplete_word):
        return self.types[type].values(incomplete_word)
    def is_type(self, word):
        return word in self.types.keys()         
    
class MuserShellType:
    def __init__(self, doc = "Doc to be defined"):
        self.cli = None
        self.doc = doc

    def set_cli(self, cli):
        self.cli = cli
        
    def validate_value(self, value):
        res = value in self.values('')
        return res
    
    def normalize(self, value):
        return value
    
    def values(self, incomplete_word):
        return []

    def help(self):
        return self.doc
    
class CsrhShellString(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Sequence of letters, digits or '_'. Ex: Xema")
    
    def validate_value(self, value):
        return True
    
    def normalize(self, value):
        return value.strip()

class CsrhShellPlanet(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Planet name of solar system. Ex: Sun, Jupiter")

    def validate_value(self, value):
        regexp = r"(MERCURY|VENUS|EARTH|MARS|JUPITER|SATURN|URANUS|NEPTUNE|PLUTO|SUN|MOON)"
        match = re.match(regexp, value.upper())
        if match == None:
            #print "PLANET NAME ERROR"
            return False
        else:
            return match.group().upper() == value.upper()

    def normalize(self, value):
        return value.strip()


class CsrhShellIp(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "IP Address in dot-decimal notation: Ex: 192.168.1.100")
    def validate_value(self, value):
        try:
            if len(value.split('.')) != 4: return False

            mm = socket.inet_aton(value)
            return True # We got through that call without an error, so it is valid
        except socket.error:
            return False # There was an error, so it is invalid

class CsrhShellDate(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Date in Chinese standard. Ex: YYYY-MM-DD")
    def validate_value(self, value):
        #regexp = r"^\d{4}(\-|\/|\.)\d{1,2}\1\d{1,2}$"
        regexp = r"^((((1[6-9]|[2-9]\d)\d{2})(\-|\/|\.)(0?[13578]|1[02])(\-|\/|\.)(0?[1-9]|[12]\d|3[01]))|(((1[6-9]|[2-9]\d)\d{2})(\-|\/|\.)(0?[13456789]|1[012])(\-|\/|\.)(0?[1-9]|[12]\d|30))|(((1[6-9]|[2-9]\d)\d{2})-0?2-(0?[1-9]|1\d|2[0-8]))|(((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00))(\-|\/|\.)0?2(\-|\/|\.)29-))$"
        match = re.match(regexp, value)
        if match == None:
            return False
        else:
            return True

class CsrhShellTime(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Time format in Chinese standard. Ex: HH:MM:SS.SSSS")
    def validate_value(self, value):
        regexp = r"([0-9][0-9]:[0-9][0-9]:[0-9][0-9](\\.[0-9])?(Z|[0-9][0-9][0-9][0-9])?)"
        regexp = r"((([0-9])|([0-1][0-9])|([2][0-3])):(([0-9])|([0-5][0-9])):(([0-9])|([0-5][0-9]))(\.\d+)?)"
        match = re.match(regexp, value.upper())
        if match == None:
            return False
        else:
            return True

class CsrhShellHost(MuserShellType): 
    def __init__(self): 
        MuserShellType.__init__(self, "Hostname. Ex: www.google.com, bif-shp1")
    def validate_value(self, value):
        return True
    
class CsrhShellIpHost(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Hostname or IP Address. Ex: 192.168.10.200, www.google.com")
        self.ip_type = CsrhShellIp()
        self.host_type = CsrhShellHost()
    def validate_value(self, value):
        return self.ip_type.validate_value(value) or \
                self.host_type.validate_value(value)


class CsrhShellWebUrl(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Web URL, Ex: http://www.debian.org or https://www-nasa.gov")

    def validate_value(self, value):
        if value.startswith("http://") or value.startswith("https://"):
            return True
        else:
            return False

    def values(self, incomplete_word):
        if incomplete_word.startswith("http://") or incomplete_word.startswith("https://"):
            return []
        else:
            return ["http://", "https://"]
    
                
class CsrhShellCidr(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self,  
                                "Address in Classless Inter-Domain Routing  notation: Ex: 192.168.1.100/24")
    def validate_value(self, value):
        try:
            ip,bits = value.split('/')
            num_bits = int(bits)
            return True    
        except:
            return False
                
class CsrhShellMac(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "MAC or Ethernet Address in hex notation. Ex: 00:01:02:03:04:08")
    def validate_value(self, value):
        regexp = r"([0-9A-F][0-9A-F]:){5}([0-9A-F][0-9A-F])"
        match = re.match(regexp, value.upper())
        if match == None:
            return False
        else:
            return match.group().upper() == value.upper()


class CsrhShellNet(MuserShellType): 
    def __init__(self): 
        MuserShellType.__init__(self)
    
class CsrhShellNetMask(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Netmask in Dot-decimal Address. Ex: 255.255.255.0")

    def validate_value(self, value):
        return  bosip.Ip.validate_netmask(value)
    

class CsrhShellNetIf(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Ethernet interface. Ex: eth0, eth1, eth0:0")
    def validate_value(self, value):
        return value in self.values('')
        
    def normalize(self, value):
        return value
    
    def values(self, incomplete_word):
        return ['eth0','eth0:0', 'eth1', 'eth1:0', 'eth2','eth2:0', 'br0']

class CsrhShellInt(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Integer value. Ex: 42")
    def validate_value(self, value):
        try:
            int(value)
            return True
        except:
            return False
        
class CsrhShellFloat(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Decimal number")
    def validate_value(self, value):
        try:
            float(value)
            return True
        except:
            return False


class CsrhShellBool(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Coditional value. Accepted values 'true', 'false', '1', '0'")
    def validate_value(self, value):
        return value in self.values('')
        
    def normalize(self, value):
        return value
    
    def values(self, incomplete_word):
        return ['true', 'false', '1', '0']

class CsrhShellState(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "State value. Accepted values 'on', 'off', 'up', 'down'")
    def validate_value(self, value):
        return value in self.values('')
        
    def normalize(self, value):
        return value
    
    def values(self, incomplete_word):
        return ['on', 'off', 'up', 'down']
    
class CsrhShellFile(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Filename with absolute path")
    def validate_value(self, Value):
        # TODO: Implement
        return True
        
    def normalize(self, value):
        return value

    def files(self, incomplete_path):
        return  [f for f in glob.glob(incomplete_path + '*') if f[-1:] not in ['~', '#']]
    
    def values(self, incomplete_word):
        paths = []
        for f in self.files(incomplete_word):
            if os.path.isdir(f):
                paths.append(f + '/')
            else:
                paths.append(f)

        if len(paths) == 1 and os.path.isdir(paths[0]):
            return self.values(paths[0])
        return paths
    

class CsrhShellKernelVar(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Kernel variable. Ex: net.unix.max_dgram_qlen or fs.mqueue.msgsize_max")
        self.kernel_vars=[]
        for line in os.popen("sysctl -A 2>/dev/null").readlines():
            self.kernel_vars.append(line.split('=')[0].strip())
    
    def values(self, incomplete_word):
        return self.kernel_vars
    
class CsrhShellCmd(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Cli available functions")
        
    def values(self, incomplete_word):
        if self.cli == None:
            return []
        else:
            functions = self.cli.get_functions()
            cmds = sorted(list(set([a.split('_')[0] for a in functions])))
            return cmds


 
 
class CsrhShellHexNumber(MuserShellType):
    def __init__(self): 
        MuserShellType.__init__(self, "Hex number value. Ex: FF02FE3A")
    def validate_value(self, value):
        try:
            int(value, 16)
            return True
        except:
            return False
 



class CsrhGenericType(MuserShellType):
    def __init__(self, values_func, short_help = None):
        self.__values_func = values_func
        if short_help != None:
            doc = short_help
        else:
            doc = "Doc to be defined"
            if self.__values_func.__doc__ != None:
                doc = self.__values_func.__doc__
        MuserShellType.__init__(self, doc)
        
    def validate_value(self, value):
        return value in self.values('')

    def values(self, incomplete_word):
        return self.__values_func(incomplete_word)


class CsrhOptionsType(MuserShellType):
    def __init__(self, options_list, doc = None):
        self.__options = options_list
        if doc == None:
            doc = "Options %s" % str(self.__options)

        MuserShellType.__init__(self, doc)
        
    def validate_value(self, value):
        return value in self.values('')

    def values(self, incomplete_word):
        return self.__options


class CsrhShellRA(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Right Ascension format. Ex: HH:MM:SS.SSSS")
    def validate_value(self, value):
        regexp = r"((([0-9])|([0-1][0-9])|([2][0-3])):(([0-9])|([0-5][0-9])):(([0-9])|([0-5][0-9]))(\.\d+)?)"
        match = re.match(regexp, value)
        if match == None:
            return False
        else:
            return True

class CsrhShellDEC(MuserShellType):
    def __init__(self):
        MuserShellType.__init__(self, "Declination format. Ex: DD:MM:SS.SSSS")
    def validate_value(self, value):
        regexp = r"([0-9][0-9]:[0-9][0-9]:[0-9][0-9](\\.[0-9])?(Z|[0-9][0-9][0-9][0-9])?)"
        regexp = r"((([0-9])|([0-8][0-9])):(([0-9])|([0-5][0-9])):(([0-9])|([0-5][0-9]))(\.\d+)?)"
        match = re.match(regexp, value)
        if match == None:
            return False
        else:
            return True


def test1(incomplete_word):
    """Type documentation"""
    print "test1"
    print incomplete_word    
    return ['one', 'two']

if __name__ == '__main__':
    type1 = CsrhGenericType(test1)
    print type1.values('incomp')
    print type1.help()

    type2 = CsrhOptionsType(['Op1', 'Op2', 'Op3'])
    print type2.values('incomp')
    print type2.help()

    type3 = CsrhGenericType(lambda x: ['Op1', 'Op2'])
    print type3.values('incomp')
    print type3.help()

    type4 = CsrhOptionsType(['Op1', 'Op2', 'Op3'], "Options type example")
    print type4.values('incomp')
    print type4.help()
    

    type5 = CsrhGenericType(test1, "with short help")
    print type5.values('incomp')
    print type5.help()

    type6 = CsrhGenericType(lambda incomplete_word: incomplete_word.lower(), "Lower case words")
    print type6.values('inCOMPle')
    print type6.help()
    
