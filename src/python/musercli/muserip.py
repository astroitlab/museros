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

# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52304
# Helper class to create static methods (class methods)
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable


class Ip:
    def __init__(self, str_ip, prefix = 24):
        self._ip = str_ip
        self._prefix = prefix
        self._netbits = 32-prefix
        
        self._netmaskbits = ((1<<(self._netbits))-1) ^ 0xFFFFFFFF
        self._netmask =  self.__int_to_ip(self._netmaskbits)
        self._network = self.__int_to_ip((self.__ip_to_int(self._ip) & self._netmaskbits))
        self._broadcast = self.__int_to_ip((self.__ip_to_int(self._ip) | (self._netmaskbits ^ 0xFFFFFFFF)))
    
    
    def to_int(ip):
        r = 0
        try:
            bytes_ip = ip.split('.')
            r = r | long(bytes_ip[0]) << 24
            r = r | long(bytes_ip[1]) << 16
            r = r | long(bytes_ip[2]) << 8
            r = r | long(bytes_ip[3]) << 0
        except:
            raise ValueError
        return r
    
    def to_str(int_ip):
        r = []
        r.append(str((int_ip & 0xFF000000) >> 24))
        r.append(str((int_ip & 0x00FF0000) >> 16))
        r.append(str((int_ip & 0x0000FF00) >>  8))
        r.append(str((int_ip & 0x000000FF) >>  0))
        return ".".join(r)

    def validate_netmask(str_netmask):
        """Return True if the netmask is valid
        """
        bit_activated = False
        data = Ip.to_int(str_netmask)

        # Test the Ip 32 bits
        for i in range(1,33):
            # get the last bit
            bit = ((data & 0xFFFFFFFF) & 0x00000001 )
            if bit_activated and bit == 0:
                return False
            if not bit_activated and bit == 1:
                bit_activated = True
            
            # shift one position
            data = data >> 1
        return True
        
    def num_ips(start, end):
        return Ip.to_int(end) - Ip.to_int(start) 
          
    def ips(start, end): 
        return map(Ip.to_str, range(Ip.to_int(start), Ip.to_int(end) + 1)) 
        
    to_str = Callable(to_str)
    to_int = Callable(to_int)
    num_ips = Callable(num_ips)
    ips = Callable(ips)
    validate_netmask = Callable(validate_netmask)
    
            
    def ip(self):
        return self._ip
    def prefix(self):
        return self._prefix
    def netmask(self):
        return self._netmask
    def network(self):
        return self._network
    def broadcast(self):
        return self._broadcast
    def ip_min(self): 
        return self.__int_to_ip(self.__ip_to_int(self._network) + 1)
    def ip_max(self):
        return self.__int_to_ip(self.__ip_to_int(self._broadcast) - 1)
    def num_ips(self):
        return self.__ip_to_int(self._broadcast) - self.__ip_to_int(self._network) - 1 
        

    def __ip_to_int(self, ip):
        r = 0
        try:
            bytes_ip = ip.split('.')
            r = r | long(bytes_ip[0]) << 24
            r = r | long(bytes_ip[1]) << 16
            r = r | long(bytes_ip[2]) << 8
            r = r | long(bytes_ip[3]) << 0
        except:
            raise ValueError
        return r
    
    def __int_to_ip(self, int_ip):
        r = []
        r.append(str((int_ip & 0xFF000000) >> 24))
        r.append(str((int_ip & 0x00FF0000) >> 16))
        r.append(str((int_ip & 0x0000FF00) >>  8))
        r.append(str((int_ip & 0x000000FF) >>  0))
        return ".".join(r)
    

