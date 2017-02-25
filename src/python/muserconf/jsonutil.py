import json

class JsonUtil(object):

    @classmethod
    def successMsgJson(cls,str):
        return "{\"success\":true,\"message\":\""+str+"\"}"

    @classmethod
    def errorMsgJson(cls,str):
        return "{\"success\":false,\"message\":\""+str+"\"}"

    @classmethod
    def successObjJson(self,obj):
        return "{\"success\":false,\"object\":"+json.dumps(obj)+"}"