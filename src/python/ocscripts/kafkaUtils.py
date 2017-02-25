import cPickle
try :
    from kafka import SimpleProducer, KafkaClient, KeyedProducer
except:
    raise Exception("kafka-python is not installed")

class KafkaUtils(object):

    def __init__(self,addr):
        self.kafka_client = KafkaClient(addr)

    def produceTasks(self,tasks):
        producer = KeyedProducer(self.kafka_client)
        for task in tasks:
            producer.send_messages(task.warehouse, task.jobName, cPickle.dumps(task))

    def close(self):
        if self.kafka_client:
            self.kafka_client.close()