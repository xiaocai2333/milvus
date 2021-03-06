import random
from locust import User, task, between
from locust_task import MilvusTask
from client import MilvusClient
from milvus import DataType

connection_type = "single"
host = "192.168.1.6"
port = 19530
collection_name = "create_collection_CZkkwJgo"
dim = 128
nb = 50000
m = MilvusClient(host=host, port=port, collection_name=collection_name)
m.clean_db()
m.create_collection(dim, data_type=DataType.FLOAT_VECTOR, auto_id=True, other_fields=None)
vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
entities = m.generate_entities(vectors)


class FlushTask(User):
    wait_time = between(0.001, 0.002)
    if connection_type == "single":
        client = MilvusTask(m=m)
    else:
        client = MilvusTask(host=host, port=port, collection_name=collection_name)

    # def insert(self):
    #     self.client.insert(entities)

    @task(1)
    def flush(self):
        self.client.insert(entities)
        self.client.flush(collection_name)
