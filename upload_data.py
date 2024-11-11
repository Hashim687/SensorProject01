from pymongo.mongo_client import MongoClient
import pandas as pd
import json

#url
uri="mongodb+srv://hashim:Ib5v1DZKGgfaj5Xm@cluster0.8szmc.mongodb.net/?retryWrites=true&w=majority"

#Create a new client connect to server
client=MongoClient(uri)

#create Database, name  and collection name
DATABASE_NAME="sensordata"
COLLECTION_NAME="waferfault"

df=pd.read_csv(r"C:\Users\hashi\OneDrive\Desktop\Sensorproject\notebooks\wafer_23012020_041211.csv")

df.drop(columns=["Unnamed: 0"],axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)