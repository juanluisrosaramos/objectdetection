import requests
import json
import wget


with open("PROJECT3.json", "r") as read_file:
    data = json.load(read_file)



for entry in data:
    print(entry['Labeled Data'])
    filename = wget.download(entry['Labeled Data'])