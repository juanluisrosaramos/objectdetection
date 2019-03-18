import json
def func():
    with open('PROJECT3.json', 'r') as json_file:
        json_data = json.load(json_file)
        print(json_data)
func()