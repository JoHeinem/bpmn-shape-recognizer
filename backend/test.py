import json

dict = {'foo': 'bar', 'test': { 'innerfield': 'foo'}}
print(json.dumps(dict))