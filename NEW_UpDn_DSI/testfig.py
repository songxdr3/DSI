import json
path = '/opt/data/private/sxd/data/vqa/'
file = 'vqacp_v2_test_annotations.json'
f = path + file
print(f)
with open(file, 'r', encoding='utf-8') as ff:
    data = json.load(ff)

print(data[0])