# script to read class wise output json file and then combine all classes
# for vxpaste
import os
import json



json_folder_path = '/home/users/u6566739/project-004/AnimateDiff/VXpaste/arxiv_json_tokencut'

json_files = sorted(os.listdir(json_folder_path))

compile = {}
for file in json_files:
    f = open(os.path.join(json_folder_path, file))
    data = json.load(f)
    for key, value in data.items():
        print(f'key: {key}')
        compile[key] = value
        # print(f'data: {file} {data.keys()}')
print(f'compile: {compile.keys()}')

output_file = f"/home/users/u6566739/project-004/AnimateDiff/VXpaste/tokencut_youtube_vis_instance_pools.json"

with open(output_file, 'w') as f:
    json.dump(compile, f)
