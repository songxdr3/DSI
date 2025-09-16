import pickle
import numpy as np
import h5py
from tqdm import tqdm
import os
import numpy as np

img_path  = '/opt/data/private/sxd/data/trans/img2features'
save_path = '/opt/data/private/sxd/data/trans/trans_features.hdf5'
pkl_path = '/opt/data/private/sxd/data/trans/trans2_id.pkl'

feat = []
name = []
for filename in tqdm(os.listdir(img_path)):
    num = int(filename[15:27])
    file = img_path +'/' + filename
    #print(file)
    saveimg = save_path +'/'+ filename
    name.append(num)
    with np.load(file) as data:
        feat.append(data['x.npy'])

print(feat[1].shape)


# img_dict = {key: index for index, key in enumerate(name)}
#
# with open(pkl_path, 'wb') as pkl_file:
#     # 使用pickle.dump()将字典写入文件
#     pickle.dump(img_dict, pkl_file)
#
# print("writing pkl done!")

with h5py.File(save_path, 'a') as h5file:
    #features_dataset = h5file.create_dataset('image_features', (len(feat),3, 36, 2048), dtype=np.float32)
    features_dataset = h5file['image_features']
    image_id_to_index = {}
    for idx, feature in tqdm(enumerate(feat), total=len(feat)):
        features_dataset[idx,2] = feature








