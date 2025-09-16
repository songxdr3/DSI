path = '/opt/data/private/sxd/data/trans/trans_features.hdf5'
#path = '/opt/data/private/sxd/data/detection_features/train36.hdf5'
import h5py
with h5py.File(path, 'r') as file:
    for key in file.keys():
        print(key)

    print(file)
    print((file['image_features']).shape)
    print(file['image_features'][82782][2])

# import pickle
# path = '/opt/data/private/sxd/data/detection_features/train36_imgid2img.pkl'
# # 打开.pkl文件
# with open(path, 'rb') as file:
#     # 加载.pkl文件中的内容
#     data = pickle.load(file, encoding="latin1")
#
# # 现在可以使用data变量中的数据了
# print(len(data))
# #print(sorted(data.values()))