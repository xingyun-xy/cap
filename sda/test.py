# import torch
# def replace_state_dict_key(state_dict, old_key='bottom_up', 
#                             new_key='backbone'):
#     new_state_dict = dict()
#     for k, v in state_dict.items():
#         new_k = k.replace(old_key, new_key)
#         new_state_dict[new_k] = v
#     return new_state_dict

# path = "/root/cap-xy/ckpts/official/fcos3d_vovnet_imgbackbone-remapped.pth"
# path = "/root/cap-xy/epoch_11.pth"
# # path = "/root/cap-xy/256_576_nuscenes_r50.pth.tar"

# checkpoint = torch.load(path, map_location='cpu')
# if "state_dict" in checkpoint:
#     state_dict = checkpoint["state_dict"]
# else:
#     state_dict = checkpoint

# state_dict = replace_state_dict_key(state_dict)
# print(state_dict.keys())
# cnt = 0
# for k in state_dict.keys():
#     if k == "backbone.stage5.OSA5_3.ese.fc.weight":
#         cnt += 1
#         print(k, cnt)


# import os
# import os.path as osp
# data_root = "/root/cap-xy/data/cabev/"
# for root, dirs, file in os.walk(data_root):

#     set1 = set(dirs)
#     set2 = set()

#     for dir2 in dirs:
#         if os.path.exists(osp.join(root, dir2, 'annotation')) is True:
#             # print(dir2)
#             set2.add(dir2)
# print("set1: ",  set1)
# print("set2: ",  set2)
# print(set1 - set2)

version_file = 'cap/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    import sys

    # return short version for sdist
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        return locals()['short_version']
    else:
        return locals()['__version__']

print(get_version())