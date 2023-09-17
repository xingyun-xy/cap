import torch
def replace_state_dict_key(state_dict, old_key='img_backbone', 
                            new_key='backbone'):
    new_state_dict = dict()
    for k, v in state_dict.items():
        new_k = k.replace(old_key, new_key)
        new_state_dict[new_k] = v
    return new_state_dict

path = "/root/cap-xy/fcos3d_vovnet_imgbackbone-remapped.pth"
path = "/root/cap-xy/256_576_nuscenes_r50.pth.tar"

checkpoint = torch.load(path, map_location='cpu')
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# state_dict = replace_state_dict_key(state_dict)
print(state_dict.keys())