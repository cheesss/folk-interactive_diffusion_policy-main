from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape   # t: image수 (time step)
            co,ho,wo = shape   # cfg에 정의된 shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            # 이거는 pushT 전용; env_obs에는 3D pose 저장이라서 x,y만 사용
            # if 'pose' in key and shape == (2,):
            #     # take X,Y coordinates
            #     this_data_in = this_data_in[...,[0,1]]
            obs_dict_np[key] = this_data_in
    return obs_dict_np

# obs에서 image의 해상도 출력 (width, height)
def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
