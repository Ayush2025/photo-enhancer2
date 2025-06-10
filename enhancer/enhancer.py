import os
import sys
import types
import torch
from tqdm import tqdm
import cv2
import requests

# -------------------------------------------------------------------
# Monkey-patch basicsr.utils.diffjpeg to avoid import-time numpy errors
# -------------------------------------------------------------------
# Create a dummy module for basicsr.utils.diffjpeg before it's imported
sys.modules['basicsr.utils.diffjpeg'] = types.ModuleType('basicsr.utils.diffjpeg')

# -------------------------------------------------------------------
def download_model_if_not_exists(url: str, dst_path: str):
    """Download a file from `url` to `dst_path` if it doesn't already exist."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.isfile(dst_path):
        print(f"📥 Downloading model from {url} to {dst_path}...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Download complete.")

# -------------------------------------------------------------------
class Enhancer:
    def __init__(self, method='gfpgan', background_enhancement=True, upscale=2):
        # -----------------------------
        # 1. Background enhancement
        # -----------------------------
        if background_enhancement:
            if upscale == 2:
                if not torch.cuda.is_available():  # CPU
                    import warnings
                    warnings.warn(
                        'CPU RealESRGAN is very slow. Skipping background upsampling.'
                    )
                    self.bg_upsampler = None
                else:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                    model = RRDBNet(
                        num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2
                    )
                    self.bg_upsampler = RealESRGANer(
                        scale=2,
                        model_path=(
                            'https://github.com/xinntao/Real-ESRGAN/'
                            'releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                        ),
                        model=model,
                        tile=400,
                        tile_pad=10,
                        pre_pad=0,
                        half=True
                    )
            elif upscale == 4:
                if not torch.cuda.is_available():
                    import warnings
                    warnings.warn(
                        'CPU RealESRGAN is very slow. Skipping background upsampling.'
                    )
                    self.bg_upsampler = None
                else:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                    model = RRDBNet(
                        num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4
                    )
                    self.bg_upsampler = RealESRGANer(
                        scale=4,
                        model_path=(
                            'https://github.com/xinntao/Real-ESRGAN/'
                            'releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                        ),
                        model=model,
                        tile=400,
                        tile_pad=10,
                        pre_pad=0,
                        half=True
                    )
            else:
                raise ValueError(f'Wrong upscale constant {upscale}.')
        else:
            self.bg_upsampler = None

        # -----------------------------
        # 2. Face enhancement settings
        # -----------------------------
        if method == 'gfpgan':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.4'
            self.model_url = (
                'https://drive.google.com/uc?export=download'
                '&id=1Cw1Hx5m4b861xXsrP79-M26Zn6q9tmV7'
            )
        elif method == 'RestoreFormer':
            self.arch = 'RestoreFormer'
            self.channel_multiplier = 2
            self.model_name = 'RestoreFormer'
            self.model_url = (
                'https://github.com/TencentARC/GFPGAN/'
                'releases/download/v1.3.4/RestoreFormer.pth'
            )
        elif method == 'codeformer':
            self.arch = 'CodeFormer'
            self.channel_multiplier = 2
            self.model_name = 'CodeFormer'
            self.model_url = (
                'https://github.com/sczhou/CodeFormer/'
                'releases/download/v0.1.0/codeformer.pth'
            )
        else:
            raise ValueError(f'Wrong model version {method}.')

        # --------------------------------------------
        # 3. Ensure the GFPGAN model weights are present
        # --------------------------------------------
        weights_dir = os.path.join('gfpgan', 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        local_model_path = os.path.join(weights_dir, self.model_name + '.pth')

        download_model_if_not_exists(self.model_url, local_model_path)
        model_path = local_model_path

        # -----------------------------
        # 4. Lazy-import and instantiate GFPGANer
        # -----------------------------
        from gfpgan import GFPGANer
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=self.bg_upsampler
        )

    def check_image_dimensions(self, image):
        height, width, _ = image.shape
        if width > 2048 or height > 2048:
            print("Image dimensions exceed 2048 pixels.")
            return False
        print("Image dimensions are within the limit.")
        return True

    def enhance(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.check_image_dimensions(img):
            _, _, result = self.restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
        else:
            result = img
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
