import os
import torch
from tqdm import tqdm
import cv2
import gdown
import sys

# Add libraries to path
sys.path.insert(0, './libs/gfpgan')
sys.path.insert(0, './libs/basicsr')

from gfpgan import GFPGANer


class Enhancer:
    def __init__(self, method='gfpgan', background_enhancement=True, upscale=2):
        # -----------------------------
        # 1. Background enhancement setup
        # -----------------------------
        if background_enhancement:
            if upscale == 2:
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
                        num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23,
                        num_grow_ch=32, scale=2
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
                        num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23,
                        num_grow_ch=32, scale=4
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
        # 2. GFPGAN settings
        # -----------------------------
        if method == 'gfpgan':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.4'
            self.drive_id = "1Cw1Hx5m4b861xXsrP79-M26Zn6q9tmV7"
        elif method == 'RestoreFormer':
            self.arch = 'RestoreFormer'
            self.channel_multiplier = 2
            self.model_name = 'RestoreFormer'
            self.drive_id = None
        elif method == 'codeformer':
            self.arch = 'CodeFormer'
            self.channel_multiplier = 2
            self.model_name = 'CodeFormer'
            self.drive_id = None
        else:
            raise ValueError(f'Wrong model version {method}.')

        # ---------------------------------------------------
        # 3. Ensure the GFPGAN model is present locally or via URL
        # ---------------------------------------------------
        weights_dir = os.path.join('libs', 'gfpgan', 'weights')
        os.makedirs(weights_dir, exist_ok=True)

        # default to a local file path
        local_path = os.path.join(weights_dir, f"{self.model_name}.pth")

        if self.drive_id:
            # download GFPGANv1.4 from Drive if missing
            if not os.path.isfile(local_path):
                print(f"Downloading {self.model_name} from Google Drive...")
                url = f"https://drive.google.com/uc?id={self.drive_id}"
                gdown.download(url, local_path, quiet=False)
            model_path = local_path
        else:
            # for RestoreFormer & CodeFormer use official GitHub release URLs
            if self.model_name == 'RestoreFormer':
                model_path = (
                    'https://github.com/TencentARC/GFPGAN/'
                    'releases/download/v1.3.4/RestoreFormer.pth'
                )
            elif self.model_name == 'CodeFormer':
                model_path = (
                    'https://github.com/TencentARC/GFPGAN/'
                    'releases/download/v1.3.4/CodeFormer.pth'
                )
            else:
                model_path = local_path

        # ---------------------------------------------------
        # 4. Lazy-import GFPGANer and create restorer
        # ---------------------------------------------------
        self.restorer = GFPGANer(
            model_path=model_path,
            model_rootpath=weights_dir,
            upscale=upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=self.bg_upsampler
        )

    def check_image_dimensions(self, image):
        h, w, _ = image.shape
        if w > 2048 or h > 2048:
            print("Image dimensions exceed 2048px; skipping face enhancement.")
            return False
        return True

    def enhance(self, image):
        # Convert RGB → BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.check_image_dimensions(bgr):
            _, _, output = self.restorer.enhance(
                bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
        else:
            output = bgr

        # Convert BGR → RGB
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
