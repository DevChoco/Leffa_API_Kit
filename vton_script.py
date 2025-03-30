import os
import numpy as np
from PIL import Image

from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa.transform import LeffaTransform


class LeffaVirtualTryOn:
    def __init__(self, ckpt_dir="ckpts"):
        self.mask_predictor = AutoMasker(
            densepose_path=os.path.join(ckpt_dir, "densepose"),
            schp_path=os.path.join(ckpt_dir, "schp"),
        )
        self.densepose_predictor = DensePosePredictor(
            config_path=os.path.join(ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"),
            weights_path=os.path.join(ckpt_dir, "densepose", "model_final_162be9.pkl"),
        )
        self.parsing = Parsing(
            atr_path=os.path.join(ckpt_dir, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(ckpt_dir, "humanparsing", "parsing_lip.onnx"),
        )
        self.openpose = OpenPose(
            body_model_path=os.path.join(ckpt_dir, "openpose", "body_pose_model.pth"),
        )

        vt_model = LeffaModel(
            pretrained_model_name_or_path=os.path.join(ckpt_dir, "stable-diffusion-inpainting"),
            pretrained_model=os.path.join(ckpt_dir, "virtual_tryon.pth"),
            dtype="float16",
        )
        self.vt_inference = LeffaInference(model=vt_model)
        self.transform = LeffaTransform()

    def run(self,
            person_image_path,
            garment_image_path,
            output_path="output.jpg",
            garment_type="upper_body",
            repaint=False,
            step=20,
            scale=2.5,
            seed=42,
            preprocess_garment=False):
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        src_image = Image.open(person_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)

        if preprocess_garment:
            if garment_image_path.lower().endswith(".png"):
                ref_image = preprocess_garment_image(garment_image_path)
            else:
                raise ValueError("Garment preprocessing only supports PNG files.")
        else:
            ref_image = Image.open(garment_image_path).convert("RGB")
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, garment_type)
        mask = mask.resize((768, 1024))

        seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        densepose = Image.fromarray(seg_array)

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }

        data = self.transform(data)

        output = self.vt_inference(
            data,
            ref_acceleration=False,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=repaint
        )
        gen_image = output["generated_image"][0]
        gen_image.save(output_path)

        return gen_image, mask, densepose
