# vton_script.py

import numpy as np
from PIL import Image
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import torch

class LeffaVirtualTryOn:
    def __init__(self, ckpt_dir: str):
        self.mask_predictor = AutoMasker(
            densepose_path=f"{ckpt_dir}/densepose",
            schp_path=f"{ckpt_dir}/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path=f"{ckpt_dir}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path=f"{ckpt_dir}/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path=f"{ckpt_dir}/humanparsing/parsing_atr.onnx",
            lip_path=f"{ckpt_dir}/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path=f"{ckpt_dir}/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
            pretrained_model=f"{ckpt_dir}/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
            pretrained_model=f"{ckpt_dir}/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_hd)

    def run(
        self,
        person_image_path: str,
        garment_image_path: str,
        output_path: str = None,
        garment_type: str = "upper_body",
        repaint: bool = False,
        model_type: str = "viton_hd", #"viton_hd","dress_code" 
        preprocess_garment: bool = False,
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42
    ):
        person_img = Image.open(person_image_path)
        person_img = resize_and_center(person_img, 768, 1024)
        person_arr = np.array(person_img)

        if preprocess_garment:
            if not garment_image_path.lower().endswith(".png"):
                raise ValueError("Garment image must be PNG if preprocessing is enabled.")
            garment_img = preprocess_garment_image(garment_image_path)
        else:
            garment_img = Image.open(garment_image_path)
        garment_img = resize_and_center(garment_img, 768, 1024)

        model_parse, _ = self.parsing(person_img.resize((384, 512)))
        keypoints = self.openpose(person_img.resize((384, 512)))

        if model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, garment_type)
        elif model_type == "dress_code":
            mask = get_agnostic_mask_dc(model_parse, keypoints, garment_type)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        mask = mask.resize((768, 1024))

        if model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(person_arr)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(person_arr)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)

        transform = LeffaTransform()
        data = {
            "src_image": [person_img],
            "ref_image": [garment_img],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        inference = self.vt_inference_hd if model_type == "viton_hd" else self.vt_inference_dc
        result = inference(
            data,
            ref_acceleration=False,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=repaint
        )
        gen_image = result["generated_image"][0]

        if output_path:
            gen_image.save(output_path)

        return gen_image, mask, densepose

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        output_path: str = None,
        step=30,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        src_mask_path=None  # 선택적 경로 저장
    ):
        assert control_type in ["virtual_tryon", "pose_transfer"], f"Invalid control type: {control_type}"
        
        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")

        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask 생성
        if control_type == "virtual_tryon":
            # vt_garment_type을 AutoMasker의 mask_type에 매핑
            if vt_garment_type == "dresses":
                garment_type_hd = "overall"  # AutoMasker에서 허용되는 값으로 매핑
            elif vt_garment_type == "upper_body":
                garment_type_hd = "upper"
            elif vt_garment_type == "lower_body":
                garment_type_hd = "lower"
            else:
                raise ValueError(f"Invalid vt_garment_type: {vt_garment_type}")

            mask = self.mask_predictor(src_image, garment_type_hd)["mask"]

            if src_mask_path:
                mask.save(src_mask_path)

        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array, dtype=np.uint8) * 255)

        # DensePose
        if vt_model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(src_image_array)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)

        # Transform 및 inference
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        result = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint
        )

        gen_image = result["generated_image"][0]

        torch.cuda.empty_cache()

        if output_path:
            gen_image.save(output_path)
        
        return gen_image, mask, densepose