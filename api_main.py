from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import StreamingResponse
from io import BytesIO
import os
from vton_script import LeffaVirtualTryOn

print("api 실행")

app = FastAPI()

# API 키
API_KEY = "2004"

# 인증 함수
def api_key_auth(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

vton = LeffaVirtualTryOn(ckpt_dir="./ckpts")

@app.post("/virtual-tryon")
async def virtual_tryon(src_image: UploadFile = File(...), ref_image: UploadFile = File(...)):
    
    # API 키 인증
    #api_key_auth(api_key)
    
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    src_img_path = os.path.join(temp_dir, src_image.filename)
    ref_img_path = os.path.join(temp_dir, ref_image.filename)

    with open(src_img_path, "wb") as f:
        f.write(await src_image.read())
    with open(ref_img_path, "wb") as f:
        f.write(await ref_image.read())

    try:
        output_image, mask, densepose = vton.leffa_predict(
            src_image_path=src_img_path,
            ref_image_path=ref_img_path,
            control_type="virtual_tryon",
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",  # 또는 dresses
            vt_repaint=True,
            output_path="output/virtual_tryon_result.jpg"
        )

        # 이미지를 메모리로 읽어 StreamingResponse로 반환
        img_io = BytesIO()
        output_image.save(img_io, format="JPEG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
    
    finally:
        os.remove(src_img_path)
        os.remove(ref_img_path)

