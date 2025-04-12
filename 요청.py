import requests

url = "http://127.0.0.1:8000/virtual-tryon"

files = {
    "src_image": open("upper_img2.jpg", 'rb'),
    "ref_image": open("s0047.jpg", 'rb')
}

# API 키
headers = {"api_key": "2004"}

# 요청 
response = requests.post(url, files=files, headers=headers)

print(response.status_code)

if response.status_code == 200:
    with open("output/virtual_tryon_result.jpg", "wb") as f:
        f.write(response.content)
    print("✅ 결과 이미지 저장 완료!")
else:
    print("❌ 오류 발생:", response.status_code, response.text)
