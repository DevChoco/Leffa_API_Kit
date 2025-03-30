# Leffa_API_Kit
> Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation

## Acknowledgement
Created by modifying [LEFFA](https://github.com/franciszzj/LEFFA).

# 설치전 작업
1. wsl 우분투 설치
2. 우분투 아나콘다 설치 - 파이썬 3.10
3. 우분투 아나콘다 주피터 노트북 설치


# 설치
1. clone Leffa (우분투 아나콘다 콘솔)
```
git clone https://github.com/franciszzj/Leffa.git
```
```
cd Leffa
```
2. 추가 코드 clone
```
git clone https://github.com/DevChoco/Leffa_API_Kit.git
```
3. 라이브러리 설치
```
pip install -r requirements.txt
```
4. 모델설치
```
python app.py
```
- 설치바가 완전히 찬 후, 웹ui에 들어가기전 램 용량이 24Gb 이하이면 콘솔창에 Killed 라는 메시지가 나올거임.
- Killed가 나와도 우리는 app.py를 사용하지 않을거라서 상관없음.
- 안나오더라도 app.py 동작후 `Ctrl + C`로 종료.
# 실행
1. 주피터 노트북 집입 후 `leffa_test.ipynb` 열기
2. 7번칸 실행
```
3rdparty   app.py      in_img				 leffa_utils
LICENSE    ckpts       leffa				 preprocess
README.md  densepose   leffa_test.ipynb			 requirements.txt
SCHP	   detectron2  leffa_test.ipynb:Zone.Identifier  vton_script.py
```
실행후 이렇게 나와야 정상
```
!cd ..

or

%cd leffa
```
위의 코드 활용하여 `leffa`폴더로 이동
3. 8번 코드를 실행하여 합성실행
```
gen_img, mask_img, dense_img = vton.run(
    person_image_path="in_img/00034_00.jpg",
    garment_image_path="in_img/09133_00.jpg",
    output_path="output/virtual_tryon_result.jpg",
    garment_type="upper_body",
    repaint=False
)
```
- person_image_path : 사람 이미지 경로
- garment_image_path : 옷 이미지 경로
- garment_type : `upper_body` or `lower_body` or `dress`
