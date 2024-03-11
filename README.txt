# 폴더 설명
# Encoder : 이미지 복원 모델 관련 폴더
# Decoder : Segmentation 모델 관련 폴더
# torch_script : torch_script 변환 관련 폴더

# 버전에대한 설명
# Encoder
# Default : 기존의 이미지 복원, Local정보으로 이미지를 복원하기 때문에 정밀도 떨어짐
# v2 : UNet에 약간의 aspp 추가
# v3 : GAN을 활용한 이미지 복원시도, UNet에 좀더많은 aspp 네트워크추가
# train_encoder.py -> Encdoer 이미지 복원 모델
# train_encoder_v3.py -> GAN-Encdoer 이미지 복원 모델

# Decoder
# Default : 기존 Segmentation 작업 수행
# v2 : UNet에 약간의 aspp 추가

# configuration 내 파라미터 값을 조절하여 모델 컨트롤