import torch

# MPS(Metal Performance Shaders) 장치가 사용 가능한지 확인
print(f"PyTorch 버전: {torch.__version__}")
print(f"MPS 사용 가능 여부: {torch.backends.mps.is_available()}")
print(f"MPS 빌드 여부: {torch.backends.mps.is_built()}")

# 간단한 텐서 연산 테스트
if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print("GPU(MPS)에서 텐서 생성 성공!")