from models.metrics import ArcMarginProduct
import torch


Head = ArcMarginProduct(512, 1000, m=0.5).to('cuda')

print("🧪 Forward Pass 테스트:")


# 더미 입력 생성
batch_size = 2
feature_dim = 512
num_classes = 1000

features = torch.randn(batch_size, feature_dim).to('cuda')
labels = torch.randint(0, num_classes, (batch_size,)).to('cuda')

print(f"입력 feature 크기: {features.shape}")
print(f"입력 label 크기: {labels.shape}")

# Forward pass
try:
    with torch.no_grad():
        output = Head(features, labels)
    print(f"✅ 출력 크기: {output.shape}")
    print(f"출력 범위: [{output.min():.4f}, {output.max():.4f}]")
    print("🎉 Forward pass 성공!")
except Exception as e:
    print(f"❌ Forward pass 실패: {e}")