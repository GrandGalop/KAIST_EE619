import torch

# 모델 정의 (예시로는 4개의 유닛으로 이루어진 fully connected 레이어 사용)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 4)
)

# 예시로 특정 유닛과 관련된 파라미터를 업데이트하고자 하는 경우
target_unit_index = 1

# 예시로 사용할 손실 (tensor([0, 10, 0, 0]))
loss = torch.tensor([0, 10, 0, 0], dtype=torch.float32)
loss = torch.autograd.Variable(loss, requires_grad=True)  # 그래디언트 계산을 위해 Variable로 변환

# 역전파 전에 모든 그래디언트를 0으로 초기화
model.zero_grad()

# 손실을 기반으로 그래디언트 계산
loss.backward(retain_graph=True)

# 특정 유닛과 관련된 파라미터만 업데이트
for name, param in model.named_parameters():
    if 'weight' in name and int(name.split('.')[0]) == target_unit_index:
        param.grad *= 0.0  # 해당 유닛과 관련된 파라미터의 그래디언트를 0으로 설정

# 업데이트된 모델 확인
print(model)
