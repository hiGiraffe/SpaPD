import torch

x = torch.randn(100, device='cuda:0')
storage = x.untyped_storage()
handle = storage._share_cuda_()
print(type(handle),len(handle))
for _ in range(8):
    print(handle[_])