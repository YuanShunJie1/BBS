
import torch


# Seed
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(0))
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
else:
    device = torch.device('cpu')
    torch.manual_seed(1)
