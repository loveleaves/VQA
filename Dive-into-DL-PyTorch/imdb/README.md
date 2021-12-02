# Note

### 版本问题

若运行eat_pytorch_in_20_days代码，建议安装pytorch版本1.5.0

torchtext安装包 与 pytorch对应版本 [github](https://github.com/pytorch/text/)

**torchtext版本**

伴随着 2021年3月5日TorchText 0.9.0更新，一些API调用也发生变化

将from torchtext.data import Field
改为
from torchtext.legacy.data import Field
同理，对于
from torchtext.data import *的其它AttributeError问题，也可以

改为from torchtext.legacy.data import *

**pytorch_lightning版本**

module 'pytorch_lightning' has no attribute 'metrics'

1.5.0及以后去掉了metrics，详见[ChangeLog](https://pytorch-lightning.readthedocs.io/en/stable/generated/CHANGELOG.html)