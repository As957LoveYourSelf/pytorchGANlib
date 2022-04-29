from torchviz import make_dot
from torch.utils.data import Dataset
import torch.cuda
from torchsummary import summary
from PIL import Image
from torchvision.transforms import transforms


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DefinedData(Dataset):
    def __init__(self, x_dir, y_dir, resize=None, type_='img'):
        super(DefinedData, self).__init__()
        self.x_dir = x_dir
        self.label_dir = y_dir
        self.type_ = type_
        self.resize = resize
        self.transformer = transforms.Compose(
            [
                lambda x:Image.open(x).convert('RGB'),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, item):
        pass


class BaseBackend:
    def __init__(self):
        super(BaseBackend, self).__init__()
        self._generator = None
        self._discriminator = None
        self._deep_network = None

    def show_GD_structure(self):
        pass

    def show_G_structure(self):
        pass

    def show_D_structure(self):
        pass

    def summary_GD(self, input_size, batch_size=-1, device='cuda'):
        if self._deep_network is not None:
            summary(self._deep_network,
                    input_size=input_size,
                    batch_size=batch_size,
                    device=device)
        else:
            raise RuntimeError("Deep Network Backend is None!")

    def summary_G(self, input_size, batch_size=-1, device='cuda'):
        if self._generator is not None:
            summary(self._generator,
                    input_size=input_size,
                    batch_size=batch_size,
                    device=device)
        else:
            raise RuntimeError("Generator Network Backend is None!")

    def summary_D(self, input_size, batch_size=-1, device='cuda'):
        if self._discriminator is not None:
            summary(self._discriminator,
                    input_size=input_size,
                    batch_size=batch_size,
                    device=device)
        else:
            raise RuntimeError("Discriminator Network Backend is None!")

    def train(self, x, y, x_dir, y_dir, pretrainmodel=None, save_loss=False):
        if pretrainmodel is not None:
            pass



    def evaluate(self):
        pass
