from torchviz import make_dot
from torch.utils.data import Dataset
import torch.cuda
from torchsummary import summary
from PIL import Image
from torchvision.transforms import transforms
import os


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DefinedData(Dataset):
    def __init__(self, x_dir=None, y_dir=None, resize=None, type_='img'):
        super(DefinedData, self).__init__()
        self.x_dir = os.listdir(x_dir) if x_dir is not None else None
        self.label_dir = os.listdir(y_dir) if y_dir is not None else None
        self.type_ = type_
        self.resize = resize
        self.transformer = transforms.Compose(
            [
                lambda x: Image.open(x).convert('RGB'),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __getitem__(self, item):
        if self.label_dir is None:
            return self.transformer(self.x_dir[item]), None
        elif self.x_dir is None:
            return None, self.transformer(self.label_dir[item])
        else:
            return self.transformer(self.x_dir[item]), self.transformer(self.label_dir[item])

    def __len__(self):
        return len(self.x_dir)


class BaseBackend:
    def __init__(self):
        super(BaseBackend, self).__init__()
        self._generator = None
        self._discriminator = None
        self._deep_network = None

    def show_GD_structure(self, input_dim=(1, 3, 28, 28), save_dir="./"):
        viz = make_dot(self._deep_network(torch.rand(size=input_dim)),
                       params=dict(self._deep_network.named_parameters()))
        viz.directory = save_dir
        viz.view()

    def show_G_structure(self, input_dim=(1, 3, 28, 28), save_dir="./"):
        viz = make_dot(self._generator(torch.rand(size=input_dim)),
                       params=dict(self._generator.named_parameters()))
        viz.directory = save_dir
        viz.view()

    def show_D_structure(self, input_dim=(1, 3, 28, 28), save_dir="./"):
        viz = make_dot(self._discriminator(torch.rand(size=input_dim)),
                       params=dict(self._discriminator.named_parameters()))
        viz.directory = save_dir
        viz.view()

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

    def evaluate(self, x, y):
        # evaluate model by test data set
        pass
