import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf
from torchvision.transforms import RandomCrop, Pad, Resize, ToTensor


class UIEBTrain(Dataset):
    _INPUT_ = "input"
    _TARGET_ = "target"

    def __init__(self, folder: str, size: int):
        self._size = size
        self.input_dir = os.path.join(folder, self._INPUT_)
        self.target_dir = os.path.join(folder, self._TARGET_)
        self._filenames = sorted(os.listdir(self.input_dir))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        name = self._filenames[item]

        inp = Image.open(os.path.join(self.input_dir, name)).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, name)).convert("RGB")

        inp, tgt = self._aug(inp, tgt)

        return inp, tgt

    def _aug(self, inp, tgt):
        pad_w = max(0, self._size - inp.width)
        pad_h = max(0, self._size - inp.height)

        inp = Pad((0, 0, pad_w, pad_h), padding_mode="reflect")(inp)
        tgt = Pad((0, 0, pad_w, pad_h), padding_mode="reflect")(tgt)

        i, j, h, w = RandomCrop.get_params(inp, (self._size, self._size))
        inp = ttf.crop(inp, i, j, h, w)
        tgt = ttf.crop(tgt, i, j, h, w)

        if random.random() > 0.5:
            inp = ttf.hflip(inp)
            tgt = ttf.hflip(tgt)

        if random.random() > 0.5:
            inp = ttf.vflip(inp)
            tgt = ttf.vflip(tgt)

        k = random.randint(0, 3)
        inp = ttf.rotate(inp, 90 * k)
        tgt = ttf.rotate(tgt, 90 * k)

        return ToTensor()(inp), ToTensor()(tgt)


class UIEBValid(Dataset):
    _INPUT_ = "input"
    _TARGET_ = "target"

    def __init__(self, folder: str, size: int):
        self.input_dir = os.path.join(folder, self._INPUT_)
        self.target_dir = os.path.join(folder, self._TARGET_)
        self._filenames = sorted(os.listdir(self.input_dir))
        self.resize = Resize((size, size))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        name = self._filenames[item]

        inp = Image.open(os.path.join(self.input_dir, name)).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, name)).convert("RGB")

        inp = ToTensor()(self.resize(inp))
        tgt = ToTensor()(self.resize(tgt))

        return inp, tgt
