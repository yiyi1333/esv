from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from dataset.dataset_build import build, load

if __name__ == '__main__':
    build(170, 200)
    train_data = load('../dataset', True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    for data in train_loader:
        imgs, targets = data
        imgs = imgs[0]
        print(imgs)
        # imgs 从 tensor 转为 PIL.Image
        pil_image = ToPILImage()(imgs)
        pil_image.show()
        print(targets)
        break
