import torchvision.transforms as transforms
import augly.image as imaugs
import random, os, copy
from PIL import Image, ImageFilter, ImageOps


class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")


class random_pixelization(object):
    def __init__(self, ratio=[0.1, 1]):
        self.ratio = ratio

    def __call__(self, x):
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        x = imaugs.pixelization(x, ratio=ratio)
        return x


class random_shuffle_pixels(object):
    def __init__(self, factor=[0, 0.4]):
        self.factor = factor

    def __call__(self, x):
        factor = random.uniform(self.factor[0], self.factor[1])
        x = imaugs.shuffle_pixels(x, factor=factor)
        return x


class random_perspective_transform(object):
    def __init__(self, sigma=[0, 50], crop=[0, 1]):
        self.sigma = sigma
        self.crop = crop

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        crop = random.randint(self.crop[0], self.crop[1])
        x = imaugs.perspective_transform(x, sigma=sigma, crop_out_black_border=False)
        return x


class random_padding(object):
    def __init__(self, w_factor=[0, 0.1], h_factor=[0, 0.1], color_1=[0, 255], color_2=[0, 255], color_3=[0, 255]):
        self.w_factor = w_factor
        self.h_factor = h_factor
        self.color_1 = color_1
        self.color_2 = color_2
        self.color_3 = color_3

    def __call__(self, x):
        w_factor = random.uniform(self.w_factor[0], self.w_factor[1])
        h_factor = random.uniform(self.h_factor[0], self.h_factor[1])
        color = (random.randint(self.color_1[0], self.color_1[1]),
                 random.randint(self.color_2[0], self.color_2[1]),
                 random.randint(self.color_3[0], self.color_3[1]))
        x = imaugs.pad(x, w_factor=w_factor,
                       h_factor=h_factor,
                       color=color)
        return x


class random_image_underlay(object):
    def __init__(self, path='/media/augustinas/T7/DISC2021/SmallData/images/train/', which=[0, 10000],
                 new_size=[0.04, 0.1], shapx_pos=[0, 0.5],
                 x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.new_size = new_size
        self.path = path
        self.which = which
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.imgs = os.listdir(self.path)

    def __call__(self, x):
        which = random.randint(self.which[0], self.which[1])
        up = Image.open(self.path + self.imgs[which])
        size_x = x.size
        size_up = up.size
        new_size = random.uniform(self.new_size[0], self.new_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        up = up.resize((int(size_up[0] * new_size), int(size_up[1] * new_size)))
        y = copy.deepcopy(x)
        y.paste(up, (int(size_x[0] * x_pos), int(size_x[1] * y_pos)))
        return y


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class random_emoji_overlay(object):
    def __init__(self, opacity=[0, 1], emoji_size=[0, 1], x_pos=[0, 0.5], y_pos=[0, 0.5], p=0.5):
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.p = p

    def __call__(self, x):
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        emoji_size = random.uniform(self.emoji_size[0], self.emoji_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = imaugs.RandomEmojiOverlay(opacity=opacity,
                                      emoji_size=emoji_size,
                                      x_pos=x_pos,
                                      y_pos=y_pos,
                                      p=self.p)(x)
        return x


class random_text_overlay(object):
    def __init__(self, num=2, text=[0, 20], color_1=[0, 255], color_2=[0, 255], color_3=[0, 255], font_size=[0, 1],
                 opacity=[0, 1], x_pos=[0, 0.5], y_pos=[0, 0.5], p=0.5):
        self.num = num
        self.text = text
        self.color_1 = color_1
        self.color_2 = color_2
        self.color_3 = color_3
        self.opacity = opacity
        self.font_size = font_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.p = p

    def __call__(self, x):
        for i in range(self.num):
            text = random.choices(range(100), k=random.randint(self.text[0], self.text[1]))
            color = [random.randint(self.color_1[0], self.color_1[1]),
                     random.randint(self.color_2[0], self.color_2[1]),
                     random.randint(self.color_3[0], self.color_3[1])]
            opacity = random.uniform(self.opacity[0], self.opacity[1])
            font_size = random.uniform(self.font_size[0], self.font_size[1])
            x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
            y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
            x = imaugs.OverlayText(text=text,
                                   font_size=font_size,
                                   opacity=opacity,
                                   color=color,
                                   x_pos=x_pos,
                                   y_pos=y_pos,
                                   p=self.p)(x)
        return x


class random_image_overlay(object):
    def __init__(self, path='/media/augustinas/T7/DISC2021/SmallData/images/train/', which=[0, 10000],
                 new_size=[2.5, 3.5], shapx_pos=[0, 0.5], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.new_size = new_size
        self.path = path
        self.which = which
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.imgs = os.listdir(self.path)

    def __call__(self, x):
        which = random.randint(self.which[0], self.which[1])
        base = Image.open(self.path + self.imgs[which])
        size_x = x.size
        size_base = base.size
        new_size = random.uniform(self.new_size[0], self.new_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = x.resize((int(size_x[0] * new_size), int(size_x[1] * new_size)))
        base.paste(x, (int(size_base[0] * x_pos), int(size_base[1] * y_pos)))
        return base


class random_image_opaque(object):
    def __init__(self, opa=[0.2, 1.0], path='/media/augustinas/T7/DISC2021/SmallData/images/train/', which=[0, 10000]):
        self.opa = opa
        self.path = path
        self.which = which
        self.imgs = os.listdir(self.path)

    def __call__(self, x):
        opa = random.uniform(self.opa[0], self.opa[1])
        which = random.randint(self.which[0], self.which[1])
        base = Image.open(self.path + self.imgs[which])
        x = x.resize((256, 256))
        base = base.resize((256, 256))
        assert base.size == x.size
        x = imaugs.overlay_image(base, x, x_pos=0, y_pos=0, opacity=opa)  # 0.2~1
        return x


class RandomColorize(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = ImageOps.colorize(x.convert("L"),
                              black=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                              white=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                              mid=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                              ).convert("RGB")
        return x


def get_augmentation_chain(image_path: str, mean, std, grayscale: bool = False, add_opaque: bool = False,
                           add_underlay: bool = False, add_blur: bool = False, add_colorize: bool = False,
                           add_dark: bool = False):
    return transforms.Compose(
        [
            ToRGB(),
            transforms.RandomApply([RandomColorize()], p=0.4 if add_colorize else 0),
            transforms.RandomResizedCrop(256, scale=(0.2, 1)),
            transforms.RandomApply([transforms.RandomRotation(50)], p=0.3),
            transforms.RandomApply([random_pixelization()], p=0.3),
            transforms.RandomApply([random_shuffle_pixels()], p=0.3),
            transforms.RandomApply([random_perspective_transform()], p=0.3),
            transforms.RandomApply([random_padding()], p=0.3),
            transforms.RandomApply([random_image_underlay(path=image_path, new_size=[0.2, 0.3])] if add_underlay else [
                random_image_underlay(path=image_path)], p=0.4 if add_underlay else 0.3),
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.3, 0.2)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter((0.1, 0.5), 0, 0, 0)], p=0.5 if add_dark else 0),
            transforms.RandomApply([GaussianBlur([5, 20] if add_blur else [0.1, 2.0])], p=0.3 if add_blur else 1.0),
            transforms.RandomGrayscale(p=0 if grayscale else 0.3),
            transforms.RandomHorizontalFlip(),
            random_emoji_overlay(p=0.3),
            random_text_overlay(p=0.3),
            transforms.RandomApply([random_image_opaque(path=image_path)], p=0.4 if add_opaque else 0),
            transforms.RandomApply([random_image_overlay(path=image_path)], p=0.2),
            transforms.Resize((256, 256)),
            transforms.RandomGrayscale(p=1 if grayscale else 0),
            ToRGB(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
