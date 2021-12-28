import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import optim
from torchvision.transforms import transforms, InterpolationMode

from internal.normalize_inverse import NormalizeInverse
from internal.params import Params
from internal.vgg19 import Vgg19

params = Params()


def load_img(img_path):
    im = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=params['imagenet_mean'], std=params['imagenet_std']),

    ])
    if params['image_height'] > 0:
        transform.transforms.append(transforms.Resize(params['image_height'], interpolation=InterpolationMode.BICUBIC))
    img = transform(im).unsqueeze(0)
    return img


def total_variation(y):
    return torch.mean(torch.abs(torch.diff(y, dim=-1))) + torch.mean(torch.abs(torch.diff(y, dim=-2)))


def neural_style_transfer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_img(params['content_img_fn']).to(device)
    style_img = load_img(params['style_img_fn']).to(device)

    optimizing_img = torch.autograd.Variable(content_img, requires_grad=True)
    model = Vgg19(requires_grad=False, show_progress=True).to(device).eval()
    target_content_representation, _ = model(content_img)
    _, target_style_representation = model(style_img)

    loss_func = torch.nn.MSELoss(reduction='mean')

    optimizer = optim.LBFGS((optimizing_img,), max_iter=params['n_iterations'], line_search_fn='strong_wolfe')
    itr = 0
    img_denorm = NormalizeInverse(mean=params['imagenet_mean'], std=params['imagenet_std'])

    def closure():
        nonlocal itr
        step_content_representation, step_style_representation = model(optimizing_img)
        content_loss = loss_func(step_content_representation, target_content_representation)
        style_loss = sum(
            [loss_func(x, y) for x, y in zip(step_style_representation, target_style_representation)]) / len(
            target_style_representation)
        tv_loss = total_variation(optimizing_img)
        total_loss = content_loss * params['loss_weights']['content'] + \
                     style_loss * params['loss_weights']['style'] + \
                     tv_loss * params['loss_weights']['tv']
        optimizer.zero_grad()
        total_loss.backward()

        print(
            f'L-BFGS | iteration: {itr:03}, ',
            f'total loss={total_loss.item():12.4f}, ',
            f'content_loss={params["loss_weights"]["content"] * content_loss.item():12.4f}, ',
            f'style loss={params["loss_weights"]["style"] * style_loss.item():12.4f}, ',
            f'tv loss={params["loss_weights"]["tv"] * tv_loss.item():12.4f}')

        if itr % params["display_every"] == 0:
            im_disp = img_denorm(optimizing_img).cpu().detach().numpy().squeeze().transpose(
                [1, 2, 0])
            im_disp = (np.clip(im_disp, 0, 255)).astype(np.uint8)
            plt.imshow(im_disp)
            plt.draw()
            plt.pause(0.1)

        itr += 1
        return total_loss

    optimizer.step(closure)


if __name__ == '__main__':
    params.parse_args()
    neural_style_transfer()
