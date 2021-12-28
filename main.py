import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim
from torchvision.transforms import transforms, InterpolationMode
from tqdm import trange


from internal.normalize_inverse import NormalizeInverse
from internal.params import Params
from internal.vgg19 import Vgg19

params = Params()


def gram_matrix(x):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram


def load_img(img_path):
    im = Image.open(img_path)
    # im = cv2.imread(img_path)[:, :, ::-1]  # convert BGR to RGB when reading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=params['imagenet_mean'], std=params['imagenet_std']),

    ])
    if params['image_height'] > 0:
        transform.transforms.append(transforms.Resize(params['image_height'],interpolation=InterpolationMode.BICUBIC))
    img = transform(im).unsqueeze(0)
    return img


def total_variation(y):
    '''
    Calculate total variation.
    '''
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
        torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))




def neural_style_transfer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_img(params['content_img_fn']).to(device)
    style_img = load_img(params['style_img_fn']).to(device)

    optimizing_img = torch.autograd.Variable(content_img.clone(), requires_grad=True)
    model = Vgg19(requires_grad=False, show_progress=True).to(device).eval()
    target_content_representation, _ = model(content_img)
    _, target_style_representation = model(style_img)
    target_style_representation = [gram_matrix(x) for x in target_style_representation]

    loss_func = torch.nn.MSELoss(reduction='mean')

    optimizer = optim.LBFGS((optimizing_img,), max_iter=params['n_iterations'], line_search_fn='strong_wolfe')
    itr = 0
    imgDenorm = NormalizeInverse(mean=params['imagenet_mean'], std=params['imagenet_std'])
    def closure():
        nonlocal itr
        content_fv_step, style_fv_step = model(optimizing_img)
        style_fv_step = [gram_matrix(x) for x in style_fv_step]
        content_loss = torch.nn.MSELoss(reduction='mean')(content_fv_step, target_content_representation)
        style_loss = sum([torch.nn.MSELoss(reduction='sum')(x, y) for x, y in zip(style_fv_step, target_style_representation)]) / len(target_style_representation)
        tv_loss = total_variation(optimizing_img)
        total_loss = content_loss * params['loss_weights']['content'] + \
                     style_loss * params['loss_weights']['style'] + \
                     tv_loss * params['loss_weights']['tv']
        optimizer.zero_grad()
        total_loss.backward()

        print(
            f'L-BFGS | iteration: {itr:03}, total loss={total_loss.item():12.4f}, content_loss={params["loss_weights"]["content"] * content_loss.item():12.4f}, style loss={params["loss_weights"]["style"] * style_loss.item():12.4f}, tv loss={params["loss_weights"]["tv"] * tv_loss.item():12.4f}')


        if itr%params["display_every"]==0:
            im_disp = imgDenorm(optimizing_img).cpu().detach().numpy().squeeze().transpose(
                [1, 2, 0])
            im_disp = (np.clip(im_disp, 0, 255)).astype(np.uint8)
            plt.imshow(im_disp)
            plt.draw()
            plt.pause(0.1)

        itr += 1
        return total_loss

    optimizer.step(closure)

    #
    #
    #
    #
    # range_disp = trange(params['n_iterations'])
    # for itr in range_disp:
    #     pass
    #     content_fv_step, style_fv_step = model(optimizing_img)
    #     style_fv_step = [gram_matrix(x) for x in style_fv_step]
    #     content_loss = loss_func(content_fv_step, target_content_representation)
    #     style_loss = sum([loss_func(x, y) for x, y in zip(style_fv_step, target_style_representation)])/len(target_style_representation)
    #     tv_loss = total_variation(optimizing_img)
    #     total_loss = content_loss * params['loss_weights']['content'] + \
    #                  style_loss * params['loss_weights']['style'] + \
    #                  tv_loss * params['loss_weights']['tv']
    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()
    #     range_disp.set_description(f"loss: {total_loss}")
    #     if itr%params["display_every"]==0:
    #
    #         im_disp=torch.concat([content_img,optimizing_img],dim=-1).cpu().detach().numpy().squeeze().transpose([1,2,0])
    #         im_disp = (np.clip(im_disp,0,1)*255).astype(np.uint8)
    #         plt.imshow(im_disp)
    #         plt.draw()
    #         plt.pause(0.1)



if __name__ == '__main__':
    params.parse_args()
    neural_style_transfer()
