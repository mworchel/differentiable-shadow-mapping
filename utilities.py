import matplotlib.pyplot as plt
import torch

from diffshadow import to_display_image

def plot(*args, titles=None, title_args={}, axis_off=False, size=3):
    fig, axs = plt.subplots(1, len(args), figsize=(size*len(args), size), constrained_layout=True)
    for i, arg in enumerate(args):
        if torch.is_tensor(arg):
            axs[i].imshow(to_display_image(arg, grayscale_to_rgb=True))
            if axis_off:
                axs[i].set_axis_off()
        else:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    axs[i].plot(v, label=k)
                    axs[i].legend()
            else:
                axs[i].plot(arg)

    if titles is not None:
        assert len(titles) == len(args)

        for i, title in enumerate(titles): 
            axs[i].set_title(title, **title_args)

    plt.show(fig)