import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        """ """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


from embed import get_embedder
from render import run_network
import config
import os


def create_nerf(**kwargs):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(kwargs["multires"])

    input_ch_views = 0
    embeddirs_fn = None
    if kwargs["use_viewdirs"]:
        embeddirs_fn, input_ch_views = get_embedder(kwargs["multires_views"])
    output_ch = 5 if kwargs["N_importance"] > 0 else 4
    skips = [4]
    model = NeRF(
        D=kwargs["netdepth"],
        W=kwargs["netwidth"],
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=kwargs["use_viewdirs"],
    ).to(config.device)
    grad_vars = list(model.parameters())

    model_fine = None
    if kwargs["N_importance"] > 0:
        model_fine = NeRF(
            D=kwargs["netdepth_fine"],
            W=kwargs["netwidth_fine"],
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=kwargs["use_viewdirs"],
        ).to(config.device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=kwargs["netchunk"],
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=kwargs["lrate"], betas=(0.9, 0.999)
    )

    start = 0
    basedir = kwargs["basedir"]
    expname = kwargs["expname"]

    ##########################

    # Load checkpoints
    if kwargs["ft_path"] is not None and kwargs["ft_path"] != "None":
        ckpts = [kwargs["ft_path"]]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not kwargs["no_reload"]:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": kwargs["perturb"],
        "N_importance": kwargs["N_importance"],
        "network_fine": model_fine,
        "N_samples": kwargs["N_samples"],
        "network_fn": model,
        "use_viewdirs": kwargs["use_viewdirs"],
        "white_bkgd": kwargs["white_bkgd"],
        "raw_noise_std": kwargs["raw_noise_std"],
    }

    # NDC only good for LLFF-style forward facing data
    if kwargs["dataset_type"] != "llff" or kwargs["no_ndc"]:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = kwargs["lindisp"]

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
