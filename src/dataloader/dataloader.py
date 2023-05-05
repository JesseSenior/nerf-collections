import numpy as np

from llff import load_llff_data
from blender import load_blender_data
from LINEMOD import load_LINEMOD_data
from deepvoxels import load_dv_data


def load_data(dataset_type, datadir, **kwargs):
    loader = {
        "llff": lambda: load_llff_data(
            datadir,
            kwargs["factor"],
            recenter=True,
            bd_factor=0.75,
            **kwargs,
        ),
        "blender": lambda: load_blender_data(datadir, **kwargs),
        "LINEMOD": lambda: load_LINEMOD_data(datadir, **kwargs),
        "deepvoxels": lambda: load_dv_data(
            scene=kwargs["shape"],
            basedir=kwargs["datadir"],
            testskip=kwargs["testskip"],
            **kwargs,
        ),
    }
    if dataset_type not in loader.keys():
        raise ValueError(f"Unknown dataset type {dataset_type}.")
    (
        images,
        poses,
        render_poses,
        hwf,
        K,
        (near, far),
        (i_train, i_val, i_test),
    ) = loader[dataset_type]()

    if kwargs["render_test"]:
        render_poses = np.array(poses[i_test])

    return images, poses, render_poses, hwf, K, (near, far), (i_train, i_val, i_test)
