import os
import numpy as np
import imageio


def load_dv_data(
    scene="cube",
    basedir="/data/deepvoxels",
    testskip=8,
    **kwargs,
):
    def parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
        # Get camera intrinsics
        with open(filepath, "r") as file:
            f, cx, cy = list(map(float, file.readline().split()))[:3]
            grid_barycenter = np.array(list(map(float, file.readline().split())))
            near_plane = float(file.readline())
            scale = float(file.readline())
            height, width = map(float, file.readline().split())

            try:
                world2cam_poses = int(file.readline())
            except ValueError:
                world2cam_poses = None

        if world2cam_poses is None:
            world2cam_poses = False

        world2cam_poses = bool(world2cam_poses)

        print(cx, cy, f, height, width)

        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

        fx = f
        if invert_y:
            fy = -f
        else:
            fy = f

        # Build the intrinsic matrices
        full_intrinsic = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0], [0.0, 0, 1, 0], [0, 0, 0, 1]]
        )

        return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses

    def load_pose(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    H = 512
    W = 512
    deepvoxels_base = "{}/train/{}/".format(basedir, scene)

    (
        full_intrinsic,
        grid_barycenter,
        scale,
        near_plane,
        world2cam_poses,
    ) = parse_intrinsics(os.path.join(deepvoxels_base, "intrinsics.txt"), H)
    print(full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses)
    focal = full_intrinsic[0, 0]
    print(H, W, focal)

    def dir2poses(posedir):
        poses = np.stack(
            [
                load_pose(os.path.join(posedir, f))
                for f in sorted(os.listdir(posedir))
                if f.endswith("txt")
            ],
            0,
        )
        transf = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1.0],
            ]
        )
        poses = poses @ transf
        poses = poses[:, :3, :4].astype(np.float32)
        return poses

    posedir = os.path.join(deepvoxels_base, "pose")
    poses = dir2poses(posedir)
    testposes = dir2poses("{}/test/{}/pose".format(basedir, scene))
    testposes = testposes[::testskip]
    valposes = dir2poses("{}/validation/{}/pose".format(basedir, scene))
    valposes = valposes[::testskip]

    imgfiles = [
        f
        for f in sorted(os.listdir(os.path.join(deepvoxels_base, "rgb")))
        if f.endswith("png")
    ]
    images = np.stack(
        [
            imageio.imread(os.path.join(deepvoxels_base, "rgb", f)) / 255.0
            for f in imgfiles
        ],
        0,
    ).astype(np.float32)

    testimgd = "{}/test/{}/rgb".format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(testimgd)) if f.endswith("png")]
    testimgs = np.stack(
        [
            imageio.imread(os.path.join(testimgd, f)) / 255.0
            for f in imgfiles[::testskip]
        ],
        0,
    ).astype(np.float32)

    valimgd = "{}/validation/{}/rgb".format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(valimgd)) if f.endswith("png")]
    valimgs = np.stack(
        [
            imageio.imread(os.path.join(valimgd, f)) / 255.0
            for f in imgfiles[::testskip]
        ],
        0,
    ).astype(np.float32)

    all_imgs = [images, valimgs, testimgs]
    counts = [0] + [x.shape[0] for x in all_imgs]
    counts = np.cumsum(counts)
    i_split = (np.arange(counts[i], counts[i + 1]) for i in range(3))

    images = np.concatenate(all_imgs, 0)
    poses = np.concatenate([poses, valposes, testposes], 0)

    render_poses = testposes

    print(poses.shape, images.shape)
    hwf = [H, W, focal]
    print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, basedir)

    hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
    near = hemi_R - 1.0
    far = hemi_R + 1.0
    
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    return images, poses, render_poses, hwf, K, (near, far), i_split
