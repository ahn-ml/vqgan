import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
# import h5pickle as h5py
import h5py

# set gpu device 6
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import glob
import torch
from PIL import Image
import torchvision
from torchvision import transforms

import numpy as np

from torch.utils.data import Dataset

discrete_colors = [  # K, R, B, C, M, Y, W
    [0, 0, 0],
    [255, 0, 0],
    [0, 0, 255],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [255, 255, 255],
]
color_palette = np.array(discrete_colors).astype(np.uint8)


class RNEMBallDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        image_norm_mode="imagenet",
        img_size=64,
        video_len=10,
        output_discrete_pixel_value=False,
        stochastic_sample=False,
    ):

        if os.path.isfile(root):
            assert split in ["training", "validation", "test", "train", "val", "test"]
            self.dataset_attributes = [
                "collisions",
                "events",
                "features",
                "groups",
                "positions",
                "velocities",
            ]
            assert image_norm_mode in ["zero_one"]
            self.dataset_type = "binary"
        elif os.path.isdir(root):
            assert split in ["train", "val", "test"]
            self.dataset_attributes = [
                "colors",
                "ids",
                "imgs",
                "in_camera",
                "layers",
                "masses",
                "positions",
                "present",
                "shapes",
                "sizes",
                "velocities",
            ]
            if not "blinking" in root.lower():
                root = os.path.join(root, f"{split}.hdf5")
            self.dataset_type = "color"
        else:
            raise ValueError("Invalid file_path")

        self.stochastic_sample = stochastic_sample

        self.file_path = root
        self.mode = split
        self.output_discrete_pixel_value = output_discrete_pixel_value
        self.img_size = img_size
        norm_mean = [0.5] * 3
        norm_std = [0.5] * 3

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                # transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        if "blinking" in root.lower():
            file_list = sorted(glob.glob(os.path.join(root, f"{split}*.hdf5")))
            self.full_dataset = []
            self.full_color_id = []
            self.full_shapes = []
            self.full_pos = []
            for file in file_list:
                print(file)
                with h5py.File(file, "r") as f:
                    self.full_dataset.append(f["imgs"][()])
                    self.full_color_id.append(f["color_ids"][()])
                    self.full_shapes.append(f["shapes"][()])
                    self.full_pos.append(f["positions"][()])
            self.full_dataset = np.concatenate(self.full_dataset, 0)
            self.full_color_id = np.concatenate(self.full_color_id, 0)
            self.full_shapes = np.concatenate(self.full_shapes, 0)
            self.full_pos = np.concatenate(self.full_pos, 0)

            self.dataset_size = self.full_dataset.shape[0]
            self.full_episode_length = self.full_dataset.shape[1]
            # print(self.full_episode_length)
            # assert self.full_episode_length == video_len
        else:
            with h5py.File(self.file_path, "r") as f:
                if self.dataset_type == "binary":
                    self.full_dataset = f[split]["features"][()]
                    self.dataset_size = self.full_dataset.shape[1]
                    self.full_episode_length = self.full_dataset.shape[0]
                elif self.dataset_type == "color":
                    full_dataset_size = f["imgs"].shape[0]
                    self.full_dataset = f["imgs"][: full_dataset_size // 2]
                    self.dataset_size = self.full_dataset.shape[0]
                    self.full_episode_length = self.full_dataset.shape[1]

        # self.segments_per_video = self.full_episode_length // self.video_length
        self.set_video_length(video_len)
        if stochastic_sample:
            self.segments_per_video = 1

    def set_video_length(self, video_length):
        self.video_length = video_length
        self.segments_per_video = self.full_episode_length // self.video_length

    def __len__(self):
        return self.dataset_size * (self.segments_per_video)

    def __getitem__(self, idx):

        video_idx = idx // self.segments_per_video
        segment_idx = idx % self.segments_per_video

        if self.dataset_type == "binary":
            video = self.full_dataset[
                segment_idx * self.video_length : (segment_idx + 1) * self.video_length,
                video_idx,
            ]

            example = {}
            frames = []
            if self.stochastic_sample:
                start_idx = torch.randperm(len(video) - self.video_length + 1)[0]
            else:
                start_idx = 0
            for i, frame in enumerate(video):
                image = Image.fromarray(frame[..., 0])
                image = self.transform(image)
                frames.append(image)

            example["pixel_values"] = torch.stack(frames)
        elif self.dataset_type == "color":
            video = self.full_dataset[
                video_idx,
                segment_idx * self.video_length : (segment_idx + 1) * self.video_length,
            ]
            video_expand = video[:, :, :, None]
            frames_discrete = np.argmin(
                np.abs(video_expand - color_palette[None, None, None]).sum(axis=-1),
                axis=-1,
            )
            video = color_palette[frames_discrete]

            x = self.full_color_id[
                video_idx,
                segment_idx * self.video_length : (segment_idx + 1) * self.video_length,
                :,
                0,
            ]
            # shapes = self.full_shapes[video_idx, segment_idx*self.video_length:(segment_idx + 1)*self.video_length]
            # shapes_sorted_idx = np.argsort(shapes, axis=-1)
            color = torch.tensor(x).sum(1).long() - 1
            activated = torch.tensor(x).argmax(1).view(-1, 1, 1).repeat(1, 1, 2)

            x = self.full_pos[
                video_idx,
                segment_idx * self.video_length : (segment_idx + 1) * self.video_length,
            ]

            pos = torch.tensor(x).gather(1, activated).squeeze(-2)
            pos_2 = pos // (self.img_size // 2)
            pos_3 = pos // (self.img_size // 3)
            pos_4 = pos // (self.img_size // 4)
            pos_2 = (pos_2[:, 0] * 2 + pos_2[:, 1]).long()
            pos_3 = (pos_3[:, 0] * 3 + pos_3[:, 1]).long()
            pos_4 = (pos_4[:, 0] * 4 + pos_4[:, 1]).long()

            if self.stochastic_sample:
                start_idx = torch.randperm(len(video) - self.video_length + 1)[0]
            else:
                start_idx = 0
            target = torch.stack([color, pos_2, pos_3, pos_4], -1)[
                start_idx : start_idx + self.video_length
            ]
            # #target = torch.tensor(np.sum([(np.any(x == i,axis=-1))*(2**i) for i in range(7)],axis=0)).long().unsqueeze(-1)
            # target = torch.tensor(x).long()

            example = {}
            frames = []
            for i, frame in enumerate(video):
                image = Image.fromarray(frame)
                image = self.transform(image)
                frames.append(image)

            example["pixel_values"] = torch.stack(frames)[
                start_idx : start_idx + self.video_length
            ]
            if self.output_discrete_pixel_value:
                example["pixel_values_discrete"] = torch.tensor(frames_discrete)[
                    ..., None, :, :
                ][start_idx : start_idx + self.video_length]
            # example = torch.stack(frames)

        if self.mode in ["training", "train"]:
            if not self.output_discrete_pixel_value:
                return {"pixel_values": example["pixel_values"].squeeze()}
            return {
                "pixel_values": example["pixel_values"].squeeze(),
                "pixel_values_discrete": example["pixel_values_discrete"].squeeze(),
            }

        else:
            if not self.output_discrete_pixel_value:
                return (
                    {"pixel_values": example["pixel_values"].squeeze()},
                    target.squeeze(),
                    0,
                )
            return (
                {
                    "pixel_values": example["pixel_values"].squeeze(),
                    "pixel_values_discrete": example["pixel_values_discrete"].squeeze(),
                },
                target.squeeze(),
                0,
            )


class BlinkingTrain(RNEMBallDataset):
    def __init__(
        self,
        size,
        video_len,
        root="/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_6Frames_Blinking_Count",
        output_discrete_pixel_value=False,
        stochastic_sample=False,
    ):
        super().__init__(
            root,
            split="train",
            image_norm_mode="zero_one",
            img_size=size,
            video_len=video_len,
            output_discrete_pixel_value=output_discrete_pixel_value,
            stochastic_sample=stochastic_sample,
        )


class BlinkingTest(RNEMBallDataset):
    def __init__(
        self,
        size,
        video_len,
        root="/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_6Frames_Blinking_Count",
        output_discrete_pixel_value=False,
        stochastic_sample=False,
    ):
        super().__init__(
            root,
            split="val",
            image_norm_mode="zero_one",
            img_size=size,
            video_len=video_len,
            output_discrete_pixel_value=output_discrete_pixel_value,
            stochastic_sample=stochastic_sample,
        )


un_normalize = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / torch.tensor(1)),
        transforms.Normalize(mean=-torch.tensor(0), std=[1.0, 1.0, 1.0]),
    ]
)

if __name__ == "__main__":
    from tqdm import tqdm

    col_dist = [0] * 6
    p_dist1 = [0] * 4
    p_dist2 = [0] * 9
    p_dist3 = [0] * 16
    d = RNEMBallDataset(
        "/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_6Frames_Blinking_Count",
        split="val",
        video_len=5,
    )
    iterator = iter(d)
    for a, t, _ in tqdm(iterator):

        for i in range(5):
            col_dist[t[i, 0]] += 1
            p_dist1[t[i, 1]] += 1
            p_dist2[t[i, 2]] += 1
            p_dist3[t[i, 3]] += 1

    # [6439, 6329, 6363, 6407, 6462, 0]
    # [2736, 3686, 2898, 3649, 4857, 3913, 3039, 4033, 3189, 0, 0, 0, 0, 0, 0, 0]
    print(col_dist, p_dist1, p_dist2, p_dist3)
    # accs = np.zeros((4,),dtype=int)
    # accs_zero = np.zeros((4,),dtype=int)
    # for a, t, _ in tqdm(d):
    #     accs+= (t==np.random.randint(0, 7, size=(1,4))).squeeze().numpy()
    #     accs_zero+= (t==0).squeeze().numpy()
    # print(accs/len(d))
    # print(accs_zero/len(d))

    # print(bins)
