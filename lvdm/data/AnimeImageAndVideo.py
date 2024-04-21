import os
import random
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from PIL import Image
import glob
import pdb

class AnimeImageAndVideo(Dataset):
    """
    Anime Dataset.
    Assumes data is structured as follows.
    Anime/
        image/
            train/
                xxx.jpg
                ...
            test/
                xxx.jpg
                ...
    """
    def __init__(self,
                 data_root,
                 resolution,
                 video_length,
                 subset_split,
                 frame_stride,
                 data_type='video', # video/image
                 ):
        self.data_root = data_root
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.data_type = data_type
        assert(self.subset_split in ['train', 'test', 'all'])
        assert(self.data_type in ['video', 'image'])

        self.exts = ['avi', 'mp4', 'webm']
        self.exts_img = ['jpg', 'png'] # and other type

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        assert(isinstance(self.resolution, list) and len(self.resolution) == 2)

        self._make_dataset()
    
    def _make_dataset(self):
        if self.subset_split == 'all':
            data_folder = self.data_root
        else:
            data_folder = os.path.join(self.data_root, self.subset_split)
        if self.data_type == 'video':
            self.videos = sum([glob.glob(os.path.join(data_folder, '**', f'*.{ext}'), recursive=True)
                        for ext in self.exts], [])
            print(f'Number of videos = {len(self.videos)}')
        elif self.data_type == 'image':
            self.images = sum([glob.glob(os.path.join(data_folder, '**', f'*.{ext}'), recursive=True)
                        for ext in self.exts_img], [])
            print(f'Number of images = {len(self.images)}')

    def __getitem__(self, index):
        if self.data_type == 'video':
            while True:
                video_path = self.videos[index]

                try:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                    if len(video_reader) < self.video_length:
                        index += 1
                        continue
                    else:
                        break
                except:
                    index += 1
                    print(f"Load video failed! path = {video_path}")
        
            all_frames = list(range(0, len(video_reader), self.frame_stride))
            if len(all_frames) < self.video_length:
                all_frames = list(range(0, len(video_reader), 1))

            # select random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = list(range(rand_idx, rand_idx+self.video_length))
            frames = video_reader.get_batch(frame_indices)
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'

            num_frames = frames.shape[0]
            random_frame_index = random.randint(0, num_frames - 1)
            random_frame = frames.asnumpy()[random_frame_index].copy()

            # test
            print(f'random_frame.shape: {random_frame.shape}')

            frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
            assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
            frames = (frames / 255 - 0.5) * 2
            data = {'video': frames, 'caption': 'test', 'random_frame': random_frame}
            return data
        elif self.data_type == 'image':
            while True:
                img_path = self.images[index]
                image = Image.open(img_path).convert('RGB')
                try:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                    if len(video_reader) < self.video_length:
                        index += 1
                        continue
                    else:
                        break
                except:
                    index += 1
                    print(f"Load video failed! path = {video_path}")
        
            all_frames = list(range(0, len(video_reader), self.frame_stride))
            if len(all_frames) < self.video_length:
                all_frames = list(range(0, len(video_reader), 1))

            # select random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = list(range(rand_idx, rand_idx+self.video_length))
            frames = video_reader.get_batch(frame_indices)
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'

            num_frames = frames.shape[0]
            random_frame_index = random.randint(0, num_frames - 1)
            random_frame = frames.asnumpy()[random_frame_index].copy()

            # test
            print(f'random_frame.shape: {random_frame.shape}')

            frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
            assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
            frames = (frames / 255 - 0.5) * 2
            data = {'video': frames, 'caption': 'test', 'random_frame': random_frame}
            return data
    
    def __len__(self):
        if self.data_type == 'video':
            return len(self.videos)
        elif self.data_type=='image':
            return len(self.images)