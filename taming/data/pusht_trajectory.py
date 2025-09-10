import os
import sys
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Optional
import cv2
import h5py



class PushTDataset(Dataset):
    """
    Dataset for PushT trajectory data stored in separate files.
    
    Compatible with Hydra instantiation system used in discrete-jepa.
    Now uses the same __init__ arguments as HanoiTrajectoryDataset for consistency.
    
    Your data structure:
    root/
    ├── train/
    │   ├── abs_actions.pth       # Absolute actions
    │   ├── rel_actions.pth       # Relative actions  
    │   ├── states.pth           # State information
    │   ├── seq_lengths.pkl      # Sequence lengths
    │   └── obses/              # Observation videos
    │       ├── episode_000.mp4
    │       └── ...
    └── val/
        └── ...
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 64,
        sequence_length: int = 16,
        stride: int = 1,
        transform: Optional[transforms.Compose] = None,
        action_dim: int = 2,  # PushT has 2D continuous actions
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
        max_trajectories: Optional[int] = None,
        num_train_images: Optional[int] = None,  # For compatibility with existing datasets
        # PushT-specific parameters (passed via kwargs)
        image_norm_mode: str = "imagenet",
        **kwargs
    ):
        """
        Args:
            root: Path to directory containing train/val/test subdirectories
            split: Data split ("train", "val", "test")
            img_size: Target image size (default: 64)
            sequence_length: Number of observations to sample per sequence (default: 10)
            stride: Stride between sequences (default: 1)
            transform: Optional image transforms
            action_dim: Number of action dimensions (2 for PushT: x, y)
            train_split: Fraction of trajectories for training (not used in current implementation)
            val_split: Fraction of trajectories for validation (not used in current implementation)
            seed: Random seed for splitting (not used in current implementation)
            max_trajectories: Maximum trajectories to load (for debugging)
            num_train_images: Compatibility parameter (not used)
            image_norm_mode: Image normalization mode ("imagenet", "centered", "zero_one")
            **kwargs: PushT-specific parameters:
                - output_dict: Whether to return dictionary output (default: False)
                - stochastic_sample: Whether to use stochastic sampling (default: False)
                - squeeze_time: Whether to squeeze time dimension for single frames (default: True)
                - position_bins: Number of bins for position discretization (default: 4)
                - angle_bins: Number of bins for angle discretization (default: 4)
        """
        

        self.root = root
        self.split = split
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.action_dim = action_dim
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.max_trajectories = max_trajectories
        
        # PushT-specific parameters
        self.output_dict = kwargs.get("output_dict", False)
        self.video_len = sequence_length  # For backward compatibility
        self.stochastic_sample = kwargs.get("stochastic_sample", False)  # Changed default to False
        self.squeeze_time = kwargs.get("squeeze_time", True)
        self.position_bins = kwargs.get("position_bins", 4)
        self.angle_bins = kwargs.get("angle_bins", 4)

        # Handle transform parameter
        if transform is None:
            # Image normalization
            match image_norm_mode:
                case "imagenet":
                    norm_mean = [0.485, 0.456, 0.406]
                    norm_std = [0.229, 0.224, 0.225]
                case "centered":
                    norm_mean = [0.5, 0.5, 0.5]
                    norm_std = [0.5, 0.5, 0.5]
                case "zero_one":
                    norm_mean = [0.0, 0.0, 0.0]
                    norm_std = [1.0, 1.0, 1.0]
                case _:
                    raise ValueError(f"Invalid image_norm_mode: {image_norm_mode}")

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
        else:
            self.transform = transform
        
        # Load split-specific data
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.abs_actions = torch.load(os.path.join(split_dir, "abs_actions.pth"))
        self.rel_actions = torch.load(os.path.join(split_dir, "rel_actions.pth"))
        self.states = torch.load(os.path.join(split_dir, "states.pth"))
        
        with open(os.path.join(split_dir, "seq_lengths.pkl"), 'rb') as f:
            self.seq_lengths = pickle.load(f)
        
        # Check for HDF5 files first, then fall back to MP4
        h5_dir = os.path.join(split_dir, "obses_h5")
        obs_dir = os.path.join(split_dir, "obses")
        
        if os.path.exists(h5_dir):
            # Use HDF5 files
            self.use_h5 = True
            self.use_video = False
            self.h5_dir = h5_dir
            
            # Load episode index
            index_path = os.path.join(h5_dir, "episode_index.pkl")
            if os.path.exists(index_path):
                with open(index_path, 'rb') as f:
                    self.episode_to_h5 = pickle.load(f)
                print(f"Loaded HDF5 episode index: {len(self.episode_to_h5)} episodes")
            else:
                raise FileNotFoundError(f"Episode index not found: {index_path}")
            
        elif os.path.exists(obs_dir):
            # Use MP4 files
            self.use_h5 = False
            self.obs_files = sorted([f for f in os.listdir(obs_dir) if f.endswith((".mp4", ".pth"))])
            self.use_video = len(self.obs_files) > 0 and self.obs_files[0].endswith(".mp4")
            
        else:
            # No observations available
            self.use_h5 = False
            self.obs_files = []
            self.use_video = False
        
        self.num_episodes = len(self.seq_lengths)
        
        # Create stride-based sequences (similar to HanoiTrajectoryDataset)
        self._create_sequences()
        
        print(f"PushTNoise Dataset - Split: {split}")
        print(f"#episodes={self.num_episodes}, use_h5={self.use_h5}, use_video={self.use_video}, video_len={self.video_len}, stride={self.stride}")
        print(f"Total sequences: {len(self.episode_sequences)}")

    def _create_sequences(self):
        """Create stride-based sequences similar to HanoiTrajectoryDataset."""
        self.episode_sequences = []
        
        print(f"Creating sequences with sequence_length={self.sequence_length}, stride={self.stride}")
        
        for episode_idx, seq_len in enumerate(self.seq_lengths):
            # Check if we have enough frames for the requested sequence_length
            if seq_len >= self.sequence_length:
                # Calculate the number of possible sequences
                max_start_idx = seq_len - self.sequence_length
                
                # Create sequences with proper stride
                # stride=1: dense sampling (every possible sequence)  
                # stride>1: sparse sampling (non-overlapping or reduced overlap)
                for start_idx in range(0, max_start_idx + 1, self.stride):
                    self.episode_sequences.append({
                        'episode_idx': episode_idx,
                        'start_idx': start_idx,
                        'seq_length': seq_len
                    })
            else:
                # Skip episodes that are too short
                print(f"Warning: Episode {episode_idx} has {seq_len} frames, "
                      f"but sequence_length={self.sequence_length} requested. Skipping.")
                continue
        
        total_sequences = len(self.episode_sequences)
        print(f"Created {total_sequences} sequences from {self.num_episodes} episodes")
        
        if total_sequences == 0:
            print(f"ERROR: No sequences created!")
            print(f"  Split: {self.split}")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Stride: {self.stride}")
            print(f"  Available episodes: {self.num_episodes}")
            if len(self.seq_lengths) > 0:
                print(f"  Sample episode lengths: {self.seq_lengths[:5]}")

    def __len__(self):
        return len(self.episode_sequences)
    
    def get_action_dim(self) -> int:
        """Get the number of action dimensions."""
        return self.action_dim

    def _load_video_frames(self, video_path, start_frame, num_frames):
        """Load frames from MP4 video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _load_h5_frames(self, episode_idx, start_frame, num_frames):
        """Load frames from HDF5 file"""
        if episode_idx not in self.episode_to_h5:
            raise ValueError(f"Episode {episode_idx} not found in HDF5 index")
        
        h5_filename = self.episode_to_h5[episode_idx]
        h5_path = os.path.join(self.h5_dir, h5_filename)
        
        with h5py.File(h5_path, 'r') as h5f:
            episode_key = f"episode_{episode_idx:06d}"
            if episode_key not in h5f['observations']:
                raise ValueError(f"Episode {episode_key} not found in {h5_filename}")
            
            # Load the required frames
            episode_data = h5f['observations'][episode_key]
            end_frame = min(start_frame + num_frames, episode_data.shape[0])
            frames = episode_data[start_frame:end_frame]
            
            # Convert to list of numpy arrays (same format as MP4 loader)
            frame_list = [frames[i] for i in range(frames.shape[0])]
            
        return frame_list

    def __getitem__(self, idx):
        # Get sequence info from pre-computed sequences
        seq_info = self.episode_sequences[idx]
        episode_idx = seq_info['episode_idx']
        offset = seq_info['start_idx']
        seq_len = seq_info['seq_length']
        
        # Calculate actual video length (should be sequence_length for valid sequences)
        actual_video_len = min(self.video_len, seq_len - offset)
        
        # Load video observations
        frames = None
        
        if self.use_h5:
            # Load from HDF5 files
            try:
                frames = self._load_h5_frames(episode_idx, offset, actual_video_len)
            except Exception as e:
                print(f"Error loading HDF5 frames for episode {episode_idx}: {e}")
                frames = None
                
        elif self.use_video and len(self.obs_files) > episode_idx:
            # Load from MP4 files
            video_path = os.path.join(self.root, self.split, "obses", f"episode_{episode_idx:03d}.mp4")
            if os.path.exists(video_path):
                frames = self._load_video_frames(video_path, offset, actual_video_len)
        
        # Process loaded frames
        if frames is not None and len(frames) > 0:
            # Transform frames
            transformed_frames = []
            for frame in frames:
                frame_tensor = self.transform(frame)
                transformed_frames.append(frame_tensor)
            
            # Pad if necessary
            while len(transformed_frames) < self.video_len:
                transformed_frames.append(transformed_frames[-1].clone())
            
            pixel_values = torch.stack(transformed_frames[:self.video_len])
        else:
            # Fallback: create dummy frames
            dummy_frame = torch.zeros(3, self.img_size, self.img_size)
            pixel_values = dummy_frame.unsqueeze(0).repeat(self.video_len, 1, 1, 1)
        
        if self.squeeze_time and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        
        # Get auxiliary data
        episode_abs_actions = self.abs_actions[episode_idx, offset:offset+self.video_len]
        episode_rel_actions = self.rel_actions[episode_idx, offset:offset+self.video_len]
        episode_states = self.states[episode_idx, offset:offset+self.video_len]
        
        # Pad actions and states if necessary
        if episode_abs_actions.shape[0] < self.video_len:
            pad_length = self.video_len - episode_abs_actions.shape[0]
            episode_abs_actions = torch.cat([
                episode_abs_actions, 
                episode_abs_actions[-1:].repeat(pad_length, 1)
            ], dim=0)
            episode_rel_actions = torch.cat([
                episode_rel_actions, 
                episode_rel_actions[-1:].repeat(pad_length, 1)
            ], dim=0)
            episode_states = torch.cat([
                episode_states, 
                episode_states[-1:].repeat(pad_length, 1)
            ], dim=0)
        
        if self.output_dict:
            return {
                "pixel_values": pixel_values,
                "abs_actions": episode_abs_actions,
                "rel_actions": episode_rel_actions,
                "states": episode_states,
                "seq_length": torch.tensor(actual_video_len),
            }
        return pixel_values


class PushTDatasetWithFactors(PushTDataset):
    """
    Extended PushT dataset that also returns factor labels.
    
    This is useful for evaluation and analysis of learned representations.
    For PushT data, factors represent agent position, block position, and block angle.
    """
    
    def __init__(self, *args, factor_extractor=None, labels=None, **kwargs):
        """
        Args:
            factor_extractor: Function to extract factors from observations (not used)
            labels: List of factor names (for compatibility with existing code)
        """
        super().__init__(*args, **kwargs)
        self.factor_extractor = factor_extractor
        self.labels = labels or ["agent_pos", "block_pos", "block_angle"]
    
    def __getitem__(self, idx):
        """Get trajectory sequence with factor labels."""
        # Get data from base class
        result = super().__getitem__(idx)
        
        # Extract pixel values and states
        if self.output_dict:
            pixel_values = result["pixel_values"]
            episode_states = result["states"]  # Shape: (sequence_length, 5)
        else:
            pixel_values = result
            # Need to get states separately for factor extraction
            # This requires accessing the internal data loading logic
            seq_info = self._get_sequence_info(idx)
            episode_states = self._get_episode_states(seq_info)
        
        # During training, return only the image
        if self.split == "train":
            return pixel_values
        
        # During evaluation, return 3-tuple with factors
        # Generate labels from states and use middle frame's factors
        state_labels = self._generate_factors(episode_states)
        middle_idx = self.sequence_length // 2
        factors = state_labels[middle_idx]  # Shape: (3,) - [agent_pos, block_pos, block_angle]
        
        dummy_masks = torch.ones(1, 1, self.img_size, self.img_size)  # Dummy masks
        
        return pixel_values, factors, dummy_masks
    
    def _generate_factors(self, states):
        """
        Generate labels for property prediction from states.
        States format: [agent_x, agent_y, block_x, block_y, block_angle]
        Returns labels with shape [video_len, 3] where:
        - Column 0: Agent position label (position_bins^2 classes)
        - Column 1: Block position label (position_bins^2 classes) 
        - Column 2: Block angle label (angle_bins classes)
        """
        # Define normalization ranges based on data analysis
        agent_min = 10
        block_min = 140
        agent_max = 490
        block_max = 440
        
        # Extract state components
        agent_x = states[:, 0]  # [video_len]
        agent_y = states[:, 1]  # [video_len]
        block_x = states[:, 2]  # [video_len]
        block_y = states[:, 3]  # [video_len] 
        block_angle = states[:, 4]  # [video_len]
        
        # Normalize positions to [0, 1] range
        agent_x_norm = (agent_x - agent_min) / (agent_max - agent_min)
        agent_y_norm = (agent_y - agent_min) / (agent_max - agent_min)
        block_x_norm = (block_x - block_min) / (block_max - block_min)
        block_y_norm = (block_y - block_min) / (block_max - block_min)
        
        # Clamp normalized values to [0, 1] to handle outliers
        agent_x_norm = torch.clamp(agent_x_norm, 0, 1)
        agent_y_norm = torch.clamp(agent_y_norm, 0, 1)
        block_x_norm = torch.clamp(block_x_norm, 0, 1)
        block_y_norm = torch.clamp(block_y_norm, 0, 1)
        block_angle = torch.clamp(block_angle, 0, 2 * np.pi)
        
        # Calculate bin sizes for normalized ranges
        pos_bin_size = 1.0 / self.position_bins  # For normalized [0, 1] range
        angle_bin_size = (2 * np.pi) / self.angle_bins
        
        # Quantize normalized agent position into position_bins x position_bins grid
        agent_x_bin = torch.floor(agent_x_norm / pos_bin_size).long().clamp(0, self.position_bins - 1)
        agent_y_bin = torch.floor(agent_y_norm / pos_bin_size).long().clamp(0, self.position_bins - 1)
        agent_pos_label = agent_y_bin * self.position_bins + agent_x_bin  # [0, position_bins^2 - 1]
        
        # Quantize normalized block position into position_bins x position_bins grid
        block_x_bin = torch.floor(block_x_norm / pos_bin_size).long().clamp(0, self.position_bins - 1)
        block_y_bin = torch.floor(block_y_norm / pos_bin_size).long().clamp(0, self.position_bins - 1)
        block_pos_label = block_y_bin * self.position_bins + block_x_bin  # [0, position_bins^2 - 1]
        
        # Discretize block angle into angle_bins classes
        block_angle_label = torch.floor(block_angle / angle_bin_size).long().clamp(0, self.angle_bins - 1)
        
        # Stack labels
        labels = torch.stack([agent_x_bin, agent_y_bin, agent_pos_label,block_x_bin, block_y_bin, block_pos_label, block_angle_label], dim=1)
        return labels

    def _get_sequence_info(self, idx):
        """Get sequence information for accessing states."""
        # Use the new stride-based sequence info
        seq_info = self.episode_sequences[idx]
        return {
            "episode_idx": seq_info['episode_idx'], 
            "offset": seq_info['start_idx']
        }
    
    def _get_episode_states(self, seq_info):
        """Get episode states for factor extraction."""
        episode_idx = seq_info["episode_idx"]
        offset = seq_info["offset"]
        
        # Note: seq_lengths available if needed for validation
        
        # Load states
        episode_states = self.states[episode_idx, offset:offset+self.video_len]
        
        # Pad states if necessary
        if episode_states.shape[0] < self.video_len:
            pad_length = self.video_len - episode_states.shape[0]
            episode_states = torch.cat([
                episode_states, 
                episode_states[-1:].repeat(pad_length, 1)
            ], dim=0)
        
        return episode_states
    
if __name__ == "__main__":
    dataset = PushTDatasetWithFactors('runner/datasets/data/pusht_noise',split="val", video_len=10)
    
    binc = [0]*16
    bins = [0]*16
    bina = [0]*4
    for i in range(0,len(dataset),10):
        a,b,c = dataset[i][1].tolist()
        binc[a] += 1
        bins[b] += 1
        bina[c] += 1

    print(binc)
    print(bins)
    print(bina)