# dataset.py
import os, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class KeypointSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=30, transform=None):
        self.root = root_dir
        self.seq_len = seq_len
        self.samples = []
        for cls in os.listdir(root_dir):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for f in os.listdir(cls_dir):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir,f), cls))
        self.le = LabelEncoder()
        classes = sorted({s[1] for s in self.samples})
        self.le.fit(classes)
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        arr = np.load(path)  # (T,21,3)
        # normalize: center on wrist (landmark 0)
        wrist = arr[:,0:1,:]  # (T,1,3)
        arr = arr - wrist  # translate
        # Flatten spatial dims: (T, 21*3)
        arr = arr.reshape(self.seq_len, -1).astype('float32')
        if self.transform:
            arr = self.transform(arr)
        label = int(self.le.transform([cls])[0])
        return torch.from_numpy(arr), torch.tensor(label)

    def classes(self): return list(self.le.classes_)
