import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

KVASIR_LABELS = [
    "dyed lifted polyps",
    "dyed resection margins",
    "esophagitis",
    "normal cecum",
    "normal pylorus",
    "normal z-line",
    "polyps",
    "ulcerative colitis"
]

class KvasirDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df
        self.data_root = data_root
        self.transform = transform
        self.label_to_idx = {label: i for i, label in enumerate(KVASIR_LABELS)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract class from text: "An endoscopic image of <class>"
        text = row['text']
        prefix = "An endoscopic image of "
        label_str = text.replace(prefix, "").strip()
        label = self.label_to_idx[label_str]

        return image, label

class KvasirDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, data_root, batch_size=32, image_size=224, num_workers=8, val_split=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.data_root = data_root
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        train_df, val_df = train_test_split(df, test_size=self.val_split, random_state=42, stratify=df['text'])

        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = KvasirDataset(train_df, self.data_root, train_transform)
        self.val_dataset = KvasirDataset(val_df, self.data_root, val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

class SyntheticKvasirDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, original_data_root, synthetic_data_root, batch_size=32, image_size=224, num_workers=8, val_split=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.original_data_root = original_data_root
        self.synthetic_data_root = synthetic_data_root
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        # Split original files into train and val
        train_df, val_df = train_test_split(df, test_size=self.val_split, random_state=42, stratify=df['text'])

        # Prepare labels mapping
        label_map = {row['file_name']: row['text'] for _, row in df.iterrows()}
        stem_to_label = {os.path.splitext(f)[0]: label for f, label in label_map.items()}

        # Discover synthetic files for training
        train_stems = set(os.path.splitext(f)[0] for f in train_df['file_name'])
        all_gen_files = [f for f in os.listdir(self.synthetic_data_root) if f.endswith(".jpg")]

        synthetic_train_data = []
        for f in all_gen_files:
            stem = f.rsplit('_', 1)[0]
            if stem in train_stems:
                synthetic_train_data.append({
                    'file_name': f,
                    'text': stem_to_label[stem]
                })

        synthetic_train_df = pd.DataFrame(synthetic_train_data)

        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Train on synthetic, val on real
        self.train_dataset = KvasirDataset(synthetic_train_df, self.synthetic_data_root, train_transform)
        self.val_dataset = KvasirDataset(val_df, self.original_data_root, val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
