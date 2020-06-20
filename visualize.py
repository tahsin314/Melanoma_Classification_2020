import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from torchvision.utils import save_image
from config import *
from MelanomaDataset import *
from utils import *

denorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def visualize(original_image):
    fontsize = 18
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(8,8,1)
    plt.axis('off')
    ax.imshow(original_image, cmap='gray')
    for i in range(63):
        augment = train_aug(image = image)
        aug_img = augment['image']
        ax = fig.add_subplot(8,8,i+2)
        plt.axis('off')
        ax.imshow(aug_img, cmap='gray')
    fig.savefig('aug.png')

train_df = pd.read_csv('data/folds.csv')
train_df = meta_df(train_df)
train_df['path'] = train_df['image_id'].map(lambda x: os.path.join(image_path,'{}.jpg'.format(x)))
train_meta = np.array(train_df[meta_features].values, dtype=np.float32)
train_ds = MelanomaDataset(train_df.path.values, train_meta, train_df.target.values, dim=512, transforms=train_aug)
train_loader = DataLoader(train_ds,batch_size=64, shuffle=True, num_workers=4)
im, _, _ = iter(train_loader).next()
# print(im.size(), torch.max(denorm(im)))
save_image(im.float(), 'Aug.png', nrow=8, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)