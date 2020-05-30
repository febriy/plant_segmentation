from dataloader import PlantDataloader, PlantToInferloader, mean, std
import torch
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent.parent
data_path = Path(base_path / "data/").resolve()

df = pd.read_csv(data_path / "my_metadata.csv")

# location of original and mask image
img_fol = data_path / "mydata-512"
mask_fol = data_path / "mytrain_masks_bw-128"


# test_dataloader = PlantDataloader(df, img_fol, mask_fol, mean, std, "val", 1, 4)
test_dataloader = PlantToInferloader(img_fol, mean, std, 1, 4)
ckpt_path = base_path / "model_office.pth"

device = torch.device("cuda")
model = smp.Unet("resnet50", encoder_weights=None, classes=1, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# start prediction
for i, batch in enumerate(test_dataloader):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    fig.suptitle("predicted_mask//original_mask")
    print("i", i)
    images, mask_target = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
    ax1.imshow(np.squeeze(batch_preds), cmap="gray")
    ax2.imshow(np.squeeze(mask_target))  # if using own dataset
    # ax2.imshow(np.squeeze(mask_target), cmap="gray")
    plt.show()
