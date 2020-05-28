from evaluation import *
from dataloader import *
from preprocessing import *

base_path = Path(__file__).parent.parent

test_dataloader = PlantDataloader(df, img_fol, mask_fol, mean, std, "val", 1, 4)
ckpt_path = base_path / "model_office.pth"

device = torch.device("cuda")
model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
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
    ax2.imshow(np.squeeze(mask_target), cmap="gray")
    plt.show()  # show the figure, non-blocking
