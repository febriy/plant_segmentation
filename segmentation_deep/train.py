from evaluation import *
from dataloader import *
from preprocessing import *


class Trainer(object):
    def __init__(self, model):
        self.num_workers = 4
        self.batch_size = {"train": 1, "val": 1}
        self.accumulation_steps = 4 // self.batch_size["train"]
        self.lr = 5e-4
        self.num_epochs = 10
        self.phases = ["train", "val"]
        self.best_loss = float("inf")
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model.to(self.device)
        cudnn.benchmark = True
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, verbose=True
        )
        self.dataloaders = {
            phase: PlantDataloader(
                df,
                img_fol,
                mask_fol,
                mean,
                std,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }

        self.losses = {phase: [] for phase in self.phases}
        self.dice_score = {phase: [] for phase in self.phases}

    def forward(self, inp_images, tar_mask):
        inp_images = inp_images.to(self.device)
        tar_mask = tar_mask.to(self.device)
        pred_mask = self.net(inp_images)
        loss = self.criterion(pred_mask, tar_mask)
        return loss, pred_mask

    def iterate(self, epoch, phase):
        measure = Scores(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase:{phase} | 🙊':{start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, mask_target = batch
            loss, pred_mask = self.forward(images, mask_target)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            pred_mask = pred_mask.detach().cpu()
            measure.update(mask_target, pred_mask)
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(epoch_loss, measure)
        self.losses[phase].append(epoch_loss)
        self.dice_score[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office.pth")
            print()


if __name__ == "__main__":
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    model_trainer = Trainer(model)
    model_trainer.start()
