import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
# my implementation
from V_NAS import Network, get_device, get_params_to_update
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)

# Define the train loader and the val loader
Train_datalist = load_decathlon_datalist("./data/dataset.json", True, "training")
# Val_datalist = load_decathlon_datalist("./data/dataset.json", True, "val")




train_ds = CacheDataset(
    data=Train_datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True
)
print("num of train {}".format(len(train_ds)))

device = get_device(1)
# define the model
model = Network((64, 64, 64), 3).to(device)


##################################################################

# define the loss function
dice_loss = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True
)
ce_loss = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([1., 1., 20.]).to(device)
)
def loss_function(dice, ce, pred, label, weight_ce, weight_dice):
    return weight_dice*dice(pred, label) + weight_ce*ce(pred, label.squeeze(1).long())



w_opter = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_init(model, loader, loss_w, opt_w, epoch):
    model.train()
    for i in range(epoch):
        l = 0.0
        for step, batch in enumerate(loader):
            opt_w.zero_grad()
            x, y = batch["image"].to(device), batch["label"].to(device)
            o = model(x)
            loss = loss_w(dice_loss, ce_loss, o, y, 1.4, 1)
            loss.backward()
            opt_w.step()
            l += loss.item()
            
            print(loss.item())
        torch.save(model.state_dict(), "./model/one_shot.ptb")
        print("model saved !")
        print("Finished training on " + str(i) + " epoch")
        print("loss is {}".format(l / (step + 1)))

train_init(model, train_loader, loss_function, w_opter, 4000)
