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
Train_datalist = load_decathlon_datalist("./data/dataset_nas.json", True, "training")
Val_datalist = load_decathlon_datalist("./data/dataset_nas.json", True, "val")




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


sa_ds = CacheDataset(
    data=Val_datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
sa_loader = DataLoader(
    train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True
)
print("num of train {}, num of sa {}".format(len(train_ds), len(sa_ds)))

device = get_device(2)
# define the model
model = Network((64, 64, 64), 3).to(device)


# define the optimizer for w
model.update_w()
w_opter = torch.optim.SGD(get_params_to_update(model), lr=0.005, weight_decay=0.0005, momentum=0.9)
# # define the optimizer for a
model.update_alpha()
a_opter = torch.optim.Adam(get_params_to_update(model), lr=0.0003, weight_decay=0.001)



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



def search(model, train_loader, sa_loader, loss_function, opt_a, opt_w, epoch):
    
    model.train()
    for i in range(epoch):
        w_loss = 0.
        a_loss = 0.
        model.update_w()
        for step, batch in enumerate(train_loader):
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt_w.zero_grad()
            x, y = batch["image"].to(device), batch["label"].to(device)
            o = model(x)
            loss = loss_function(dice_loss, ce_loss, o, y, 1.4, 1)
            loss.backward()
            print(loss.item())
            opt_w.step()
            w_loss += loss.item()

        print("w is trained completed with loss {}".format(w_loss / (step+1)))
        
            

        model.update_alpha()
        for step, batch in enumerate(sa_loader):
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt_a.zero_grad()
            x, y = batch["image"].to(device), batch["label"].to(device)
            o = model(x)
            loss = loss_function(dice_loss, ce_loss, o, y, 1.4, 1)
            print(loss.item())
            loss.backward()
            opt_a.step()
            a_loss += loss.item()

        print("a is trained completed with loss {}".format(a_loss / (step + 1)))

        torch.save(model.state_dict(), "./model/search.ptb")
        print("##############################")
        print("model saved !")
        print("epoch {} trained completed".format(i))
        model.log()
        print("##############################")



def train_init(model, loader, loss_w, opt_w, epoch):
    model.train()
    model.update_every()
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

        print("Finished training on " + str(i) + " epoch")
        print("loss is {}".format(l / (step + 1)))

train_init(model, train_loader, loss_function, w_opter, 20)

search(model, train_loader, sa_loader, loss_function, a_opter, w_opter, 3000)
torch.save(model.state_dict(), "./model/search.ptb")
