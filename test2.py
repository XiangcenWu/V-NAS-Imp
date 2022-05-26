import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
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


val_transforms = Compose(
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
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)


# Define the train loader and the val loader
Train_datalist = load_decathlon_datalist("./data/dataset.json", True, "training")
Val_datalist = load_decathlon_datalist("./data/dataset.json", True, "val")




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

# val_ds = CacheDataset(
#     data=Val_datalist, transform=val_transforms, cache_num=1, cache_rate=1.0, num_workers=4
# )
# val_loader = DataLoader(
#     val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
# )
# print("num of train_ds {}, num of val_ds {}".format(len(train_ds), len(val_ds)))

sa_ds = CacheDataset(
    data=Val_datalist,
    transform=train_transforms,
    cache_num=8,
    cache_rate=1.0,
    num_workers=8,
)
sa_loader = DataLoader(
    sa_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True
)



device = get_device(2)
# define the model
model = Network((64, 64, 64), 3).to(device)

# define the optimizer for w
model.update_w()
w_opter = torch.optim.SGD(get_params_to_update(model), lr=0.05, weight_decay=0.0005, momentum=0.9)
# # define the optimizer for a
model.update_alpha()
a_opter = torch.optim.Adam(get_params_to_update(model), lr=0.0003, weight_decay=0.001)




def search(model, train_loader, sa_loader, loss_a, loss_w, opt_a, opt_w, epoch):
    
    model.train()
    
    for i in range(epoch):
        w_loss = 0.
        a_loss = 0.
        print("##############This is epoch {} ################".format(i))
        model.update_every()
        for step, batch in enumerate(train_loader):
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt_w.zero_grad()
            x, y = batch["image"].to(device), batch["label"].to(device)
            o = model(x)
            loss = loss_w(o, y)
            loss.backward()
            opt_w.step()
            w_loss += loss.item()
            print(loss.item())

        print("w is trained completed with loss {}".format(w_loss / step))
        
            

        model.update_every()
        for step, batch in enumerate(sa_loader):
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt_a.zero_grad()
            x, y = batch["image"].to(device), batch["label"].to(device)
            o = model(x)
            loss = loss_a(o, y)
            loss.backward()
            opt_a.step()
            a_loss += loss.item()
            print(loss.item())

        print("a is trained completed with loss {}".format(a_loss / step))







loss_function_w = DiceCELoss(
    include_background=True, 
    to_onehot_y=True, 
    ce_weight=torch.tensor([0., 0.2, 0.8]).to(device),
)

loss_function_a = DiceCELoss(
    include_background=True, 
    to_onehot_y=True, 
    ce_weight=torch.tensor([0., 0.2, 0.8]).to(device),
    lambda_ce=0.8,
    lambda_dice=0.2
)



search(model, train_loader, sa_loader, loss_function_a, loss_function_w, a_opter, w_opter, 1)