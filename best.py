import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)



# my implementation

from V_NAS import get_device
from searched_nas import Network_best


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


# load the 
Train_datalist = load_decathlon_datalist("data/dataset.json", True, "training")
Val_datalist = load_decathlon_datalist("data/dataset.json", True, "val")



train_ds = CacheDataset(
    data=Train_datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds, batch_size=8, shuffle=True, num_workers=6, pin_memory=True
)
val_ds = CacheDataset(
    data=Val_datalist, transform=val_transforms, cache_num=1, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
)
print("num of train_ds {}, num of val_ds {}".format(len(train_ds), len(val_ds)))



device = get_device(0)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)




model = Network_best((64, 64, 64), 3).to(device)
def validation(model, val_loader):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_converted = [post_label(x) for x in val_labels_list]
            val_output_converted = [post_pred(x) for x in val_outputs]
            dice_metric(y_pred=val_output_converted, y=val_labels_converted)


        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)




eval_num = 10
max_iterations = 5000
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

val_list = []
def train(model, global_step, train_loader, dice_val_best):
    model.train()
    epoch_loss = 0

    step = 0
    for step, batch in enumerate(train_loader):
        step += 1
        x, y = batch["image"].to(device), batch["label"].to(device)
        o = model(x)
        loss = loss_function(dice_loss, ce_loss, o, y, 1.5, 0.9)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        print("--------PRINTING LOSS IN BATCHES--------")
        print("Global step {}, local step {}, loss {}, best dice {}".format(global_step, step, loss.item(), dice_val_best))



    if (global_step % eval_num == 0):
        dice_val = validation(model, val_loader)
        print("--------PRINTING DICE IN VALIDATIONS_SET--------")
        print("At global_step {}, local_step {}, dice mean score: {}".format(global_step, step, dice_val))
        val_list.append(dice_val)
        if dice_val > dice_val_best:
            dice_val_best = dice_val
            
            torch.save(
                model.state_dict(), "./model/best.ptb"
            )
            print(colored(0, 255, 0, "YYYYYYYY  Model Was Saved YYYYYYYYYYY"))
        else:
            print(colored(255, 0, 0, "xxxxxxxxx Model was not saved xxxxxxxxxx"))
            print("xxxxxxxxx Model was not saved xxxxxxxxxx")
    global_step += 1

    return global_step, dice_val_best

eval_num = 1
max_iterations = 500

dice_val_best = 0

global_step = 0
while global_step < max_iterations:
    print("This is EPOCH {}".format(global_step))
    global_step, dice_val_best = train(model, global_step, train_loader, dice_val_best)
    print("At global step{}, dice_val_best {}".format(global_step, dice_val_best))