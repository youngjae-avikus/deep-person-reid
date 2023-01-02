import torchreid
import torch

weight = 'log/osnet_x1_0_norm_std_mean/model/model.pth.tar-10'
resume = False

datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources=["veri"],
    targets="veri",
    height=256,
    width=384,
    batch_size_train=32,
    batch_size_test=100,
    # norm_mean=[0.4209, 0.4206, 0.4267],
    # norm_std=[0.1869, 0.1857, 0.1851],
    transforms=["random_flip", "random_crop"]
)

model = torchreid.models.build_model(
    name="osnet_ibn_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=True
)

print("num_classes", datamanager.num_train_pids)
model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

start_epoch=0
if resume:
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Loaded checkpoint from "{}"'.format(weight))
    print('- start epoch: {}'.format(start_epoch))
    
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir="log/osnet_x1_0_norm_std_mean",
    start_epoch=start_epoch,
    max_epoch=30,
    eval_freq=5,
    print_freq=200,
    test_only=False
)
