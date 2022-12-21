import torchreid
from torchreid.utils import (load_pretrained_weights, compute_model_complexity)

weight = 'log/osnet_x1_0_veri_m1501_comb/model/model.pth.tar-25'

datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="veri",
    targets="veri",
    height=256,
    width=384,
    batch_size_train=32,
    batch_size_test=100,
    transforms=["random_flip", "random_crop"]
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=True
)

print("num_classes", datamanager.num_train_pids)
model = model.cuda()

load_pretrained_weights(model, weight)
num_params, flops = compute_model_complexity(model, (1, 3, 256, 384))
        
print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

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
    save_dir="log/osnet_x1_0_veri_m1501_comb",
    max_epoch=30,
    eval_freq=5,
    print_freq=200,
    test_only=True,
    dist_metric="cosine"
)
