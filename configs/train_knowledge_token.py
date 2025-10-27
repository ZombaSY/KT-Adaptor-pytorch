input_size=[224, 224]   # (height, width)
crop_size=224

conf=dict(
    env=dict(
        debug=False,
        CUDA_VISIBLE_DEVICES='0',
        mode='train',
        cuda=True,
        wandb=True,
        saved_model_directory='model_ckpt',
        project_name='facial-token',
        task='regression',    # classification, regression
        train_fold=1,
        epoch=10000,
        early_stop_epoch=100,
    ),

    model=dict(
    name='KTAdaptorModel',
        facial_tasks=dict(
            Mobileone_s0_age_estimation=dict(
                name='Mobileone_s0_age_estimation',
                num_class=1,
                input_size=input_size,
                saved_ckpt='pretrained/Mobileone_s0_age_estimation-folds_0-loss_4.4750-Epoch_1293.pt',
            ),
            Mobileone_s0_face_emotion_recognition=dict(
                name='Mobileone_s0_face_emotion_recognition',
                num_class=7,
                input_size=input_size,
                saved_ckpt='pretrained/Mobileone_s0_face_emotion_recognition-folds_0-f1_0.7020-Epoch_85.pt',
            ),
            Mobileone_s0_landmark_detection=dict(
                name='Mobileone_s0_landmark_detection',
                num_class=196,
                input_size=input_size,
                saved_ckpt='pretrained/Mobileone_s0_landmark_detection-folds_0-loss_0.1636-Epoch_978-WFLW.pt',
            ),
            Mobileone_s0_face_recognition=dict(
                name='Mobileone_s0_face_recognition',
                num_class=500,
                input_size=input_size,
                inference_mode=False,
                saved_ckpt='pretrained/Mobileone_s0_face_recognition-folds_0-f1_0.8504-Epoch_914.pt',
            ),
        ),
        selected_task='Mobileone_s0_landmark_detection',
        input_size=input_size,
        inference_mode=False,
        freeze_canonical_backbone=True,
        num_class=196,  # 1, 7, 196, 500
        num_task=4,
        depths=1,
        num_heads=4,
        token_dim=1024,   # max dimension of tokens
        target_dim=1024,  # dimsension of selected task
        saved_ckpt='',
    ),

    dataloader_train=dict(
        name='Image2Landmark',
        mode='train',
        data_path='/path/to/WFLW/train.csv',
        data_cache=True,
        weighted_sampler=False,
        batch_size=32,
        input_size=input_size,
        workers=8,

        augmentations=dict(
            transform_blur=0.5,
            transform_clahe=0.5,
            transform_cutmix=0.0,
            transform_coarse_dropout=0.5,   # available albumentations >= 1.4.17
            transform_fancyPCA=0.5,
            transform_fog=0.3,
            transform_g_noise=0.5,
            transform_jitter=0.5,
            transform_hflip=0.0,  # use `transform_landmark_hflip` instead.
            transform_vflip=0.0,
            transform_jpeg=0.5,
            transform_perspective=0.2,
            transform_rand_resize=0.0,
            transform_rand_crop=crop_size,
            transform_resize=input_size,
            transform_rain=0.3,
            transform_rotate=0.2,
            transform_landmark_hflip=0.5,
        )
    ),
    dataloader_valid=dict(
        name='Image2Landmark',
        mode='valid',
        data_path='/path/to/WFLW/valid.csv',
        data_cache=True,
        weighted_sampler=False,
        batch_size=32,
        input_size=input_size,
        workers=8,
    ),

    criterion=dict(
        name='WingLoss',
        omega=0.1,
        epsilon=2
    ),

    optimizer=dict(
        name='Adam',
        lr=1e-4,
        lr_min=1e-6,
        weight_decay=0,
    ),

    scheduler=dict(
        name='WarmupCosine',
        cycles=20,  # unit: epoch
        warmup_epoch=10
    ),
)