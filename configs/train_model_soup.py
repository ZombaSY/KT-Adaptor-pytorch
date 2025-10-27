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
		task='regression',    # classification, landmark, regression, recognition
		train_fold=1,
		epoch=10000,
		early_stop_epoch=300,
	),

	model=dict(
		name='ModelEnsembleCat',
		facial_tasks=dict(
			Swin_t_face_age_estimation=dict(
			name='Swin_t_face_age_estimation',
			num_class=1,
			input_size=input_size,
			saved_ckpt='pretrained/Swin_t_face_age_estimation-folds_0-loss_4.3200-Epoch_84.pt',
			),
			Swin_t_face_emotion_recognition=dict(
			name='Swin_t_face_emotion_recognition',
			num_class=7,
			input_size=input_size,
			saved_ckpt='pretrained/Swin_t_face_emotion_recognition-folds_0-f1_0.7174-Epoch_18.pt',
			),
			Swin_t_face_landmark_detection=dict(
			name='Swin_t_face_landmark_detection',
			num_class=196,
			input_size=input_size,
			saved_ckpt='pretrained/Swin_t_face_landmark_detection-folds_0-loss_0.1416-Epoch_211-WFLW.pt',
			),
			Swin_t_face_recognition=dict(
			name='Swin_t_face_recognition',
			num_class=500,
			input_size=input_size,
				inference_mode=False,
			saved_ckpt='pretrained/Swin_t_face_recognition-folds_0-f1_0.8584-Epoch_525.pt',
			),
		),
		selected_task='Swin_t_face_age_estimation',
		input_size=input_size,
		inference_mode=False,
		num_class=500,
		num_task=4,
		channel_in=1536 * 4,     # 1024, 1536
		saved_ckpt='',
	),

	dataloader_train=dict(
		name='Image2Vector',
		mode='train',
		data_path='/path/to/UTK-face/train.csv',
		data_cache=True,
		label_cols=['label'],
		weighted_sampler=False,
		batch_size=32,
		input_size=input_size,
		workers=8,

		augmentations=dict(
			transform_blur=0.5,
			transform_clahe=0.5,
			transform_cutmix=0.0,
			transform_mixup=0.3,
			transform_coarse_dropout=0.5,   # available albumentations >= 1.4.17
			transform_fancyPCA=0.5,
			transform_fog=0.3,
			transform_g_noise=0.5,
			transform_jitter=0.5,
			transform_hflip=0.5,
			transform_vflip=0.0,
			transform_jpeg=0.5,
			transform_perspective=0.2,
			transform_rand_resize=0.0,
			transform_rand_crop=crop_size,
			transform_resize=input_size,
			transform_rain=0.3,
			transform_rotate=0.2,
		)
	),
	dataloader_valid=dict(
		name='Image2Vector',
		mode='valid',
		data_path='/path/to/UTK-face/valid.csv',
		data_cache=True,
		label_cols=['label'],
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