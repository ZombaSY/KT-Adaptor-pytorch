input_size = [224, 224]

conf = dict(
    env=dict(
        debug=True,
        CUDA_VISIBLE_DEVICES="3",
        mode="valid",  # test, valid
        cuda=True,
        task="regression",  # classification, landmark, regression, recognition
        draw_results=False,
    ),
    model=dict(
        name="ModelSoupsAverage",
        facial_tasks=dict(
            Swin_t_face_age_estimation=dict(
                name="Swin_t_face_age_estimation",
                num_class=1,
                input_size=input_size,
                saved_ckpt="pretrained/Swin_t_face_age_estimation-folds_0-loss_4.3200-Epoch_84.pt",
            ),
            Swin_t_face_emotion_recognition=dict(
                name="Swin_t_face_emotion_recognition",
                num_class=7,
                input_size=input_size,
                saved_ckpt="pretrained/Swin_t_face_emotion_recognition-folds_0-f1_0.7174-Epoch_18.pt",
            ),
            Swin_t_face_landmark_detection=dict(
                name="Swin_t_face_landmark_detection",
                num_class=196,
                input_size=input_size,
                saved_ckpt="pretrained/Swin_t_face_landmark_detection-folds_0-loss_0.1416-Epoch_211-WFLW.pt",
            ),
            Swin_t_face_recognition=dict(
                name="Swin_t_face_recognition",
                num_class=500,
                input_size=input_size,
                inference_mode=False,
                saved_ckpt="pretrained/Swin_t_face_recognition-folds_0-f1_0.8584-Epoch_525.pt",
            ),
        ),
        selected_task="Swin_t_face_age_estimation",
        input_size=input_size,
        inference_mode=False,
        num_class=1,  # 196, 212
        num_task=4,
        channel_in=1536 * 4,  # 1024, 1536
        saved_ckpt="model_ckpt/ModelSoupsAverage-folds_0-loss_y_5.3828-Epoch_817.pt",
    ),
    dataloader_valid=dict(
        name="Image2Vector",
        mode="valid",
        data_path="/path/to/UTK-face/valid.csv",
        data_cache=True,
        label_cols=["label"],
        weighted_sampler=False,
        batch_size=32,
        input_size=input_size,
        workers=8,
    ),
    criterion=dict(
        name="L1loss", omega=0.1, epsilon=2  # NMELoss, NME_ic, L1loss, CrossEntropyLoss
    ),
)
