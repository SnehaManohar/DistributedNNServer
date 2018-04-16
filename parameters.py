from torchvision.transforms import transforms


class Parameters:
    data_dir = '~'
    download = True

    image_mean = [0.49139968, 0.48215841, 0.44653091]
    image_std = [0.24703223, 0.24348513, 0.26158784]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)
    ])

    train_batch_size = 256
    test_batch_size = 256

    epochs = 1
    log_interval = 10
    verbose_pre_training = True
    verbose_training_request = True

    merge_ratio = 0.1

    update_json_time = 10
    test_model_time = 60
