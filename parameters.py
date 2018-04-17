from torchvision.transforms import transforms


class Parameters:
    data_dir = 'data'
    train_data = data_dir + '/train'
    test_data = data_dir + '/test'

    image_mean = [0.49842002987861633, 0.45849472284317017, 0.3978663980960846]
    image_std = [0.31304672360420227, 0.26907557249069214, 0.23163224756717682]
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)
    ])

    train_batch_size = 256
    test_batch_size = 256

    epochs = 0
    log_interval = 10
    verbose_pre_training = True
    verbose_training_request = True

    merge_ratio = 0.1

    update_json_time = 10
    test_model_time = 60
