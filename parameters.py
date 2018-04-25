from torchvision.transforms import transforms


class Parameters:
    data_dir = 'data'

    image_mean = [0.500424, 0.456539, 0.410315]
    image_std = [0.250922, 0.240401, 0.269415]
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)
    ])

    batch_size = 32

    log_interval = 4

    merge_ratio = 0.9

    update_json_time = 10
    test_model_time = 60
