from tqts.data.dataloader import ETTHourDataset, ETTMinDataset, ETTHPredDataset
from torch.utils.data import DataLoader

# Mapping of data identifiers to corresponding dataset classes
data_dict = {
    'ETTh1': ETTHourDataset,
    'ETTh2': ETTHourDataset,
    'ETTm1': ETTMinDataset,
    'ETTm2': ETTMinDataset,
}


def data_provider(args, flag: str):
    """
    Provides data loader for the specified dataset.

    Based on the configuration and the flag, this function selects the appropriate
    dataset, creates an instance of it, and then creates a DataLoader for batch processing.

    Parameters:
    args (object): A configuration object containing dataset parameters.
    flag (str): Specifies the mode for the data loader ('train', 'test', 'pred', etc.).

    Returns:
    Tuple[Dataset, DataLoader]: A tuple containing the dataset instance and the corresponding DataLoader.
    """
    Data = data_dict[args.data]
    time_enc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    # Setting parameters based on the mode
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = ETTHPredDataset
    else:  # Assuming train mode
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Creating the dataset instance
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        time_enc=time_enc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))

    # Creating the DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader