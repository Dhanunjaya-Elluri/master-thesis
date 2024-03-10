#!/usr/bin/env python
# coding: utf-8

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

"""
Main function for training and evaluating the Pyraformer model on different time series datasets.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyraformer.Pyraformer_LR as Pyraformer
import torch
import torch.optim as optim
from dataloader import ETTHourDataset, ETTMinDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import TopkMSELoss, metric

from tqts.utils.data_utils import vectorized_find_character
from tqts.utils.plot_utils import save_matching_distribution_plot

warnings.filterwarnings("ignore")


def prepare_dataloader(args):
    """Load data and prepare dataloader."""

    data_dict = {
        "ETTh1": ETTHourDataset,
        "ETTh1_stationary": ETTHourDataset,
        "ETTh2": ETTHourDataset,
        "ETTm1": ETTMinDataset,
        "ETTm2": ETTMinDataset,
        "electricity": ETTHourDataset,
        "exchange": ETTHourDataset,
        "traffic": ETTHourDataset,
        "weather": ETTHourDataset,
        "ili": ETTHourDataset,
    }
    Data = data_dict[args.data]

    # prepare training dataset and dataloader
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size
    train_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag="train",
        size=[args.input_size, args.predict_step],
    )
    print("train", len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last,
    )

    # prepare testing dataset and dataloader
    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    test_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag="test",
        size=[args.input_size, args.predict_step],
    )
    print("test", len(test_set))
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last,
    )

    return train_loader, train_set, test_loader, test_set


def sample_mining_scheduler(epoch, batch_size):
    if epoch < 2:
        topk = batch_size
    elif epoch < 4:
        topk = int(batch_size * (5 - epoch) / (6 - epoch))
    else:
        topk = int(0.5 * batch_size)

    return topk


def dataset_parameters(args, dataset):
    """Prepare specific parameters for different datasets"""
    dataset2enc_in = {
        "ETTh1": 1,
        "ETTh1_stationary": 1,
        "ETTh2": 1,
        "ETTm1": 1,
        "ETTm2": 1,
        "electricity": 1,
        "exchange": 8,
        "traffic": 862,
        "weather": 21,
        "ili": 7,
        "flow": 1,
        "synthetic": 1,
    }
    dataset2cov_size = {
        "ETTh1": 1,
        "ETTh1_stationary": 1,
        "ETTh2": 1,
        "ETTm1": 1,
        "ETTm2": 1,
        "electricity": 1,
        "exchange": 4,
        "traffic": 4,
        "weather": 4,
        "ili": 4,
        "elect": 3,
        "flow": 3,
        "synthetic": 3,
    }
    dataset2seq_num = {
        "ETTh1": 1,
        "ETTh1_stationary": 1,
        "ETTh2": 1,
        "ETTm1": 1,
        "ETTm2": 1,
        "electricity": 1,
        "exchange": 1,
        "traffic": 1,
        "weather": 1,
        "ili": 1,
        "elect": 321,
        "flow": 1077,
        "synthetic": 60,
    }
    dataset2embed = {
        "ETTh1": "DataEmbedding",
        "ETTh2": "DataEmbedding",
        "ETTm1": "DataEmbedding",
        "ETTm2": "DataEmbedding",
        "elect": "CustomEmbedding",
        "electricity": "CustomEmbedding",
        "exchange": "CustomEmbedding",
        "traffic": "CustomEmbedding",
        "weather": "CustomEmbedding",
        "ili": "CustomEmbedding",
        "flow": "CustomEmbedding",
        "synthetic": "CustomEmbedding",
    }

    args.enc_in = dataset2enc_in[dataset]
    args.dec_in = dataset2enc_in[dataset]
    args.covariate_size = dataset2cov_size[dataset]
    args.seq_num = dataset2seq_num[dataset]
    # args.embed_type = dataset2embed[dataset]

    return args


def train_epoch(model, train_dataset, training_loader, optimizer, opt, epoch):
    """Epoch operation in training phase."""

    model.train()
    total_loss = 0
    total_pred_number = 0
    warm = False
    for batch in tqdm(
        training_loader, mininterval=2, desc="  - (Training)   ", leave=False
    ):
        # prepare data
        batch_x, batch_y, batch_x_mark, batch_y_mark = map(
            lambda x: x.float().to(opt.device), batch
        )
        dec_inp = torch.zeros_like(batch_y).float()
        optimizer.zero_grad()

        # forward
        if opt.decoder == "attention":
            if opt.pretrain and epoch < 1:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, True)
                batch_y = torch.cat([batch_x, batch_y], dim=1)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
        elif opt.decoder == "FC":
            # Add a predict token into the history sequence
            predict_token = torch.zeros(
                batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device
            )
            batch_x = torch.cat([batch_x, predict_token], dim=1)
            batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

        # determine the loss function
        if opt.hard_sample_mining and not (opt.pretrain and epoch < 1):
            topk = sample_mining_scheduler(epoch, batch_x.size(0))
            criterion = TopkMSELoss(topk)
        else:
            criterion = torch.nn.MSELoss(reduction="none")
        # if inverse, both the output and the ground truth are denormalized.
        if opt.inverse:
            outputs, batch_y = train_dataset.inverse_transform(outputs, batch_y)
        # compute loss
        print(f"outputs shape: {outputs.shape}, batch_y shape: {batch_y.shape}")
        losses = criterion(outputs, batch_y)
        loss = losses.mean()
        loss.backward()

        """ update parameters """
        optimizer.step()
        total_loss += losses.sum().item()
        total_pred_number += losses.numel()

    return total_loss / total_pred_number


def eval_epoch(model, test_dataset, test_loader, opt, epoch, iter_index=0):
    """Epoch operation in evaluation phase."""

    model.eval()
    preds = []
    trues = []
    warm = 0
    with torch.no_grad():
        for batch in tqdm(
            test_loader, mininterval=2, desc="  - (Validation) ", leave=False
        ):
            """prepare data"""
            batch_x, batch_y, batch_x_mark, batch_y_mark = map(
                lambda x: x.float().to(opt.device), batch
            )
            dec_inp = torch.zeros_like(batch_y).float()

            # forward
            if opt.decoder == "FC":
                # Add a predict token into the history sequence
                predict_token = torch.zeros(
                    batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device
                )
                batch_x = torch.cat([batch_x, predict_token], dim=1)
                batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

            warm += 1
            # if inverse, both the output and the ground truth are denormalized.
            if opt.inverse:
                outputs, batch_y = test_dataset.inverse_transform(
                    outputs,
                    batch_y,
                )

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

    # preds = [pred.detach().cpu().numpy() for pred in preds] # np.array(preds)

    # trues = [true.detach().cpu().numpy() for true in trues] # np.array(trues)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    preds = np.concatenate(preds, axis=0)
    print(preds.shape)
    trues = np.concatenate(trues, axis=0)

    folder_path = (
        "../results/"
        + "et"
        + str(opt.embed_type)
        + "_Pyraformer_"
        + opt.data
        + "_"
        + str(iter_index)
        + "/"
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    boundaries_df = pd.read_csv(os.path.join(opt.root_path, opt.boundaries_df))
    true_values_flat = trues.flatten()
    pred_values_flat = preds.flatten()

    # Converting boundaries and alphabets to numpy arrays for vectorized operations
    lower_boundaries = boundaries_df["lower_boundaries"].to_numpy()
    upper_boundaries = boundaries_df["upper_boundaries"].to_numpy()
    alphabets = boundaries_df["alphabets"].to_numpy()

    # Applying the function to true and predicted values
    true_characters_vec = vectorized_find_character(
        true_values_flat, lower_boundaries, upper_boundaries, alphabets
    )
    pred_characters_vec = vectorized_find_character(
        pred_values_flat, lower_boundaries, upper_boundaries, alphabets
    )

    # Create a new DataFrame with true values, predicted values, and their corresponding characters
    result_df = pd.DataFrame(
        {
            "True Values": true_values_flat,
            "Predicted Values": pred_values_flat,
            "True Characters": true_characters_vec,
            "Predicted Characters": pred_characters_vec,
        }
    )
    # Calculating character distances
    mask = (result_df["True Characters"].notna()) & (
        result_df["Predicted Characters"].notna()
    )
    result_df["Character Distance"] = np.abs(
        result_df.loc[mask, "True Characters"].apply(ord)
        - result_df.loc[mask, "Predicted Characters"].apply(ord)
    ).astype(int)

    # Matching column creation as integer
    result_df["Matching"] = (
        result_df["True Characters"] == result_df["Predicted Characters"]
    ).astype(int)

    save_matching_distribution_plot(
        result_df, folder_path, opt.data, opt.model, opt.embed_type
    )
    result_df.to_csv(folder_path + "result_df.csv", index=False)

    # Computing the overall average character distance
    # average_distance_vec = np.mean(result_df["Character Distance"])

    # save average_distance_vec to a text file in folder_path with float format
    # np.savetxt(
    #     folder_path + "average_distance_vec.txt", [average_distance_vec], fmt="%f"
    # )

    print("test shape:{}".format(preds.shape))
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(
        "Epoch {}, mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(
            epoch, mse, mae, rmse, mape, mspe
        )
    )

    return mse, mae, rmse, mape, mspe


def train(model, optimizer, scheduler, opt, model_save_dir):
    """Start training."""

    best_mse = 100000000

    """ prepare dataloader """
    (
        training_dataloader,
        train_dataset,
        test_dataloader,
        test_dataset,
    ) = prepare_dataloader(opt)

    best_metrics = []
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print("[ Epoch", epoch, "]")

        start = time.time()
        train_mse = train_epoch(
            model, train_dataset, training_dataloader, optimizer, opt, epoch_i
        )
        print(
            "  - (Training) "
            "MSE: {mse: 8.5f}"
            "elapse: {elapse:3.3f} min".format(
                mse=train_mse, elapse=(time.time() - start) / 60
            )
        )

        mse, mae, rmse, mape, mspe = eval_epoch(
            model, test_dataset, test_dataloader, opt, epoch_i
        )

        scheduler.step()

        current_metrics = [
            float(mse),
            float(mae),
            float(rmse),
            float(mape),
            float(mspe),
        ]
        if best_mse > mse:
            best_mse = mse
            best_metrics = current_metrics
            torch.save(
                {"state_dict": model.state_dict(), "metrics": best_metrics},
                model_save_dir,
            )

    return best_metrics


def evaluate(model, opt, model_save_dir, iter_index):
    """Evaluate preptrained models"""
    best_mse = 100000000

    """ prepare dataloader """
    _, _, test_dataloader, test_dataset = prepare_dataloader(opt)
    """ load pretrained model """
    checkpoint = torch.load(model_save_dir)["state_dict"]
    model.load_state_dict(checkpoint)

    best_metrics = []
    mse, mae, rmse, mape, mspe = eval_epoch(
        model, test_dataset, test_dataloader, opt, 0, iter_index
    )

    current_metrics = [float(mse), float(mae), float(rmse), float(mape), float(mspe)]
    if best_mse > mse:
        best_mse = mse
        best_metrics = current_metrics

    return best_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    # running mode
    parser.add_argument("-eval", action="store_true", default=False)

    # Path parameters
    parser.add_argument("-data", type=str, default="ETTh1")
    parser.add_argument(
        "-root_path", type=str, default="../data", help="root path of the data file"
    )
    parser.add_argument(
        "-data_path", type=str, default="ETT-small/ETTh1_lloyd.csv", help="data file"
    )
    parser.add_argument(
        "-boundaries_df",
        type=str,
        default="ETT-small/ETTh1_lloyd_boundaries.csv",
        help="boundaries dataset",
    )
    # Dataloader parameters.
    parser.add_argument("-input_size", type=int, default=168)
    parser.add_argument("-predict_step", type=int, default=84)
    parser.add_argument(
        "-inverse", action="store_true", help="denormalize output data", default=False
    )

    # Architecture selection.
    parser.add_argument("-model", type=str, default="Pyraformer")
    parser.add_argument(
        "-decoder", type=str, default="FC"
    )  # selection: [FC, attention]
    parser.add_argument("-embed_type", type=int, default=1)

    # Training parameters.
    parser.add_argument("-epoch", type=int, default=10)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-pretrain", action="store_true", default=False)
    parser.add_argument("-hard_sample_mining", action="store_true", default=False)
    parser.add_argument("-dropout", type=float, default=0.05)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-lr_step", type=float, default=0.1)

    # Common Model parameters.
    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-d_inner_hid", type=int, default=512)
    parser.add_argument("-d_k", type=int, default=128)
    parser.add_argument("-d_v", type=int, default=128)
    parser.add_argument("-d_bottleneck", type=int, default=128)
    parser.add_argument("-n_head", type=int, default=4)
    parser.add_argument("-n_layer", type=int, default=4)

    # Pyraformer parameters.
    parser.add_argument(
        "-window_size", type=str, default="[1, 1, 1]"
    )  # The number of children of a parent node.
    parser.add_argument(
        "-inner_size", type=int, default=3
    )  # The number of ajacent nodes.
    # CSCM structure. selection: [BottleneckConstruct, ConvConstruct, MaxPoolingConstruct, AvgPoolingConstruct]
    parser.add_argument("-CSCM", type=str, default="BottleneckConstruct")
    parser.add_argument(
        "-truncate", action="store_true", default=False
    )  # Whether to remove coarse-scale nodes from the attention structure
    parser.add_argument(
        "-use_tvm", action="store_true", default=False
    )  # Whether to use TVM.

    # Experiment repeat times.
    parser.add_argument("-iter_num", type=int, default=1)  # Repeat number.

    opt = parser.parse_args()
    return opt


def main(opt, iter_index):
    """Main function."""
    print("[Info] parameters: {}".format(opt))

    if torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    """ prepare model """
    model = eval(opt.model).Model(opt)

    model.to(opt.device)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[Info] Number of parameters: {}".format(num_params))

    """ train or evaluate the model """
    model_save_dir = "models/LongRange/{}/{}/".format(opt.data, opt.predict_step)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_dir += "best_iter{}.pth".format(iter_index)
    if opt.eval:
        best_metrics = evaluate(model, opt, model_save_dir, iter_index)
    else:
        """optimizer and scheduler"""
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()), opt.lr
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt.lr_step)
        best_metrics = train(model, optimizer, scheduler, opt, model_save_dir)

    print("Iteration best metrics: {}".format(best_metrics))
    return best_metrics


if __name__ == "__main__":
    opt = parse_args()
    opt = dataset_parameters(opt, opt.data)
    opt.window_size = eval(opt.window_size)
    iter_num = opt.iter_num
    all_perf = []
    for i in range(iter_num):
        metrics = main(opt, i)
        all_perf.append(metrics)
    all_perf = np.array(all_perf)
    all_perf = all_perf.mean(0)
    print("Average Metrics: {}".format(all_perf))
