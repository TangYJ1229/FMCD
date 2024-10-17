import argparse
import os
import shutil
import dataset.dataset as dtset
import torch
import numpy as np
import random
import sys
from metrics.metric_tool import ConfuseMatrixMeter
from models.FMCD import ChangeClassifier as Model
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mt_loss import MultiTaskLoss
from torch import nn


def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        default="/home/tang/Tang/Dataset/Change_Detection/WH/data_256",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--log-path",
        default="/home/tang/Tang/code/FMCD/test_net/",
        type=str,
        help="log path",
    )

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.log_path):
        os.mkdir(parsed_arguments.log_path)
    
    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )
    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )

    return parsed_arguments


def train(dataset_train, dataset_val, model, criterion, optimizer, scheduler, logpath, writer, epochs):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = model.to(device)

    tool4metric = ConfuseMatrixMeter(n_class=2)

    def evaluate(x1, x2, mask):
        # All the tensors on the device:
        x1 = x1.to(device).float()
        x2 = x2.to(device).float()
        mask = mask.to(device).float()

        # Evaluating the model:
        pred, x1_domain, x2_domain, x1_gram, x2_gram = model(x1, x2)

        # Loss gradient descend step:
        total_loss, dloss, closs = criterion(x1, x2, mask, x1_domain, x2_domain, x1_gram, x2_gram, pred)

        # Feeding the comparison metric tool:
        bin_genmask = (pred.to("cpu") > 0.5).detach().numpy().astype(int)
        mask = mask.to("cpu").numpy().astype(int)
        tool4metric.update_cm(pr=bin_genmask, gt=mask)

        return total_loss

    def training_phase(epc):
        tool4metric.clear()
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        loop = tqdm(dataset_train, file=sys.stdout)
        for (reference, testimg), mask, name in loop:
            # Reset the gradients:

            optimizer.zero_grad()
            # Loss gradient descend step:
            it_loss = evaluate(reference, testimg, mask)
            it_loss.backward()
            optimizer.step()

            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        #########
        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU class change/epoch", scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1 class change/epoch", scores_dictionary["F1_1"], epc)
        writer.flush()

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        test_loss = []
        loop = tqdm(dataset_val, file=sys.stdout)
        with torch.no_grad():
            for (reference, testimg), mask, name in loop:
                epoch_loss_eval += evaluate(reference, testimg, mask).to("cpu").numpy()

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU_val class change/epoch", scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1_val class change/epoch", scores_dictionary["F1_1"], epc)
        epoch_result = 'kappa = {}, F1_score = {}, IoU = {}, Pre = {}, Recall = {}, Acc = {}'.format(scores_dictionary['kappa'],
                                                                                                     scores_dictionary['F1_1'],
                                                                                                     scores_dictionary['iou_1'],
                                                                                                     scores_dictionary['precision_1'],
                                                                                                     scores_dictionary['recall_1'],
                                                                                                     scores_dictionary['acc'])
        print(epoch_result)
        print()

        return scores_dictionary["F1_1"]

    score = 0

    for epc in range(epochs):
        training_phase(epc)
        f1 = validation_phase(epc)

        if f1 > score:
            score = f1
            torch.save(model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc)))

        # scheduler step 
        scheduler.step()


def run():

    # set the random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.log_path)

    # Inizialitazion of dataset and dataloader:
    trainingdata = dtset.MyDataset(args.datapath, "train")
    validationdata = dtset.MyDataset(args.datapath, "val")
    data_loader_training = DataLoader(trainingdata, batch_size=8, shuffle=True)
    data_loader_val = DataLoader(validationdata, batch_size=4, shuffle=True)

    # Initialize the model
    model = Model()
    restart_from_checkpoint = False
    model_path = None
    if restart_from_checkpoint:
        model.load_state_dict(torch.load(model_path))
        print("Checkpoint succesfully loaded")

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        # print (nom, param.data.shape)
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print()
    print("Number of model parameters {}".format(parameters_tot))
    print()

    # define the loss function for the model training.
    criterion = MultiTaskLoss("learned", model.get_loss_params())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                    weight_decay=0.01, amsgrad=False)

    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # copy the configurations
    _ = shutil.copytree(
        "./models",
        os.path.join(args.log_path, "models"),
    )

    train(
        data_loader_training,
        data_loader_val,
        model,
        criterion,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=200,
    )
    writer.close()



if __name__ == "__main__":
    run()

