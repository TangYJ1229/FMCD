import torch
from dataset.dataset import MyTestDataset
import tqdm
import cv2
import json
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
from dataset.unnormalized import transform_convert
from sandaintu import pcaplot
from PIL import Image

from models.DA_change import ChangeClassifier as Model
import argparse

def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default="/home/tang/Tang/Dataset/Change_Detection/SYSU-CD/data_256"
    )

    parser.add_argument(
        "--modelpath",
        type=str,
        help="model path",
        default="/home/tang/Tang/code/FMCD/test_net/run_0000/model_82.pth",
    )
    parser.add_argument(
        "--result_save_path",
        type=str,
        help="result path",
        default="/home/tang/Tang/code/FMCD/result/"
    )

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def evaluate(x1, x2, mask, tool4metric):
    # All the tensors on the device:
    x1 = x1.to(device).float()
    x2 = x2.to(device).float()
    mask = mask.to(device).float()

    # Evaluating the model:
    pred, x1_domain, x2_domain, x1_gram, x2_gram = model(x1, x2)

    # Loss gradient descend step:

    # Feeding the comparison metric tool:
    bin_genmask = (pred.to("cpu") > 0.5).detach().numpy().astype(int)
    mask = mask.to("cpu").numpy().astype(int)
    tool4metric.update_cm(pr=bin_genmask, gt=mask)

    return bin_genmask.squeeze(), x1_domain.to("cpu").squeeze(), x2_domain.to("cpu").squeeze()




if __name__ == "__main__":

    # Parse arguments:
    args = parse_arguments()

    # tool for metrics
    # tool_metric = ConfuseMatrixMeter(n_class=2)

    # Initialisation of the dataset
    data_path = args.datapath
    dataset = MyTestDataset(data_path, "test")
    test_loader = DataLoader(dataset, batch_size=4)

    # Initialisation of the model and print model stat
    model = Model()
    modelpath = args.modelpath
    model.load_state_dict(torch.load(modelpath))

    tool4metric = ConfuseMatrixMeter(n_class=2)

    # Set evaluation mode and cast the model to the desidered device
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    # loop to evaluate the model and print the metrics
    tool4metric.clear()
    with torch.no_grad():
        for (reference, testimg), mask, name in tqdm.tqdm(test_loader):
            bin_genmask, x1_domain, x2_domain = evaluate(reference, testimg, mask, tool4metric)

            # x1_domain = x1_domain.transpose(0, 2).transpose(0, 1).numpy()
            # cv2.imwrite(args.result_save_path + "/DS/" + ''.join(name) + '.png', bin_genmask * 255)
            # x1_domain = transform_convert(x1_domain)
            # x2_domain = transform_convert(x2_domain)
            # domain = pcaplot(x1_domain, x2_domain, mask.numpy(), args.result_save_path + "/domain_wolabel/" + ''.join(name) + '.png')

    scores_dictionary = tool4metric.get_scores()
    epoch_result = 'kappa = {}, F1_score = {}, IoU = {}, Pre = {}, Recall = {}, Acc = {}'.format(
        scores_dictionary['kappa'],
        scores_dictionary['F1_1'],
        scores_dictionary['iou_1'],
        scores_dictionary['precision_1'],
        scores_dictionary['recall_1'],
        scores_dictionary['acc'])
    print(epoch_result)
    print()










