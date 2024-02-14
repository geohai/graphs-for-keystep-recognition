import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model
from gravit.datasets import EgoExoOmnivoreDataset
from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds, get_formatted_preds_egoexo_omnivore, get_formatted_preds_framewise
from gravit.utils.eval_tool import get_eval_score, plot_predictions, error_analysis


def evaluate(cfg):
    """
    Run the evaluation process given the configuration
    """
    print(cfg)

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    if 'split' in cfg:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')

    # Prepare the logger
    logger = get_logger(path_result, file_name='eval')
    logger.info(cfg['exp_name'])

    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = build_model(cfg, device)

    val_loader = DataLoader(EgoExoOmnivoreDataset(cfg['split'], validation=True, eval_mode=True), batch_size=cfg['batch_size'], shuffle=False, num_workers=128)

    num_val_graphs = len(val_loader)

    # Load the trained model
    logger.info('Loading the trained model')
    state_dict = torch.load(os.path.join(path_result, 'ckpt_best.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Load the feature files to properly format the evaluation results
    logger.info('Retrieving the formatting dictionary')
    data_dict = get_formatting_data_dict(cfg)

    # Run the evaluation process
    logger.info('Evaluation process started')

    preds_all = []
    # labels_all = []
    with torch.no_grad():
        print(f'Num batches: {len(val_loader)}')
        print(f'Batch size: {cfg["batch_size"]}')
        
        for i, data in enumerate(val_loader, 1):
            x, y, video_id, frame_num = data
            x = x.to(device)
            # y = y.to(device)÷
            # g = None

            logits = model(x)
            logits = logits.squeeze(1)

            # Change the format of the model output
            if 'egoexo-omnivore' in cfg['dataset']:
                preds = get_formatted_preds_framewise(cfg, logits, video_id, frame_num, data_dict)
              
            else:
                # TODO: probably need to fix this
                print('Not implemented yet')
                # preds = get_formatted_preds(cfg, logits, g, data_dict)

            # plot_predictions(cfg, preds)
            preds_all.extend(preds)
            # labels_all.extend(y÷)

            logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')


    # Compute the evaluation score
    # error_analysis(cfg, preds_all)
    logger.info('Computing the evaluation score')
    eval_score = get_eval_score(cfg, preds_all)
    
    
    logger.info(f'{cfg["eval_type"]} evaluation finished: {eval_score}')


if __name__ == "__main__":
    """
    Evaluate the trained model from the experiment "exp_name"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset')
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment', required=True)
    parser.add_argument('--eval_type',     type=str,   help='Type of the evaluation', required=True)

    args = parser.parse_args()

    path_result = os.path.join(args.root_result, args.exp_name)
    if not os.path.isdir(path_result):
        raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

    args.cfg = os.path.join(path_result, 'cfg.yaml')
    print(args.cfg)
    cfg = get_cfg(args)
    evaluate(cfg)
