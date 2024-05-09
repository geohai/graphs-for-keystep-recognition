# This code is based the official ActivityNet repository: https://github.com/activitynet/ActivityNet
# The owner of the official ActivityNet repository: ActivityNet
# Copyright (c) 2015 ActivityNet
# Licensed under The MIT License
# Please refer to https://github.com/activitynet/ActivityNet/blob/master/LICENSE

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import f1_score
from gravit.utils.data_loader import crop_to_start_and_end
import shutil
import torch
from collections import defaultdict
import csv
import decimal
import heapq
from .ava import object_detection_evaluation
from .ava import standard_fields
from mycolorpy import colorlist as mcp
import os
from sklearn.metrics import top_k_accuracy_score
from gravit.utils.data_loader import load_labels, get_segments_and_batch_idxs


def remove_directory(directory):
  if os.path.exists(directory):
    try:
      # forcefully remove directory
      shutil.rmtree(directory, ignore_errors=True)
      print(f"Directory '{directory}' removed successfully.")
    except OSError as e:
      print(f"Error: {directory} : {e.strerror}")


def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != float or recall.dtype != float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Smooth precision to be monotonically decreasing.
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def load_csv(filename, column_names):
  """Loads CSV from the filename using given column names.
  Adds uid column.
  Args:
    filename: Path to the CSV file to load.
    column_names: A list of column names for the data.
  Returns:
    df: A Pandas DataFrame containing the data.
  """
  # Here and elsewhere, df indicates a DataFrame variable.
  df = pd.read_csv(filename, header=None, names=column_names)
  # Creates a unique id from frame timestamp and entity id.
  df["uid"] = (df["frame_timestamp"].map(str) + ":" + df["entity_id"])
  return df


def eq(a, b, tolerance=1e-09):
  """Returns true if values are approximately equal."""
  return abs(a - b) <= tolerance


def merge_groundtruth_and_predictions(df_groundtruth, df_predictions):
  """Merges groundtruth and prediction DataFrames.
  The returned DataFrame is merged on uid field and sorted in descending order
  by score field. Bounding boxes are checked to make sure they match between
  groundtruth and predictions.
  Args:
    df_groundtruth: A DataFrame with groundtruth data.
    df_predictions: A DataFrame with predictions data.
  Returns:
    df_merged: A merged DataFrame, with rows matched on uid column.
  """
  if df_groundtruth["uid"].count() != df_predictions["uid"].count():
    raise ValueError(
        "Groundtruth and predictions CSV must have the same number of "
        "unique rows.")

  if df_predictions["label"].unique() != ["SPEAKING_AUDIBLE"]:
    raise ValueError(
        "Predictions CSV must contain only SPEAKING_AUDIBLE label.")

  if df_predictions["score"].count() < df_predictions["uid"].count():
    raise ValueError("Predictions CSV must contain score value for every row.")

  # Merges groundtruth and predictions on uid, validates that uid is unique
  # in both frames, and sorts the resulting frame by the predictions score.
  df_merged = df_groundtruth.merge(
      df_predictions,
      on="uid",
      suffixes=("_groundtruth", "_prediction"),
      validate="1:1").sort_values(
          by=["score"], ascending=False).reset_index()
  # Validates that bounding boxes in ground truth and predictions match for the
  # same uids.
  df_merged["bounding_box_correct"] = np.where(
      eq(df_merged["entity_box_x1_groundtruth"],
         df_merged["entity_box_x1_prediction"])
      & eq(df_merged["entity_box_x2_groundtruth"],
           df_merged["entity_box_x2_prediction"])
      & eq(df_merged["entity_box_y1_groundtruth"],
           df_merged["entity_box_y1_prediction"])
      & eq(df_merged["entity_box_y2_groundtruth"],
           df_merged["entity_box_y2_prediction"]), True, False)

  if (~df_merged["bounding_box_correct"]).sum() > 0:
    raise ValueError(
        "Mismatch between groundtruth and predictions bounding boxes found at "
        + str(list(df_merged[~df_merged["bounding_box_correct"]]["uid"])))

  return df_merged


def get_all_positives(df_merged):
  """Counts all positive examples in the groundtruth dataset."""
  return df_merged[df_merged["label_groundtruth"] ==
                   "SPEAKING_AUDIBLE"]["uid"].count()


def calculate_precision_recall(df_merged):
  """Calculates precision and recall arrays going through df_merged row-wise."""
  all_positives = get_all_positives(df_merged)

  # Populates each row with 1 if this row is a true positive
  # (at its score level).
  df_merged["is_tp"] = np.where(
      (df_merged["label_groundtruth"] == "SPEAKING_AUDIBLE") &
      (df_merged["label_prediction"] == "SPEAKING_AUDIBLE"), 1, 0)

  # Counts true positives up to and including that row.
  df_merged["tp"] = df_merged["is_tp"].cumsum()

  # Calculates precision for every row counting true positives up to
  # and including that row over the index (1-based) of that row.
  df_merged["precision"] = df_merged["tp"] / (df_merged.index + 1)

  # Calculates recall for every row counting true positives up to
  # and including that row over all positives in the groundtruth dataset.
  df_merged["recall"] = df_merged["tp"] / all_positives

  return np.array(df_merged["precision"]), np.array(df_merged["recall"])


def run_evaluation_asd(predictions, groundtruth):
  """Runs AVA Active Speaker evaluation, returns average precision result."""
  column_names=[
      "video_id", "frame_timestamp", "entity_box_x1", "entity_box_y1",
      "entity_box_x2", "entity_box_y2", "label", "entity_id"
  ]
  df_groundtruth = load_csv(groundtruth, column_names=column_names)
  df_predictions = pd.DataFrame(predictions, columns=column_names+["score"])
  # Creates a unique id from frame timestamp and entity id.
  df_predictions["uid"] = (df_predictions["frame_timestamp"].map(str) + ":" + df_predictions["entity_id"])

  df_merged = merge_groundtruth_and_predictions(df_groundtruth, df_predictions)
  precision, recall = calculate_precision_recall(df_merged)

  return compute_average_precision(precision, recall)


def make_image_key(video_id, timestamp):
  """Returns a unique identifier for a video id & timestamp."""
  return "%s,%.6f" % (video_id, decimal.Decimal(timestamp))


def read_csv(csv_file, class_whitelist=None, capacity=0):
  """Loads boxes and class labels from a CSV file in the AVA format.
  CSV file format described at https://research.google.com/ava/download.html.
  Args:
    csv_file: A file object.
    class_whitelist: If provided, boxes corresponding to (integer) class labels
      not in this set are skipped.
    capacity: Maximum number of labeled boxes allowed for each example. Default
      is 0 where there is no limit.
  Returns:
    boxes: A dictionary mapping each unique image key (string) to a list of
      boxes, given as coordinates [y1, x1, y2, x2].
    labels: A dictionary mapping each unique image key (string) to a list of
      integer class lables, matching the corresponding box in `boxes`.
    scores: A dictionary mapping each unique image key (string) to a list of
      score values lables, matching the corresponding label in `labels`. If
      scores are not provided in the csv, then they will default to 1.0.
    all_keys: A set of all image keys found in the csv file.
  """
  entries = defaultdict(list)
  boxes = defaultdict(list)
  labels = defaultdict(list)
  scores = defaultdict(list)
  all_keys = set()
  reader = csv.reader(csv_file)
  for row in reader:
    assert len(row) in [2, 7, 8], "Wrong number of columns: " + row
    image_key = make_image_key(row[0], row[1])
    all_keys.add(image_key)
    # Rows with 2 tokens (videoid,timestatmp) indicates images with no detected
    # / ground truth actions boxes. Add them to all_keys, so we can score
    # appropriately, but otherwise skip the box creation steps.
    if len(row) == 2:
      continue
    x1, y1, x2, y2 = [float(n) for n in row[2:6]]
    action_id = int(row[6])
    if class_whitelist and action_id not in class_whitelist:
      continue
    score = 1.0
    if len(row) == 8:
      score = float(row[7])
    if capacity < 1 or len(entries[image_key]) < capacity:
      heapq.heappush(entries[image_key], (score, action_id, y1, x1, y2, x2))
    elif score > entries[image_key][0][0]:
      heapq.heapreplace(entries[image_key], (score, action_id, y1, x1, y2, x2))
  for image_key in entries:
    # Evaluation API assumes boxes with descending scores
    entry = sorted(entries[image_key], key=lambda tup: -tup[0])
    for item in entry:
      score, action_id, y1, x1, y2, x2 = item
      boxes[image_key].append([y1, x1, y2, x2])
      labels[image_key].append(action_id)
      scores[image_key].append(score)
  return boxes, labels, scores, all_keys


def read_detections(detections, class_whitelist, capacity=50):
  """
  Loads boxes and class labels from a list of detections in the AVA format.
  """
  entries = defaultdict(list)
  boxes = defaultdict(list)
  labels = defaultdict(list)
  scores = defaultdict(list)
  for row in detections:
    image_key = make_image_key(row[0], row[1])
    x1, y1, x2, y2 = row[2:6]
    action_id = int(row[6])
    if class_whitelist and action_id not in class_whitelist:
      continue
    score = float(row[7])
    if capacity < 1 or len(entries[image_key]) < capacity:
      heapq.heappush(entries[image_key], (score, action_id, y1, x1, y2, x2))
    elif score > entries[image_key][0][0]:
      heapq.heapreplace(entries[image_key], (score, action_id, y1, x1, y2, x2))
  for image_key in entries:
    # Evaluation API assumes boxes with descending scores
    entry = sorted(entries[image_key], key=lambda tup: -tup[0])
    for item in entry:
      score, action_id, y1, x1, y2, x2 = item
      boxes[image_key].append([y1, x1, y2, x2])
      labels[image_key].append(action_id)
      scores[image_key].append(score)
  return boxes, labels, scores


def read_labelmap(labelmap_file):
  """Reads a labelmap without the dependency on protocol buffers.
  Args:
    labelmap_file: A file object containing a label map protocol buffer.
  Returns:
    labelmap: The label map in the form used by the object_detection_evaluation
      module - a list of {"id": integer, "name": classname } dicts.
    class_ids: A set containing all of the valid class id integers.
  """
  labelmap = []
  class_ids = set()
  name = ""
  class_id = ""
  for line in labelmap_file:
    if line.startswith("  name:"):
      name = line.split('"')[1]
    elif line.startswith("  id:") or line.startswith("  label_id:"):
      class_id = int(line.strip().split(" ")[-1])
      labelmap.append({"id": class_id, "name": name})
      class_ids.add(class_id)
  return labelmap, class_ids


def run_evaluation_al(detections, groundtruth, labelmap):
  """
  Runs AVA Actions evaluation, returns mean average precision result
  """
  with open(labelmap, 'r') as f:
    categories, class_whitelist = read_labelmap(f)

  pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

  # Reads the ground truth data.
  with open(groundtruth, 'r') as f:
    boxes, labels, _, included_keys = read_csv(f, class_whitelist)
  for image_key in boxes:
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array(boxes[image_key], dtype=float),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array(labels[image_key], dtype=int),
            standard_fields.InputDataFields.groundtruth_difficult:
                np.zeros(len(boxes[image_key]), dtype=bool)
        })

  # Reads detections data.
  boxes, labels, scores = read_detections(detections, class_whitelist)
  for image_key in boxes:
    if image_key not in included_keys:
      continue
    pascal_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                np.array(boxes[image_key], dtype=float),
            standard_fields.DetectionResultFields.detection_classes:
                np.array(labels[image_key], dtype=int),
            standard_fields.DetectionResultFields.detection_scores:
                np.array(scores[image_key], dtype=float)
        })

  metrics = pascal_evaluator.evaluate()
  return metrics['PascalBoxes_Precision/mAP@0.5IOU']


def get_class_start_end_times(result):
    """
    Return the classes and their corresponding start and end times
    """
    last_class = result[0]
    classes = [last_class]
    starts = [0]
    ends = []

    for i, c in enumerate(result):
        if c != last_class:
            classes.append(c)
            starts.append(i)
            ends.append(i)
            last_class = c

    ends.append(len(result)-1)

    return classes, starts, ends


def compare_segmentation(pred, label, th):
    """
    Temporally compare the predicted and ground-truth segmentations
    """

    pc, ps, pe = get_class_start_end_times(pred)
    lc, ls, le = get_class_start_end_times(label)

    tp = 0
    fp = 0
    matched = [0]*len(lc)
    for i in range(len(pc)):
        inter = np.minimum(pe[i], le) - np.maximum(ps[i], ls)
        union = np.maximum(pe[i], le) - np.minimum(ps[i], ls)
        # print(union)
        IoU = (inter/union) * [pc[i] == lc[j] for j in range(len(lc))]

        best_idx = np.array(IoU).argmax()
        if IoU[best_idx] >= th and not matched[best_idx]:
            tp += 1
            matched[best_idx] = 1
        else:
            fp += 1

    fn = len(lc) - sum(matched)

    return tp, fp, fn


def get_eval_score_naive(path_annts, cfg, preds):
    total = 0
    correct = 0
    
    y_true = []
    y_preds = []    
    count = 0
    df = pd.DataFrame(columns=['true', 'pred'])

    for (video_id, frame_num, pred) in preds:
        # Get a list of ground-truth action labels
        try:
          with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f:
            label = [line.strip() for line in f]
            label = label[frame_num]
        except:
          video_id = video_id.rsplit('_', 1)[0]
          # print(video_id)
          with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f:
              label = [line.strip() for line in f]
              label = label[frame_num]

        if cfg['crop']:
           _, label = crop_to_start_and_end(feature=None, label=label)

        # Append labels and predictions to lists
        y_true.append(label)
        y_preds.append(pred)

        # gather data in df to write to csv
        count += 1
        new_row = pd.DataFrame({'true': [label], 'pred': [pred]})
        df = pd.concat([df, new_row], ignore_index=True)

        # save every 2000 rows to csv
        if count % 2000 == 0:
          df.to_csv(f'results/{cfg["exp_name"]}/csv/results_{count}.csv', index=False)
          df = pd.DataFrame(columns=['true', 'pred'])

        total += 1

        if pred == label:
          correct += 1

    acc = correct/total
    str_score = f'(Acc) {acc*100:.2f}%'

    # TODO: Modify this f1 score computation to use compare_segmentation()
    f1 = f1_score(y_true, y_preds, average='micro')
    str_score += f', (F1) {f1*100:.2f}%'
    
    return str_score


def get_eval_score(cfg, preds):
    """
    Compute the evaluation score
    """

    # Path to the annotations
    path_annts = os.path.join(cfg['root_data'], 'annotations')

    eval_type = cfg['eval_type']
    if eval_type == 'AVA_ASD':
        groundtruth = os.path.join(path_annts, 'ava_activespeaker_val_v1.0.csv')
        score = run_evaluation_asd(preds, groundtruth)
        str_score = f'{score*100:.2f}%'
    elif eval_type == 'AVA_AL':
        groundtruth = os.path.join(path_annts, 'ava_val_v2.2.csv')
        labelmap = os.path.join(path_annts, 'ava_action_list_v2.2_for_activitynet_2019.pbtxt')
        score = run_evaluation_al(preds, groundtruth, labelmap)
        str_score = f'{score*100:.2f}%'
    elif eval_type == 'AS':
        # Create new csv results directory
        remove_directory(f'results/{cfg["exp_name"]}/csv')
        os.makedirs(f'results/{cfg["exp_name"]}/csv')
        
        if 'mlp' in cfg['graph_name']:
           return get_eval_score_naive(path_annts, cfg, preds)
    
        total = 0
        correct = 0
        threshold = [0.1, 0.25, 0.5]
        tp, fp, fn = [0]*len(threshold), [0]*len(threshold), [0]*len(threshold)

        for (video_id, pred) in preds:
          # Get a list of ground-truth action labels
          with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f:
            label = [line.strip() for line in f]

          if len(label) != len(pred):
            print(f'len(pred): {len(pred)} | len(label): {len(label)}')
            print(f'Length of pred and label do not match for {video_id}')
            label = label[:len(pred)]
            
          # write results of each video to csv: pred vs true labels
          pd.DataFrame(data=zip(label, pred), columns=['true', 'pred']).to_csv(f'results/{cfg["exp_name"]}/csv/results_{video_id}.csv', index=False)

          total += len(label)

          for i, lb in enumerate(label):
              if pred[i] == lb:
                correct += 1
          
          for i, th in enumerate(threshold):
              tp_, fp_, fn_ = compare_segmentation(pred, label, th)
              tp[i] += tp_
              fp[i] += fp_
              fn[i] += fn_
        
        acc = correct/total
        str_score = f'(Acc) {acc*100:.2f}%'
        for i, th in enumerate(threshold):
            pre = tp[i] / (tp[i]+fp[i])
            rec = tp[i] / (tp[i]+fn[i])
            if pre+rec == 0:
              f1 = 0
            else:
              f1 = np.nan_to_num(2*pre*rec / (pre+rec))
            str_score += f', (F1@{th}) {f1*100:.2f}%'


    elif eval_type == 'KR': # keystep recognition for egoexo benchmark task
        # Create new csv results directory
        remove_directory(f'results/{cfg["exp_name"]}/csv')
        os.makedirs(f'results/{cfg["exp_name"]}/csv')

        # if 'mlp' in cfg['graph_name']:
        #    return get_eval_score_naive(path_annts, cfg, preds)
    
        total = 0
        correct = 0
        threshold = [0.1, 0.25, 0.5]
        tp, fp, fn = [0]*len(threshold), [0]*len(threshold), [0]*len(threshold)

        for (video_id, pred) in preds:
          # Get a list of ground-truth action labels
          with open(os.path.join(path_annts, f'{cfg["annotations_dataset"]}/groundTruth/{video_id}.txt')) as f:
            label = [line.strip() for line in f]
            #print(f'Loaded path: {path_annts}/{cfg["dataset"]}/groundTruth/{video_id}.txt')


          if len(label) != len(pred):
            print(f'len(pred): {len(pred)} | len(label): {len(label)}')
            print(f'Length of pred and label do not match for {video_id}')
            label = label[:len(pred)]
            
          # write results of each video to csv: pred vs true labels
          path = f"results/{cfg['exp_name']}/csv/results_{video_id}.csv"
          pd.DataFrame(data=zip(label, pred), columns=['true', 'pred']).to_csv(path, index=False)
          # print(f'Predictions saved to {path}')
          total += len(label)

          for i, lb in enumerate(label):
              if pred[i] == lb:
                correct += 1

          for i, th in enumerate(threshold):
              tp_, fp_, fn_ = compare_segmentation(pred, label, th)
              tp[i] += tp_
              fp[i] += fp_
              fn[i] += fn_


        acc = correct/total
        str_score = f'(Acc) {acc*100:.2f}%'
        for i, th in enumerate(threshold):
            pre = tp[i] / (tp[i]+fp[i])
            rec = tp[i] / (tp[i]+fn[i])
            if pre+rec == 0:
              f1 = 0
            else:
              f1 = np.nan_to_num(2*pre*rec / (pre+rec))
            str_score += f', (F1@{th}) {f1*100:.2f}%'
      
    return str_score


def get_top1_accuracy(true, preds):
    return top_k_accuracy_score(true, preds, k=1, normalize=False)


def plot_ground_truth(cfg, video_id, actions, reverse):
  # Define the color map
    cmap = 'tab20'
    colors = mcp.gen_color(cmap="tab20",n=len(actions))

    # Path to the annotations
    path_annts = os.path.join(cfg['root_data'], 'annotations')

    eval_type = cfg['eval_type']
  
    if eval_type == 'AS':
            # Get a list of ground-truth action labels
          with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f:
              label = [line.strip() for line in f]

          label = [actions[i] for i in label]

          # plot each session
          f, ax = plt.subplots(1, 1, figsize=(18, 3))

          method = 'Ground Truth'
          method_data = label

          # Plot each row
          ax.imshow([method_data], aspect='auto', cmap=cmap)
          ax.set_yticks([0])
          ax.set_yticklabels([method])
          ax.set_xticks([])  # Remove x-ticks if not needed

          # Set the legend
          # You need to create a patch (proxy artist) for each color in the colormap
        
          legend_patches = [Patch(color=colors[i], label=reverse[i]) for i in range(len(colors))]
          plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

          # Adjust layout
          plt.tight_layout()
          
          plt.show()


def plot_predictions(cfg, preds):
    """
    Plot action segmentation predictions vs ground truth.
    """

    # Build a mapping from action classes to action ids
    actions = {}
    reverse = {}
    with open(os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            actions[cls] = int(aid)
    with open(os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            reverse[int(aid)] = cls

  
    # Define the color map
    cmap = 'twilight' #tab20'
    colors = mcp.gen_color(cmap="tab20",n=len(actions))

    # Path to the annotations
    path_annts = os.path.join(cfg['root_data'], 'annotations')

    eval_type = cfg['eval_type']
    if eval_type == 'AVA_ASD':
        print('incomplete function')
    elif eval_type == 'AVA_AL':
        print('incomplete function')
    elif eval_type == 'AS' or eval_type == 'KR':
       for video_id, pred in preds:
            # Get a list of ground-truth action labels
            with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f:
                label = [line.strip() for line in f]

            pred = [actions[i] for i in pred]
            label = [actions[i] for i in label]

            # plot each session
            f, axs = plt.subplots(2, 1, figsize=(18, 6))

            methods = ['Ground Truth', 'Predicted']
            data = [label, pred]

            # Plot each row
            for ax, method_data, method in zip(axs, data, methods):
                ax.imshow([method_data], aspect='auto', cmap=cmap)
                ax.set_yticks([0])
                ax.set_yticklabels([method])
                ax.set_xticks([])  # Remove x-ticks if not needed

            # Set the legend
            # You need to create a patch (proxy artist) for each color in the colormap
            # Adjust layout
            plt.tight_layout()
            plt.savefig(f'results/{cfg["exp_name"]}/plots/action_segmentation_comparison_{video_id}.png', bbox_inches='tight')
            # plt.show()

            legend_patches = [Patch(color=colors[i], label=reverse[i]) for i in range(len(colors))]
            plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

            # Create a separate legend figure and axis
            legend_fig, legend_ax = plt.subplots(figsize=(5, 10))  # Adjust the size as needed

            # Extract the legend from the main figure
            # Create a dummy plot in the legend axis
            legend_ax.plot([], [], ' ', label="Legend") 
            legend_ax.legend(handles=legend_patches, loc='upper left')
            legend_ax.axis('off')  # Turn off the axis in the legend figure
            # legend_ax.add_artist(legend)

            # Save the legend figure
            legend_fig.savefig(f'results/{cfg["exp_name"]}/plots/legend_figure.png', bbox_inches='tight')

    return 


def error_analysis(cfg, preds):
    """
    Generate confusion matrix plot.
    """
    # Path to the annotations
    # Build a mapping from action classes to action ids
    actions = {}
    reverse = {}
    with open(os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            actions[cls] = int(aid)
    with open(os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            reverse[int(aid)] = cls


    ##### Compute Proportion of incorrect predictions and save to csv ######
    path_annts = os.path.join(cfg['root_data'], 'annotations')

    eval_type = cfg['eval_type']
    if eval_type == 'AVA_ASD':
        print('incomplete function')
    elif eval_type == 'AVA_AL':
        print('incomplete function')
    elif eval_type == 'AS' or eval_type == 'KR':
     
      total = None
      for i, (video_id, pred) in enumerate(preds):
          print(video_id)
          # Get a list of ground-truth action labels
          with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f: 
              label = [line.strip() for line in f]

          if 'mlp' in cfg['graph_name']:
             frame_num = preds[i][1]
             print(frame_num)
             label = label[frame_num]

          data = pd.DataFrame(data=zip(label, pred), columns=['true', 'pred'])
          proportion_correct = data.groupby('true').apply(lambda x: (x.true == x.pred).sum()/x.true.count()).reset_index(drop=False).rename(columns={'true': 'action', 0: video_id}) # num of correct predictions over total predictions

          if total is None:
              total = proportion_correct
          else:
              total = total.merge(proportion_correct, on='action')
            

      total.to_csv(f'results/{cfg["exp_name"]}/individual_results.csv', index=False)
      total.set_index('action', inplace=True)
      means = total.mean(axis=1).reset_index(drop=False).rename(columns={0: 'mean'})
      stds = total.std(axis=1).reset_index(drop=False).rename(columns={0: 'standard deviation'})
      
      stats = means.merge(stds, on='action')
      stats.to_csv(f'results/{cfg["exp_name"]}/aggregated_results.csv', index=False)


      #####  PLOT FALSE PREDICTIONS #####
      # now for each class, get the counts of incorrect predictions for each other class
      incorrect_counts = np.zeros([len(actions), len(actions)])
      label_idx_map = actions
      for i, (video_id, pred) in enumerate(preds):
          
          if (video_id + '.txt') not in os.listdir(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth')):
              print(f'Skipping {video_id}')
              continue
          
          # Get a list of ground-truth action labels
          with open(os.path.join(path_annts, f'{cfg["dataset"]}/groundTruth/{video_id}.txt')) as f: 
              label = [line.strip() for line in f]

          if 'mlp' in cfg['graph_name']:
              frame_num = preds[i][1]
              label = label[frame_num]

          data = pd.DataFrame(data=zip(label, pred), columns=['true', 'pred']).set_index('true')

          # iterate over each class
          for i in data.index:
              incorrect_label = data.loc[i, 'pred'].values[0]  
              incorrect_counts[label_idx_map[i], label_idx_map[incorrect_label]] += 1
        

      df = pd.DataFrame(incorrect_counts)
      df.index= list(label_idx_map.keys())
      df.columns = list(label_idx_map.keys())
      df = df.div(df.sum(axis=0)) # compute proportion of incorrect predictions for each class

      # plot heatmap of incorrect predictions
      sns.heatmap(df, annot=False)
      plt.ylabel('True Label')
      plt.xlabel('Predicted Label')
      plt.title('Proportion of Predicted Labels for each Class')

          # Note to self -> aggregate across
      plt.savefig(f'results/{cfg["exp_name"]}/heatmap_incorrect_predictions', bbox_inches='tight')

    return
