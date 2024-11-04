import os
import glob
import numpy as np
import argparse
import sys
import pandas as pd
sys.path.append('/home/juro4948/gravit/data/egoexo4d')
sys.path.append('/home/juro4948/segment_utils')

from preprocess_utils import *
from gravit.utils.data_loader import load_labels, load_labels_raw, get_segment_labels_by_batch_idxs, load_batch_indices #get_segments_and_batch_idxs
from gravit.utils.parser import get_args, get_cfg
from frame_to_segment import load_annotations, get_frames_for_segment, get_num_segments

"""
Script to convert framewise features and labels to segmentwise features and labels.
"""



if __name__ == "__main__":
    root_data = './data'

    load_annotations()


    # Load non-segmented features and labels
    # features = 'omnivore-trimmed-ego' #'egoexo-brp-level4' #'omnivore-ego' # name of the directory in data/features/ that are framewise
    # output_features_dir = 'omnivore-segmentwise-ego' # name of the output directory in data/features/
    # egocentric = True
    
    features = 'omnivore-trimmed-ego-exo'  #-exo
    output_features_dir = 'omnivore-segmentwise-ego-exo' #-exo
    egocentric = False

    save_features = True
 
 
    dataset =  'egoexo-regular-all-categories' #'egoexo-regular' #  name of corresponding annotation dataset in data/annotations/
    output_annotations_dir = 'egoexo-segmentwise-all-categories' # name of the output directory in data/annotations/
    save_labels = False
    annotation_type =  'groundTruth' #'descriptions' #

    batch_idx_path = f"/local/juro4948/data/egoexo4d/gravit_data/data_new/annotations/{dataset}/batch_idx/"


    ##########################################
        

    output_dir_path = os.path.join(root_data, f'annotations/{output_annotations_dir}/')
    if annotation_type == 'descriptions':
        load_descriptions = True
    else:
        load_descriptions = False


    if not os.path.exists(output_dir_path):
        print(f'Output directory does not exist: {output_dir_path}')
        user_input = input(f"Create the output directory named {output_annotations_dir} and it must contain the splits/ and mapping.txt files. Is this done? (Y/N) ")
        if user_input.lower() != 'y':
            print('Exiting...')
            exit()
        if not os.path.exists(output_dir_path):
            print(f'Output directory does not exist: {output_dir_path}')
            print('Exiting...')
            exit()
        if not os.path.exists(os.path.join(output_dir_path, 'splits')):
            print(f'Output directory does not contain splits/ directory: {output_dir_path}')
            print('Exiting...')
            exit()


    if save_labels:
        # make new directory if groundTruth folder already exist
        if os.path.exists(os.path.join(output_dir_path, annotation_type)):
            import shutil
            shutil.rmtree(os.path.join(output_dir_path, annotation_type))
            print(f'Removing existing {annotation_type} directory: os.path.join(output_dir_path, annotation_type)')
        os.makedirs(os.path.join(output_dir_path, annotation_type))
    

        # load ground truth long-tailed classes
        df = pd.read_csv('~/gravit/GraVi-T/data/annotations/label_mapping.csv')
        step_names = df['step_name'].values
        step_ids = df['step_unique_id'].values.astype(str)
        converted_names = []
        for name in step_names:
            words = name.split()
            converted_label = "_".join([w.lower() for w in words])
            converted_names.append(converted_label)
        label_ids = df['label_id'].values.astype(str)

        mapping = dict(zip(step_ids, converted_names))
        reverse_mapping = dict(zip(converted_names, step_ids))
        reverse = dict(zip(step_ids, label_ids))
        actions = dict(zip(label_ids, converted_names))

        # save mapping
        with open(os.path.join(output_dir_path, 'mapping.txt'), 'w') as f:
            for k, v in actions.items():
                f.write(f'{k} {v}\n')

    if save_features:
        # make new directory if groundTruth folder already exist
        if os.path.exists(os.path.join(root_data, f'features/{output_features_dir}')):
            import shutil
            shutil.rmtree(os.path.join(root_data, f'features/{output_features_dir}'))
            print(f'Removing existing {output_features_dir} directory: os.path.join(root_data, f"features/{output_features_dir}")')
        os.makedirs(os.path.join(root_data, f'features/{output_features_dir}'))


    # Get a list of all video ids
    all_ids = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(root_data, f'annotations/{dataset}/{annotation_type}'))])
    list_splits = sorted(os.listdir(os.path.join(root_data, f'features/{features}')))

    for split in list_splits:
        # Get a list of training video ids
        print(f'Reading splits at {os.path.join(root_data, f"annotations/{dataset}/splits/train.{split}.bundle")}')
        with open(os.path.join(root_data, f'annotations/{dataset}/splits/train.{split}.bundle')) as f:
            train_ids = [os.path.splitext(line.strip())[0] for line in f]

        list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/{split}/*/*.npy')))
     
        for data_file in list_data_files:
            video_id = os.path.splitext(os.path.basename(data_file))[0]

            # load the features
            omnivore_data = load_features(data_file)
            
            # load the labels
            if video_id not in all_ids:
                video_id_label = '_'.join(video_id.split('_')[:-1])
                if video_id_label not in all_ids:
                    raise ValueError(f'Could not find {video_id_label} in annotations file"')


            # load unique step ids
            label = load_labels_raw(video_id=video_id_label, root_data=root_data, annotation_dataset=dataset)  

            assert len(label) == omnivore_data.shape[0], f'Feature-Label Length mismatch for {data_file}'

    
            # Aggregate frame labels to segment labels by taking the mode
            num_segments = get_num_segments(video_id_label)

            batch_idx_designation = load_batch_indices(batch_idx_path, video_id_label)
            labels = get_segment_labels_by_batch_idxs(label, batch_idx_designation)

            # check that lengths align
            assert len(np.unique(batch_idx_designation)) == num_segments, f'Batch Index Length mismatch for {data_file}'
            assert len(labels) == num_segments, f'Label Length mismatch for {data_file}'
            # print(f'Batch Index Length: {len(np.unique(batch_idx_designation))} | Num Segments: {num_segments} | Label Length: {len(labels)}')


            label = []
            for segment in labels:
                # append mode of the segment
                label.append(np.argmax(np.bincount(segment)))


            assert len(batch_idx_designation) == omnivore_data.shape[0], f'BatchIndex-Omnivore Feature Length mismatch for {data_file}'

            batchwise_averages = []
            for batch_idx in np.unique(batch_idx_designation):
                batch_data = omnivore_data[batch_idx_designation == batch_idx]
                batch_average = np.mean(batch_data, axis=0)
                batchwise_averages.append(batch_average)
            batchwise_averages = np.array(batchwise_averages)


            assert batchwise_averages.shape[0] == num_segments, f'Number of segments and features do not match for {video_id}: {num_segments} vs {batchwise_averages.shape[0]}'


            ### REMOVE LONG TAILED CLASSES ###
            df = pd.read_csv('~/gravit/GraVi-T/data/annotations/label_mapping.csv')
            step_ids = df['step_unique_id'].values.astype(str)
            label = [str(l) for l in label]

            # batchwise_averages = batchwise_averages[~np.isin(label, step_ids)]
            batchwise_averages = batchwise_averages[np.isin(label, step_ids)]
            label = [l for l in label if l in step_ids]
            num_segments = len(label)
            if num_segments == 0:
                print(f'No segments left after removing long-tailed classes for {video_id}')
                continue
            ############


            assert batchwise_averages.shape[0] == num_segments, f'Number of segments and features do not match for {video_id}: {num_segments} vs {batchwise_averages.shape[0]}'


            # if remove_long_tailed_segments:
            #     with open('classes_to_remove.txt', 'r') as f:
            #         classes_to_remove = f.read().splitlines()
            #     # filter out the long-tailed classes
            #     batchwise_averages = batchwise_averages[~np.isin(label, classes_to_remove)]
            #     label = [l for l in label if l not in classes_to_remove]
            #     num_segments = len(label)




            # save the segment features
            if save_features:
                path = os.path.join(root_data, f'features/{output_features_dir}/{split}')
                os.makedirs(os.path.join(path, 'train'), exist_ok=True)
                os.makedirs(os.path.join(root_data, f'features/{output_features_dir}/{split}/val'), exist_ok=True)

                if not egocentric:
                    if video_id_label in train_ids:
                        np.save(os.path.join(path, 'train', f'{video_id}.npy'), batchwise_averages)
                        # print(f'Saving {video_id} to {os.path.join(path, "train", f"{video_id}.npy")}')
                    else:
                        np.save(os.path.join(path, 'val', f'{video_id}.npy'), batchwise_averages)
                        # print(f'Saving {video_id} to {os.path.join(path, "val", f"{video_id}.npy")}')

                else:
                    if video_id_label in train_ids:
                        np.save(os.path.join(path, 'train', f'{video_id_label}_0.npy'), batchwise_averages)
                        # print(f'Saving {video_id} to {os.path.join(path, "train", f"{video_id_label}.npy")}')
                    else:
                        np.save(os.path.join(path, 'val', f'{video_id_label}_0.npy'), batchwise_averages)
                        # print(f'Saving {video_id} to {os.path.join(path, "val", f"{video_id_label}.npy")}')


            # save the segment labels
            if save_labels:
                save_folder = os.path.join(output_dir_path, annotation_type)
                os.makedirs(os.path.join(save_folder), exist_ok=True)
                # if os.path.exists(os.path.join(save_folder, f'{video_id_label}.txt')):
                #     # continue
                #     pass
                # # else:
                # print(reverse)
                with open(os.path.join(save_folder, f'{video_id_label}.txt'), 'w') as f:
                    for l in label:
                        if annotation_type == 'groundTruth':
                            # unique step id to label id
                            l = mapping[str(l)]
                        f.write(f'{l}\n')

                # print(f'Saving {video_id} to {os.path.join(save_folder, f"{video_id_label}.txt")}')



    

            
            