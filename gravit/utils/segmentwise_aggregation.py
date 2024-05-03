import os
import glob
import numpy as np
import argparse
from gravit.utils.data_loader import load_and_fuse_modalities, load_labels, crop_to_start_and_end, get_segments_and_batch_idxs
from gravit.utils.parser import get_args, get_cfg


def remove_action_indices(labels, features):    
    # Find the indices of 'action_start' and 'action_end' in labels
    start_indices = [i for i, label in enumerate(labels) if label == actions['action_start']]
    end_indices = [i for i, label in enumerate(labels) if label == actions['action_end']]
    
    # Remove the corresponding indices from labels and features
    labels = [label for i, label in enumerate(labels) if i not in start_indices and i not in end_indices]
    features = [feature for i, feature in enumerate(features) if i not in start_indices and i not in end_indices]
    
    return labels, np.array(features)


if __name__ == "__main__":
    root_data = './data'

    # Load non-segmented features and labels
    features = 'omnivore-trimmed-ego' #'egoexo-brp-level4' #'omnivore-ego' # name of the directory in data/features/ that are framewise
    output_features_dir = 'omnivore-segmentwise-ego' # name of the output directory in data/features/
    features_exo = 'omnivore-trimmed-ego-exo' 
    output_features_dir_exo = 'omnivore-segmentwise-ego-exo'
    save_features = True

    dataset =  'egoexo-regular' #  name of corresponding annotation dataset in data/annotations/
    output_annotations_dir = 'egoexo-segmentwise' # name of the output directory in data/annotations/
    save_labels = False
    annotation_type =  'groundTruth' #'descriptions' #

    crop_start_end = False

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
        if not os.path.exists(os.path.join(output_dir_path, 'mapping.txt')):
            print(f'Output directory does not contain mapping.txt file: {output_dir_path}')
            print('Exiting...')
            exit()

    if save_labels:
        # make dir/clear dir if groundTruth folder already exist
        if os.path.exists(os.path.join(output_dir_path, annotation_type)):
            import shutil
            shutil.rmtree(os.path.join(output_dir_path, annotation_type))
            print(f'Removing existing {annotation_type} directory: os.path.join(output_dir_path, annotation_type)')
        os.makedirs(os.path.join(output_dir_path, annotation_type))

    # Build a mapping from action classes to action ids
    actions = {}
    with open(os.path.join(root_data, f'annotations/{dataset}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            actions[cls] = int(aid)
    reverse = {}
    for k, v in actions.items():
        reverse[v] = k

    # Get a list of all video ids
    all_ids = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(root_data, f'annotations/{dataset}/{annotation_type}'))])
    list_splits = sorted(os.listdir(os.path.join(root_data, f'features/{features}')))
    # list_splits_exo = sorted(os.listdir(os.path.join(root_data, f'features/{features_exo}')))

    for split in list_splits:
        # Get a list of training video ids
        print(f'Reading splits at {os.path.join(root_data, f"annotations/{dataset}/splits/train.{split}.bundle")}')
        with open(os.path.join(root_data, f'annotations/{dataset}/splits/train.{split}.bundle')) as f:
            train_ids = [os.path.splitext(line.strip())[0] for line in f]

        list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/{split}/*/*.npy')))
     
        for data_file in list_data_files:
            video_id = os.path.splitext(os.path.basename(data_file))[0]
            # print(f'Processing {video_id}...')
            # print(data_file)

            omnivore_data = load_and_fuse_modalities(data_file,  sample_rate=1, is_multiview=False)
            
            # load the labels
            if not os.path.exists(os.path.join(root_data, f'annotations/{dataset}/{annotation_type}/{video_id}.txt')):
                video_id_label = video_id.rsplit('_', 1)[0]
            else:
                video_id_label = video_id

            if video_id_label not in all_ids:
                print(f'Could not find {video_id_label} in splits file. Skipping..."')
                continue

            label = load_labels(video_id=video_id_label, actions=actions, root_data=root_data, annotation_dataset=dataset,
                            sample_rate=1, load_descriptions=load_descriptions, verbose=1)  # if load_raw = False, code will downsample the labels to correspond to omnivore features
      
            if len(label) != omnivore_data.shape[0]:
                print(f'Feature-Label Length mismatch for {data_file}')
                print('label length:', len(label))
                print('omnivore_data.shape:', omnivore_data.shape)
                print('---------')
            if 'fair_cooking_05_2' in video_id:
                print(f'Video ID: {video_id} | Label Length: {len(label)} | Feature Shape: {omnivore_data.shape}')

                
            label, batch_idx_designation = get_segments_and_batch_idxs(label)

            if len(batch_idx_designation) != omnivore_data.shape[0]:
                print(f'Batch-Omnivore Length mismatch for {data_file}')
                print('---------')

            batchwise_averages = []
            for batch_idx in np.unique(batch_idx_designation):
                batch_data = omnivore_data[batch_idx_designation == batch_idx]
                batch_average = np.mean(batch_data, axis=0)
                batchwise_averages.append(batch_average)
            batchwise_averages = np.array(batchwise_averages)

            if crop_start_end == True:
                label, batchwise_averages = remove_action_indices(label, batchwise_averages)
                print(f'Video ID: {video_id} | Label Length: {len(label)} | Feature Shape: {batchwise_averages.shape}')

            if 'fair_cooking_05_2' in video_id:
                print(f'Video ID: {video_id} | Label Length: {len(label)} | Feature Shape: {batchwise_averages.shape}')



            if save_features:
                # save the segmentwise features
                path = os.path.join(root_data, f'features/{output_features_dir}/{split}')
                os.makedirs(os.path.join(path, 'train'), exist_ok=True)
                os.makedirs(os.path.join(root_data, f'features/{output_features_dir}/{split}/val'), exist_ok=True)

                if video_id_label in train_ids:
                    np.save(os.path.join(path, 'train', f'{video_id}.npy'), batchwise_averages)
                    # print(f'Saved {video_id} to {os.path.join(path, "train", f"{video_id}.npy")}')
                    # print(f'batchwise_averages.shape: {batchwise_averages.shape}')
                else:
                    np.save(os.path.join(path, 'val', f'{video_id}.npy'), batchwise_averages)

            # print(f'Video ID: {video_id}')
            # print(f'Video ID Label: {video_id_label}')

            if save_labels:
                save_folder = os.path.join(output_dir_path, annotation_type)
                os.makedirs(os.path.join(save_folder), exist_ok=True)
                if os.path.exists(os.path.join(save_folder, f'{video_id_label}.txt')):
                    continue
                else:
                    with open(os.path.join(save_folder, f'{video_id_label}.txt'), 'w') as f:
                        for l in label:
                            if annotation_type == 'groundTruth':
                                l = reverse[l]
                            f.write(f'{l}\n')



    

            
            