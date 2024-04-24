import os
import glob
import numpy as np
import argparse
from gravit.utils.data_loader import load_and_fuse_modalities, load_labels, crop_to_start_and_end, get_segments_and_batch_idxs
from gravit.utils.parser import get_args, get_cfg


if __name__ == "__main__":
    root_data = './data'
    add_multiview = False
    is_multiview = False
    output_features_dir = 'egoexo-omnivore-segmentwise' # name of the output directory in data/features/
    output_annotations_dir = 'egoexo-omnivore-segmentwise' # name of the output directory in data/annotations/

    # Load non-segmented features
    features = 'egoexo-omnivore-ego' # name of the directory in data/features/ that are framewise
    dataset = 'egoexo-omnivore-aria' #  name of corresponding annotation dataset in data/annotations/

    # save the labels
    annotation_type = 'descriptions' #'groundTruth'
    output_dir_path = os.path.join(root_data, f'annotations/{output_annotations_dir}/')
    save_labels = True
    save_features = False

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
    all_ids = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(root_data, f'annotations/{dataset}/groundTruth'))])
    list_splits = sorted(os.listdir(os.path.join(root_data, f'features/{features}')))

    for split in list_splits:
        # Get a list of training video ids
        print(f'Reading splits at {os.path.join(root_data, f"annotations/{dataset}/splits/train.{split}.bundle")}')
        with open(os.path.join(root_data, f'annotations/{dataset}/splits/train.{split}.bundle')) as f:
            train_ids = [os.path.splitext(line.strip())[0] for line in f]

        list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/{split}/*/*.npy')))
        # if is_multiview:
        #     vid = '_'.join(os.path.basename(multiview_data).split('_')[:-1])

  
        for data_file in list_data_files:
            video_id = os.path.splitext(os.path.basename(data_file))[0]

            omnivore_data = load_and_fuse_modalities(data_file, combine_method='concat', 
                                                      dataset=features, sample_rate=1, is_multiview=is_multiview)
            # list_feature_multiview = []
            # for multiview_data_file in list_multiview_data_files:
            #     feature_multiview = np.load(multiview_data_file)
            #     assert feature.shape == feature_multiview.shape, f'feature.shape: {feature.shape}, feature_multiview.shape: {feature_multiview.shape}'
            #     list_feature_multiview.append(feature_multiview)


            # load the labels
            if not os.path.exists(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')):
                video_id_label = video_id.rsplit('_', 1)[0]
            else:
                video_id_label = video_id

            label = load_labels(video_id=video_id_label, actions=actions, root_data=root_data, annotation_dataset=dataset,
                            sample_rate=1, feature=omnivore_data, load_raw=False, load_descriptions=True, verbose=0)  # if load_raw = False, code will downsample the labels to correspond to omnivore features
      
            if len(label) != omnivore_data.shape[0]:
                print(f'Feature-Label Length mismatch for {data_file}')
                print('label length:', len(label))
                print('omnivore_data.shape:', omnivore_data.shape)
                print('---------')
                
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
                    # print(f'Saving {video_id_label} to {os.path.join(save_folder, f"{video_id_label}.txt")}')
                    # save labels to txt file
                    # print(label[-1])
                    with open(os.path.join(save_folder, f'{video_id_label}.txt'), 'w') as f:
                        for l in label:
                            if annotation_type == 'groundTruth':
                                l = reverse[l]
                            f.write(f'{l}\n')



    

            
            