import os
from utils import *
import mxnet as mx

# base_path = '/home/chocobo/datasets/AI4Boundaries2/sentinel2'
base_path = '/decode/examples/input'
input_path = f'{base_path}/images'
label_path = f'{base_path}/masks'
input_list = []
label_list = []
total = 0

for root, dirs, files in os.walk(input_path):
    l = len(files)
    total += l
    if l:
        country = root.split('/')[-1]
        print(root, l, country)
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i, file in enumerate(files):
        file_path = f'{root}/{file}'
        # country = file.split('_')[0]
        number = file.split('_')[1]
        print(i, file, country, number)

        # Build input file
        new_inputs = file_to_nparray(file_path)
        input_list = input_list + new_inputs
        # Get label file
        label_file_name = f'{country}_{number}_S2label_10m_256.tif'
        label_file_path = f'{label_path}/{country}/{label_file_name}'
        new_labels = label_to_nparray(label_file_path, len(new_inputs))
        label_list = label_list + new_labels

        if not os.path.isfile(label_file_path):
            print(f'Error: not found {label_file_path}')
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    

    input_ndarray = mx.nd.array(input_list)
    label_ndarray = mx.nd.array(label_list)

    # Save dataset input images
    mxnet_dataset_path = f'{base_path}/mxnet_dataset'
    mx.nd.save(f'{mxnet_dataset_path}/LU_input_ndarray.mat', input_ndarray)
    mx.nd.save(f'{mxnet_dataset_path}/LU_label_ndarray.mat', label_ndarray)

print('Total files: ', total)