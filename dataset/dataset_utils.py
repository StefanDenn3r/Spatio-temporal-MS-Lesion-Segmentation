import os
import pickle
from collections import defaultdict, OrderedDict
from enum import Enum
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage


class Modalities(Enum):
    FLAIR = 'flair'
    MPRAGE = 'mprage'
    PD = 'pd'
    T2 = 't2'
    T1W = 't1w'


class Phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Views(Enum):
    AXIAL = 0
    SAGITTAL = 1
    CORONAL = 2


class Mode(Enum):
    STATIC = 'static'
    LONGITUDINAL = 'longitudinal'


class Dataset(Enum):
    ISBI = 'isbi'
    INHOUSE = 'inhouse'


class Evaluate(Enum):
    TRAINING = 'training'
    TEST = 'test'


def retrieve_data_dir_paths(data_dir, evaluate: Evaluate, phase, preprocess, val_patients, mode, view=None):
    empty_slices = None
    non_positive_slices = None
    if preprocess:
        print('Preprocessing files...')
        empty_slices, non_positive_slices = preprocess_files(data_dir, phase, evaluate)
        print('Files preprocessed.')
    if mode == Mode.LONGITUDINAL:
        data_dir_paths = retrieve_paths_longitudinal(get_patient_paths(data_dir, evaluate, phase)).items()
    else:
        data_dir_paths = retrieve_paths_static(get_patient_paths(data_dir, evaluate, phase)).items()
    data_dir_paths = OrderedDict(sorted(data_dir_paths))
    _data_dir_paths = []
    patient_keys = [key for key in data_dir_paths.keys()]
    if phase == Phase.TRAIN:
        for val_patient in val_patients[::-1]:
            patient_keys.remove(patient_keys[val_patient])

        for patient in patient_keys:
            _data_dir_paths += data_dir_paths[patient]
    elif phase == Phase.VAL:
        for val_patient in val_patients:
            _data_dir_paths += data_dir_paths[patient_keys[val_patient]]
    else:
        for patient in patient_keys:
            _data_dir_paths += data_dir_paths[patient]

    if view:
        _data_dir_paths = list(filter(lambda path: int(path.split(os.sep)[-2]) == view.value, _data_dir_paths))
    if phase == Phase.TRAIN or phase == Phase.VAL:
        _data_dir_paths = retrieve_filtered_data_dir_paths(data_dir, phase, _data_dir_paths, empty_slices, non_positive_slices,
                                                           mode, val_patients, view)
    return _data_dir_paths


def preprocess_files(root_dir, phase, evaluate, base_path='data'):
    patients = list(filter(lambda name: (evaluate.value if phase == Phase.TEST else Evaluate.TRAINING.value) in name, os.listdir(root_dir)))
    empty_slices = []
    non_positive_slices = []
    i_patients = len(patients) + 1
    for patient in patients:
        print(f'Processing patient {patient}')
        patient_path = os.path.join(root_dir, patient)
        if os.path.exists(os.path.join(patient_path, base_path)):
            continue
        patient_data_path = os.path.join(patient_path, 'preprocessed', patient)
        patient_label_path = os.path.join(patient_path, 'masks', patient)

        for modality in list(Modalities):
            mod, value = modality.name, modality.value
            for timestep in range(10):
                data_path = f'{patient_data_path}_0{timestep + 1}_{value}_pp.nii'
                if not os.path.exists(data_path):
                    continue
                rotated_data = transform_data(data_path)
                normalized_data = (rotated_data - np.min(rotated_data)) / (np.max(rotated_data) - np.min(rotated_data))
                label_path = f'{patient_label_path}_0{timestep + 1}_mask1.nii'
                if os.path.exists(label_path):
                    rotated_labels = transform_data(label_path)
                else:
                    rotated_labels = np.zeros(normalized_data.shape)

                # create slices through all views
                temp_empty_slices, temp_non_positive_slices = create_slices(normalized_data, rotated_labels,
                                                                            os.path.join(patient_path, base_path, str(timestep)), value)
                empty_slices += temp_empty_slices
                non_positive_slices += temp_non_positive_slices

        i_patients += 1
    return empty_slices, non_positive_slices


def transform_data(data_path):
    data = nib.load(data_path).get_data()
    x_dim, y_dim, z_dim = data.shape
    max_dim = max(x_dim, y_dim, z_dim)
    x_pad = get_padding(max_dim, x_dim)
    y_pad = get_padding(max_dim, y_dim)
    z_pad = get_padding(max_dim, z_dim)
    padded_data = np.pad(data, (x_pad, y_pad, z_pad), 'constant')
    rotated_data = scipy.ndimage.rotate(scipy.ndimage.rotate(padded_data, 90, axes=(1, 2)), -90, axes=(0, 1))
    return rotated_data


def get_padding(max_dim, current_dim):
    diff = max_dim - current_dim
    pad = diff // 2
    if diff % 2 == 0:
        return pad, pad
    else:
        return pad, pad + 1


def create_slices(data, label, timestep_path, modality):
    empty_slices = []
    non_positive_slices = []
    for view in list(Views):
        name, axis = view.name, view.value
        temp_data = np.moveaxis(data, axis, 0)
        temp_labels = np.moveaxis(label, axis, 0)
        for i, (data_slice, label_slice) in enumerate(zip(temp_data, temp_labels)):
            path = os.path.join(timestep_path, str(axis), f'{i:03}')
            full_path = os.path.join(path, f'{modality}.h5')
            if np.sum(data_slice) <= 1e-5:
                empty_slices.append(path)

            if np.sum(label_slice) <= 1e-5:
                non_positive_slices.append(path)

            while not os.path.exists(full_path):  # sometimes file is not created correctly => Just redo until it exists
                if not os.path.exists(path):
                    os.makedirs(path)
                with h5py.File(full_path, 'w') as data_file:
                    data_file.create_dataset('data', data=data_slice, dtype='f')
                    data_file.create_dataset('label', data=label_slice, dtype='i')

    return empty_slices, non_positive_slices


def retrieve_paths_static(patient_paths):
    data_dir_paths = defaultdict(list)
    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        patient = patient_path.split(os.sep)[-2]
        for timestep in filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path)):
            timestep_path = os.path.join(patient_path, timestep)
            for axis in filter(lambda x: os.path.isdir(os.path.join(timestep_path, x)), os.listdir(timestep_path)):
                axis_path = os.path.join(timestep_path, axis)
                slice_paths = filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path)))
                data_dir_paths[patient] += slice_paths

    return data_dir_paths


def retrieve_paths_longitudinal(patient_paths):
    data_dir_paths = defaultdict(list)
    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        patient = patient_path.split(os.sep)[-2]
        for timestep_x in sorted(filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
            x_timestep = defaultdict(list)
            timestep_x_int = int(timestep_x)
            timestep_x_path = os.path.join(patient_path, timestep_x)
            for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_path, x)), os.listdir(timestep_x_path))):
                axis_path = os.path.join(timestep_x_path, axis)
                slice_paths = sorted(filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                x_timestep[axis] = slice_paths

            for timestep_x_ref in sorted(filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
                x_ref_timestep = defaultdict(list)
                timestep_x_ref_int = int(timestep_x_ref)
                timestep_x_ref_path = os.path.join(patient_path, timestep_x_ref)
                for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_ref_path, x)), os.listdir(timestep_x_ref_path))):
                    axis_path = os.path.join(timestep_x_ref_path, axis)
                    slice_paths = sorted(filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                    x_ref_timestep[axis] = slice_paths

                    if timestep_x_int != timestep_x_ref_int:
                        data_dir_paths[patient] += zip(x_ref_timestep[axis], x_timestep[axis])

    return data_dir_paths


def get_patient_paths(data_dir, evaluate, phase):
    patient_paths = map(lambda name: os.path.join(name, 'data'),
                        (filter(lambda name: (evaluate.value if phase == Phase.TEST else Evaluate.TRAINING.value) in name,
                                glob(os.path.join(data_dir, '*')))))
    return patient_paths


def retrieve_filtered_data_dir_paths(root_dir, phase, data_dir_paths, empty_slices, non_positive_slices, mode, val_patients, view: Views = None):
    empty_file_path = os.path.join(root_dir, 'empty_slices.pckl')
    non_positive_slices_path = os.path.join(root_dir, 'non_positive_slices.pckl')

    if empty_slices:
        pickle.dump(empty_slices, open(empty_file_path, 'wb'))
    if non_positive_slices:
        pickle.dump(non_positive_slices, open(non_positive_slices_path, 'wb'))

    data_dir_path = os.path.join(root_dir, f'data_dir_{mode.value}_{phase.value}_{val_patients}{f"_{view.name}" if view else ""}.pckl')
    if os.path.exists(data_dir_path):
        # means it has been preprocessed before -> directly load data_dir_paths
        data_dir_paths = pickle.load(open(data_dir_path, 'rb'))
        print(f'Elements in data_dir_paths: {len(data_dir_paths)}')
    else:
        if not empty_slices:
            empty_slices = pickle.load(open(empty_file_path, 'rb'))
        if not non_positive_slices:
            non_positive_slices = pickle.load(open(non_positive_slices_path, 'rb'))
        print(f'Elements in data_dir_paths before filtering empty slices: {len(data_dir_paths)}')
        if mode == Mode.STATIC:
            data_dir_paths = [x for x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]
        else:
            data_dir_paths = [(x_ref, x) for x_ref, x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]

        print(f'Elements in data_dir_paths after filtering empty slices: {len(data_dir_paths)}')
        pickle.dump(data_dir_paths, open(data_dir_path, 'wb'))

    return data_dir_paths
