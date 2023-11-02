import numpy as np
from numpy.random import RandomState
from torch.utils.data import Sampler


class PatientPhaseSliceBatchSampler(Sampler):

    def __init__(self, flat_data, hierarchical_data, batch_size, seed=0, use_patient=False, use_phase=False, use_slice_pos=False, reset_every_epoch=False):
        """Data is stored in a hierarchical list structure based on patient, phase, and slice position"""
        self.flat_data = flat_data
        self.hierarchical_data = hierarchical_data
        self.batch_size = batch_size
        self.reset_every_epoch = reset_every_epoch
        self.random_state = RandomState(seed=seed)
        self.use_patient = use_patient
        self.use_phase = use_phase
        self.use_slice_pos = use_slice_pos
        assert self.use_patient or self.use_phase or self.use_slice_pos, "Must use at least one of patient, phase, or slice position"
        self.num_grouped_per_batch = 1 + int(self.use_patient) + int(self.use_phase) + int(self.use_slice_pos)
        assert self.batch_size % self.num_grouped_per_batch == 0, "Batch size must be divisible by the number of positives per batch"
        assert self.batch_size > self.num_grouped_per_batch, "Batch size must be greater than the number of positives per batch"
        self.batches = None
        self.setup()
        print(f"Done setting up custom batch sampler with {len(self.batches)} batches that uses {self.num_grouped_per_batch} grouped per batch and use_patient {self.use_patient}, use_phase {self.use_phase}, use_slice_pos {self.use_slice_pos}, data length of {len(self.flat_data)}, and total number of samples {len(self)}")

    def __len__(self):
        return len(self.flat_data) * self.num_grouped_per_batch

    def __iter__(self):
        for batch in self.batches:
            print(f"batch {batch}")
            for idx in batch:
                print(f"idx {idx}, self.flat_data[idx] {self.flat_data[idx]}")
            yield batch
        if self.reset_every_epoch:
            print("Resetting batch sampler every epoch")
            self.setup()

    def setup(self):
        nested_by_patient_index_groups, flat_index_groups = self.generate_nested_and_flat_patient_data_groups()
        self.batches = []
        patient_group_meta_indices = np.arange(len(nested_by_patient_index_groups)).tolist()
        while len(flat_index_groups) > 1 and len(patient_group_meta_indices) > 1:
            random_patient_index_group1, random_patient_group_meta_index1 = self.get_valid_random_patient_index_group_and_meta_index(
                patient_group_meta_indices,
                nested_by_patient_index_groups)
            if random_patient_index_group1 is None:
                break
            non_matching_patient_group_meta_indices = [random_patient_group_meta_index1]
            new_batch_patient_index_groups = [random_patient_index_group1]
            # generate random unique patient groups get slice groups from
            while (len(new_batch_patient_index_groups) * self.num_grouped_per_batch) < self.batch_size:
                random_patient_index_group_adtl, random_patient_group_adtl_meta_index = self.get_valid_random_patient_index_group_and_meta_index(
                    patient_group_meta_indices,
                    nested_by_patient_index_groups,
                    non_matching_patient_group_meta_indices=non_matching_patient_group_meta_indices)
                if random_patient_index_group_adtl is None:
                    break
                new_batch_patient_index_groups.append(random_patient_index_group_adtl)
                non_matching_patient_group_meta_indices.append(random_patient_group_adtl_meta_index)
            if (len(new_batch_patient_index_groups) * self.num_grouped_per_batch) < self.batch_size:
                break
            new_batch = self.create_random_batch(new_batch_patient_index_groups)
            self.batches.append(new_batch)

    def create_random_batch(self, new_batch_patient_index_groups):
        new_batch = []
        for patient_index_group in new_batch_patient_index_groups:
            random_slice_index_group = self.get_random_slice_index_group(patient_index_group)
            patient_index_group.remove(random_slice_index_group)
            new_batch.extend(random_slice_index_group)
        return new_batch

    def get_valid_random_patient_index_group_and_meta_index(self, patient_group_meta_indices, nested_by_patient_index_groups,
                                                            non_matching_patient_group_meta_indices=None):
        if len(patient_group_meta_indices) == 0:
            return None, None
        if (non_matching_patient_group_meta_indices is not None) and (len(patient_group_meta_indices) <= len(non_matching_patient_group_meta_indices)):
            return None, None
        random_patient_group_meta_index = self.random_state.choice(patient_group_meta_indices)
        if (non_matching_patient_group_meta_indices is not None) and (random_patient_group_meta_index in non_matching_patient_group_meta_indices):
            return self.get_valid_random_patient_index_group_and_meta_index(patient_group_meta_indices, nested_by_patient_index_groups,
                                                                            non_matching_patient_group_meta_indices)
        random_patient_index_group = nested_by_patient_index_groups[random_patient_group_meta_index]
        if len(random_patient_index_group) == 0:
            patient_group_meta_indices.remove(random_patient_group_meta_index)
            return self.get_valid_random_patient_index_group_and_meta_index(patient_group_meta_indices, nested_by_patient_index_groups,
                                                                            non_matching_patient_group_meta_indices)
        else:
            return random_patient_index_group, random_patient_group_meta_index

    def get_random_slice_index_group(self, random_patient_index_group):
        slice_group_meta_indices = np.arange(len(random_patient_index_group))
        random_slice_meta_index = self.random_state.choice(slice_group_meta_indices)
        random_slice_index_group = random_patient_index_group[random_slice_meta_index]
        return random_slice_index_group

    def generate_nested_and_flat_patient_data_groups(self):
        flat_index_groups = []
        nested_by_patient_index_groups = []
        current_flat_index = 0
        for patient in self.hierarchical_data:
            patient_data_groups = []
            phase_1 = patient[0]
            phase_2 = patient[1]
            full_slice_lst = phase_1 + phase_2
            for slice_list in [phase_1, phase_2]:
                for i, slice in enumerate(slice_list):
                    current_data_group = list()
                    current_data_group.append(current_flat_index)
                    if self.use_slice_pos:
                        if i == 0:
                            current_data_group.append(current_flat_index + 1)
                        elif i == len(slice_list) - 1:
                            current_data_group.append(current_flat_index - 1)
                        else:
                            # randomly pick a slice before or after the current slice
                            random_slice_offset = self.random_state.choice([-1, 1])
                            current_data_group.append(current_flat_index + random_slice_offset)
                    if self.use_phase:
                        # randomly pick a slice in the same patient and phase
                        current_phase_flat_index = current_flat_index - i
                        random_slice_in_phase = None
                        while random_slice_in_phase not in current_data_group:
                            random_slice_in_phase = current_phase_flat_index + self.random_state.choice(np.arange(len(slice_list)))
                        current_data_group.append(random_slice_in_phase)
                    if self.use_patient:
                        # randomly pick a slice in the same patient
                        if slice_list is phase_1:
                            current_patient_flat_index = current_flat_index - i
                        else:
                            current_patient_flat_index = current_flat_index - i - len(phase_1)
                        random_slice_in_patient = None
                        while random_slice_in_patient not in current_data_group:
                            random_slice_in_patient = current_patient_flat_index + self.random_state.choice(np.arange(len(full_slice_lst)))
                    current_data_group.append(random_slice_in_patient)
                    patient_data_groups.append(current_data_group)
                    flat_index_groups.append(current_data_group)
                    current_flat_index += 1
            nested_by_patient_index_groups.append(patient_data_groups)
        return nested_by_patient_index_groups, flat_index_groups