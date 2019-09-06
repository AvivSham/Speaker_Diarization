import os
import random
from glob import glob
from pydub import AudioSegment
import numpy as np
from progressbar import progressbar
from scipy.io import wavfile as wavf


class SyntheticMultiSpeakerGen:
    def __init__(self, path_to_vox, output_path, sample_len_sec=10):
        self.sample_len_sec = sample_len_sec
        self.output_path = output_path
        self.path_to_vox = path_to_vox
        self.all_files = glob(path_to_vox + "/**/*.wav", recursive=True)
        self.persons = list(filter(lambda x: os.path.isdir(os.path.join(path_to_vox, x)), os.listdir(path_to_vox)))
        self.person_files = {person: glob(path_to_vox + "/" + person + "/**/*.wav", recursive=True) for person in
                             self.persons}
        for dir in ["label", "wav"]:
            if not os.path.exists(f"{self.output_path}/{dir}"):
                os.mkdir(f"{self.output_path}/{dir}")

    def generate_samples(self, num_samples=2e6):
        [self.get_sample() for _ in progressbar(range(num_samples))]

    def get_sample(self):
        person_ids = random.sample(self.persons,2)
        label_0, sample_0 = self.get_single_person_sample(person_ids[0])
        label_1, sample_1 = self.get_single_person_sample(person_ids[1])
        label = label_0 + label_1*2
        sample = sample_0.overlay(sample_1, 0)

        id = random.randint(0, 10000000)
        sample.export(f"{self.output_path}/wav/{person_ids[0]}_{person_ids[1]}_{id}.wav", format='wav')
        np.save(f"{self.output_path}/label/{person_ids[0]}_{person_ids[1]}_{id}.npy", label)

    def get_single_person_sample(self, person_id):
        sample = AudioSegment.silent(duration=self.sample_len_sec * 1000)
        label = np.zeros(len(sample))
        start_idx = 0
        while start_idx < len(sample):
            if random.random() > .5:
                start_idx += random.randint(250, 1500)
            else:
                random_file = random.choice(self.person_files[person_id])
                new_seg = self.get_audio_segment(random_file)
                sample = sample.overlay(new_seg, position=start_idx)
                label[start_idx: start_idx + len(new_seg)] = 1
                start_idx += len(new_seg)
        return label, sample

    def get_audio_segment(self, random_file):
        seg = AudioSegment.from_file(random_file)
        seg_len = int(len(seg))
        seg_middle = int(seg_len // 2 + random.randint(-seg_len // 2 + 100, seg_len // 2 - 100))
        seg = seg[
              random.randint(0, seg_middle - 100):random.randint(seg_middle + 100, seg_len)]
        return seg


class SyntheticMultiSpeakerGen_ver2:
    def __init__(self, path_to_vox, output_path, generated_len=10_000):
        self.generated_len = generated_len
        self.output_path = output_path
        self.path_to_vox = path_to_vox
        self.all_files = glob(path_to_vox + "/**/*.wav", recursive=True)
        self.persons = list(filter(lambda x: os.path.isdir(os.path.join(path_to_vox, x)), os.listdir(path_to_vox)))
        self.person_files = {person: glob(path_to_vox + "/" + person + "/**/*.wav", recursive=True) for person in
                             self.persons}
        for dir in ["label", "wav"]:
            if not os.path.exists(f"{self.output_path}/{dir}"):
                os.mkdir(f"{self.output_path}/{dir}")

    def generate_samples(self, n_samples):
        return

    def create_sample(self):
        person_curr = random.sample(self.persons, 2)
        silent_percent = np.random.randint(3, 7, 2) / 10
        max_start_ind = int((1 - silent_percent) * self.generated_len)
        silent_idxs = np.random.randint(0, max_start_ind, 2)

        speaker_1, label_1 = self.get_voc(person_curr[0])
        speaker_2, label_2 = self.get_voc(person_curr[1])

        speaker_1, label_1 = self.add_silent(speaker_1, silent_idxs[0],
                                             silent_idxs[0] + self.generated_len * silent_percent)
        speaker_2, label_2 = self.add_silent(speaker_2, silent_idxs[1],
                                             silent_idxs[1] + self.generated_len * silent_percent)

        generated_sample = speaker_1 + speaker_2
        generated_label = label_1 + label_2

        fs = 1000
        wavf.write(self.output_path, fs, generated_sample)
        wavf.write(self.output_path + 'bla', fs, speaker_1)
        wavf.write(self.output_path + 'bla bla', fs, speaker_2)

        np.savetxt(self.output_path + 'blabla_label.txt', generated_label, delimiter=',')

    def get_voc(self, person_id, output_path='/home/aviv/Desktop/', save=False):
        max_length = 10000
        count = 0
        generated_voc = []
        while (1):
            if count == max_length:
                break
            else:
                random_file = random.choice(self.person_files[person_id])
                signal = AudioSegment.from_file(random_file)
                cut_length = np.random.randint(500, 1000)
                # for edge case when count + cut_length will exceed max_length
                cut_length = count if cut_length > count else cut_length
                start_ind = np.random.randint(0, len(signal) - cut_length, 1)
                generated_voc.append(signal[start_ind:start_ind + cut_length])
                count += cut_length
        if save:
            fs = 1000
            wavf.write(output_path, fs, generated_voc)

        return generated_voc

    def add_silent(self, speaker, start_idx, stop_idx):
        speaker[start_idx:stop_idx] = AudioSegment.silent(duration=(stop_idx - start_idx))
        label = np.ones(len(speaker))
        label[start_idx:stop_idx] = 0
        return speaker, label


if __name__ == '__main__':
    snyth_convo_gen = SyntheticMultiSpeakerGen("/home/dan/Downloads/vox_celebs/vox1_dev_wav/wav",
                                                    "/home/dan/Downloads/vox_celebs/synth_convs")
    snyth_convo_gen.generate_samples(1000)
