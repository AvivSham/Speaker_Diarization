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
        person_ids = random.sample(self.persons, 2)
        label_0, sample_0 = self.get_single_person_sample(person_ids[0])
        label_1, sample_1 = self.get_single_person_sample(person_ids[1])
        label = label_0 + label_1 * 2
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


if __name__ == '__main__':
    snyth_convo_gen = SyntheticMultiSpeakerGen("/home/dan/Downloads/vox_celebs/vox1_dev_wav/wav",
                                               "/home/dan/Downloads/vox_celebs/synth_convs")
    snyth_convo_gen.generate_samples(1000)
