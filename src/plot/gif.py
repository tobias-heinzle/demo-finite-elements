import imageio
from os import listdir
from os.path import isfile, join
import re


def generate_gif(frames_directory: str, output_file: str):
    filenames = [frames_directory + "/" +
                 f for f in listdir(frames_directory) if isfile(join(frames_directory, f))]
    n = len(filenames)

    filenames = map(lambda tup: tup[0], sorted(
        [(f, int(re.sub("[^0-9]", "", f))) for f in filenames], key=lambda tup: tup[1]))

    with imageio.get_writer(output_file, mode='I') as writer:
        for k, filename in enumerate(filenames):
            print(f"{k}/{n}", end="\r")
            image = imageio.imread(filename)
            writer.append_data(image)  # type: ignore
