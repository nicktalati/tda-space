from PIL import Image
from os import listdir

def png_to_gif(folder, name):
    filenames = listdir(folder)
    filenames = sorted(filenames, key=lambda x: int(x[:-4]))
    filenames = [folder + '/' + name for name in filenames]
    images = [Image.open(image) for image in filenames]
    images[0].save(name,
                   save_all=True,
                   append_images=images[1:],
                   optimize=False,
                   duration=40,
                   loop=0)
    return

def png_to_gif2(folder, name):
    n_subfolders = len(listdir(folder))
    filenames = []
    folder_index = 0
    inc = 1
    for i in range(300):
        filename = str(folder_index) + '/' + "{:02d}".format(i) + ".png"
        filenames.append(folder + '/' + filename)
        if folder_index == 0:
            inc = 1
        elif folder_index == n_subfolders - 1:
            inc = -1
        folder_index += inc
    images = [Image.open(image) for image in filenames]
    images[0].save(name,
                   save_all=True,
                   append_images=images[1:],
                   optimize=False,
                   duration=40,
                   loop=0)
    return


if __name__ == "__main__":
    png_to_gif("pngs", "examples/cosmic_slice.gif")