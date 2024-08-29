import os
import imageio

def plot_as_gif(dir):
    images = []
    for file_name in sorted(os.listdir(dir)):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(dir, file_name)
            images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    # for _ in range(50):
    #     images.append(imageio.imread(file_path))

    # imageio.mimsave('../animation/gif/movie.gif', images)
    imageio.mimsave('/home/jacealdr/repos/Cascaded-Cost-Factors/output/cascaded_mpc/cascaded_mpc.gif',images)