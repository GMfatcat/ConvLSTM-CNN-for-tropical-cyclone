import imageio
import os


dir = './prediction' # change to your .png path

gif_images = []
for i in range(5):
    gif_images.append(imageio.imread(os.path.join(dir,str(i+1)+"_animate.png")))

imageio.mimsave("result.gif", gif_images, fps=1)   
