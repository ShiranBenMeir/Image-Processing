import glob
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
c=0
for filename in glob.glob('images2/no_tumor/*.jpg'):
    img=Image.open(filename)
    img = img.resize((174,230))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    #
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:/Users/shiran/PycharmProjects/final_proj/aug_n', save_prefix='img', save_format='jpeg'):
        c=c+1
        i += 1
        if i > 1:
            break

