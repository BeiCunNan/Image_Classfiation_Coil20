import glob


all_imgs_path = glob.glob(r'dataset\*.png')
print(len(all_imgs_path))
for var in all_imgs_path:
    print(var)