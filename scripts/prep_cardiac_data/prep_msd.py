import os


def remove_prefix(input_dir, output_dir=None):
    '''
        Remove the 'la_'-prefix in the file names of the MSD Dataset.
    '''
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)) and not filename.startswith('.'):
            new_filename = filename[3:]
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, new_filename)

            os.rename(input_path, output_path)




def main():
    # 1- Collect all iamge files into one directory (images_prep)

    # 2. Rename files
    #remove_prefix(r"path")



if __name__ == '__main__':
    main()