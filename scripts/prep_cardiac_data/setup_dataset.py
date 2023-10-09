import os
import shutil
import random
import SimpleITK as sitk


def copy_files_with_suffix(in_dir_path, out_dir_path, substring="_01.nii.gz"):
    r"""
        - in_dir_path:  Directory path that specifies the input directory with files to be copied.
        - out_dir_path: Directory path that specifies the destination point for the copied files.
        - substring:    Substing for identification of the files to be copied.
    
        This method copies all files from the input directory to the output directory, that have the specified
        substring in their filename.
        Note:   This method was originaly used to setup reduced acdc data (only frames 01) for application of 
                a new train and test split via divide_data().
    """
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    for filename in os.listdir(in_dir_path):
        if filename.endswith(substring):
            source_path = os.path.join(in_dir_path, filename)
            destination_path = os.path.join(out_dir_path, filename)
            shutil.copy(source_path, destination_path)
            print(f"Copied '{filename}' to '{out_dir_path}'")


def copy_content(in_dir_path, out_dir_path):
    r"""
        - in_dir_path:  Directory path that specifies the input directory for the content to be copied.
        - out_dir_path: Directory path that specifies the destination point for the copied content.
    
        This method copies all files and folders from the input directory to the output directory.
    """
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    # Iterate over items and copy them to the destination directory
    for item in os.listdir(in_dir_path):
        input_item = os.path.join(in_dir_path, item)
        output_item = os.path.join(out_dir_path, item)
        
        if os.path.isdir(input_item):
            shutil.copytree(input_item, output_item)
        else:
            shutil.copy2(input_item, output_item)
    
        print(f"Copied '{input_item}' to '{out_dir_path}'")


def random_selection(in_dir_path, out_dir_path, num_files_to_copy, seed=42):
    os.makedirs(out_dir_path, exist_ok=True)

    nii_files = [file for file in os.listdir(in_dir_path) if file.endswith('.nii.gz')]

    random.seed(seed)
    # Randomly select num_files_to_copy files
    selected_files = random.sample(nii_files, num_files_to_copy)

    # Copy selected files to the output directory
    for i, file in enumerate(selected_files):
        source_path = os.path.join(in_dir_path, file)
        dest_path = os.path.join(out_dir_path, file)
        shutil.copy2(source_path, dest_path)
    print(f"Selected {i+1} images.")


def divide_data(in_dir_path, train_dir_path, test_dir_path, tr_ratio, seed=42):
    r"""
        - in_dir_path:      Directory path to all .nii.gz image files to be used as source for the split
                            into train and test partitions.
        - train_dir_path:   Directory path for the destination of the train partition.
        - test_dir_path:    Directory path for the destination of the test partition.
        - [tr_ratio]:       Specifies the percentage of the images (in the "in_dir_path") to be used for the train partition,
                            for test partition 1-tr_ration is used (default: 0.8).
        - [seed]:           The seed for the random selection of images for the train and test partitions (default: 42).

        This method copies and assigns all images in the input directory either into the specified train directory or into the test
        directory according to the tr_ratio.
    """
    random.seed(seed)
    
    if not os.path.exists(in_dir_path):
        print("Input folder does not exist.")
        return

    if not os.path.exists(train_dir_path):
        os.makedirs(train_dir_path)
    if not os.path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    # Exclude
    nii_files = [file for file in os.listdir(in_dir_path) if file.endswith(".nii.gz")]
    print(f"Total of {len(nii_files)} images")

    random.shuffle(nii_files)
    total_files = len(nii_files)
    
    exact_count = total_files * tr_ratio
    decimal = exact_count - int(exact_count)
    if decimal < 0.5:
        train_count = int(exact_count)
    else:
        train_count = int(exact_count) + 1
    
    train_files = nii_files[:train_count]
    test_files = nii_files[train_count:]

    cnt = 0
    for file in train_files:
        cnt = cnt + 1
        source_path = os.path.join(in_dir_path, file)
        destination_path = os.path.join(train_dir_path, file)
        shutil.copy(source_path, destination_path)
    print(f"{cnt} files in Tr")
    
    cnt = 0
    for file in test_files:
        cnt = cnt + 1
        source_path = os.path.join(in_dir_path, file)
        destination_path = os.path.join(test_dir_path, file)
        shutil.copy(source_path, destination_path)
    print(f"{cnt} files in Ts")


def convert_dir_structure(in_dir_path, ds_name):
    r"""
        - in_dir_path:  Directory path that specifies the entry point of application.
        - ds_names:     Name of the dataset which will be applied as prefix to the patient names.
                        The original usage of this method includes the folling: 'acdc', 'task808', 'task809', 'msd'

        This method adjusts the directory structure of the in_dir_path to fit the JIP-format.
        The original filename (patientID) is incorporated according to the following folder naming convention:
        <ds_name>_patient_<patient_id>

        Note: This method works inplace.
        
        The method performs a conversion of a given directory the following way:
        Example with ds_name="acdc"
        Before: .../input
                    .../001-01.nii.gz
                    .../001-02.nii.gz
                    ...
        After:  .../input
                    .../acdc_patient_001-01/img/img.nii.gz
                    .../acdc_patient_001-02/img/img.nii.gz
                    ...
    """
    if not os.path.exists(in_dir_path):
        print("Input folder does not exist.")
        return

    files = [file for file in os.listdir(in_dir_path) if file.endswith('.nii.gz')]

    for file in files:
        if not file.endswith('.nii.gz'):
            continue
            
        # Define new folder name and create new folder structure
        folder_name = ds_name + "_patient_" + file[:-7]
        new_folder_path = os.path.join(in_dir_path, folder_name, 'img')
        os.makedirs(new_folder_path, exist_ok=True)

        # Move and rename the files
        original_file_path = os.path.join(in_dir_path, file)
        new_file_path = os.path.join(new_folder_path, 'img.nii.gz')
        shutil.move(original_file_path, new_file_path)
        print(f"Replaced {original_file_path} with {new_file_path}")


def copy_large_images(in_dir_path, out_dir_path, nr_slices=8):
    r"""
        - in_dir_path:  Directory path that specifies the input directory with files to be copied.
        - out_dir_path: Directory path that specifies the destination point for the copied files.
        - nr_slices:    Amout of slices that the images need to have to be copied.
    
        This method copies all files from the input directory to the output directory, that have at least 8 slices.
    """
    os.makedirs(out_dir_path, exist_ok=True)
    
    not_copied_count = 0
    # Iterate through the files in the input directory
    for file_name in os.listdir(in_dir_path):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(in_dir_path, file_name)
            try:
                image = sitk.ReadImage(file_path)
                size_z = image.GetDepth()
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                size_z = None

            # Check if the number of slices meets the condition -> then copy them
            if size_z is not None and size_z >= nr_slices:
                shutil.copy2(file_path, os.path.join(out_dir_path, file_name))
            else:
                print(f"{file_name} not copied due to insufficient z-dimension size: {size_z}.")
                not_copied_count += 1

    print(f"Total {not_copied_count} files not copied due to insufficient z-dimension size.")


def main():

    print("")
    # 1. Copy all images but skip images whose slice size is less than 8 (3 ds)
    #copy_large_images(r"path",
    #                  r"path", 8)
    #copy_large_images(r"path",
    #                  r"path", 8)
    #copy_large_images(r"path",
    #                  r"path", 8)
    
    # 2. Select from each dataset 72 images randomly (3 ds)
    #random_selection(r"path", 
    #                 r"path", 72)
    #random_selection(r"path", 
    #                 r"path", 72)
    #random_selection(r"path", 
    #                 r"path", 72)

    # 3. Split datasets into train and test data (3 ds)
    #divide_data(in_dir_path=r"path", 
    #            train_dir_path=r"path", 
    #            test_dir_path=r"path", 
    #            tr_ratio=0.8)
    #
    #divide_data(in_dir_path=r"path", 
    #            train_dir_path=r"path", 
    #            test_dir_path=r"path", 
    #            tr_ratio=0.8)
    #
    #divide_data(in_dir_path=r"path", 
    #            train_dir_path=r"path", 
    #            test_dir_path=r"path", 
    #            tr_ratio=0.8)

    # 4. Convert directory structure for each dataset (3 ds, tr/ts)
    #convert_dir_structure(r"path", "acdc")
    #convert_dir_structure(r"path", "task808")
    #convert_dir_structure(r"path", "task809")

    #convert_dir_structure(r"path", "acdc")
    #convert_dir_structure(r"path", "task808")
    #convert_dir_structure(r"path", "task809")

    # 5. Copy data of each dataset into one folder (3 ds, tr/ts)
    #copy_content(r"path", r"path")
    #copy_content(r"path", r"path")
    #copy_content(r"path", r"path")

    #copy_content(r"path", r"path")
    #copy_content(r"path", r"path")
    #copy_content(r"path", r"path")
    
    
if __name__ == '__main__':
    main()



