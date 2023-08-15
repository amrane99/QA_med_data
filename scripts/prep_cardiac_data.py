import SimpleITK as sitk
import numpy as np
import os
import shutil
import random
from scipy.stats import mode


### Utility functions for the preprocessing of cardiac datasets:

def get_info(file_path):
    r'''
        Print out file information
    '''
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    
    print("\nMeta Data:")
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        print(f'({k}) = = "{v}"')
    
    print("\nFurther Information:")
    print("Size: " + str(reader.GetSize()))
    print("Dimenstions: " + str(reader.GetDimension()))
    print("Direction: " + str(reader.GetDirection()))
    print("Origin: " + str(reader.GetOrigin()))
    print("Spacing: " + str(reader.GetSpacing()))


def analyze_sizes(in_dir_path, type='decathlon'):
    r'''
        Analyse the sizes of each .nii.gz image in a root directory.
        Don't forget to set the type of the dataset which help to read the images: possible types: 'decathlon', 'acdc', 'sunnybrook'
        This method is intended to be applied on the original datasets - download links are in the README.md
    '''
    dim = 3
    if (type =='acdc' or type =='sunnybrook'):
        dim = 4

    # Array that stores all sizes of the images
    sizes = np.empty((0, dim), dtype=np.uint64)

    # Gather sizes of all images
    for dir_path, dir_names, file_names in os.walk(in_dir_path):
        for file_name in file_names:
            
            # Skip irelevant files
            if file_name.startswith("."):
                continue
            if not (file_name.endswith(".nii.gz") or file_name.endswith(".dcm")):
                continue
            if type == 'acdc':
                if not "4d" in file_name:
                    continue
            
            # Save the size of the image
            file_path = os.path.join(dir_path, file_name)
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            img_size = reader.GetSize()
            if len(img_size) != dim:
                print(f"Warning: Found unexpected size of image {file_path}")
            print(img_size)
            sizes = np.vstack((sizes, img_size))
    
    # Analyse the sizes for each dimension
    x_sizes = sizes[:, 0]
    y_sizes = sizes[:, 1]
    z_sizes = sizes[:, 2]
    print(f"\nnumber of images: {len(sizes)}")
    print(f"\nxy-ratios =\n {x_sizes/y_sizes}")
    print(f"mean ratio = {np.mean(x_sizes/y_sizes)}")

    print("\nx-sizes:")
    unique_values, counts = np.unique(x_sizes, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")
    print(f"max = {np.max(x_sizes)}")
    print(f"min = {np.min(x_sizes)}")
    print(f"mean = {np.mean(x_sizes)}")
    print(f"mode = {mode(x_sizes, keepdims=True).mode[0]}")
    
    print("\ny-sizes:")
    unique_values, counts = np.unique(y_sizes, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")
    print(f"max = {np.max(y_sizes)}")
    print(f"min = {np.min(y_sizes)}")
    print(f"mean = {np.mean(y_sizes)}")
    print(f"mode = {mode(y_sizes, keepdims=True).mode[0]}")

    print("\nz-sizes:")
    unique_values, counts = np.unique(z_sizes, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")
    print(f"max = {np.max(z_sizes)}")
    print(f"min = {np.min(z_sizes)}")
    print(f"mean = {np.mean(z_sizes)}") 
    print(f"mode = {mode(z_sizes, keepdims=True).mode[0]}")


def repair_nifty(in_file_path, out_dir_path=None, prefix=None):
    r'''
        - in_file_path:     File path which specifies the image to be repaired
        - [out_dir_path]:   Optional path to a directory where the repaired file will be stored.
        - [prefix]:         Optional prefix for the filename of the repaired image

        This method performs a back and forth conversion (image <-> array) on a copy of a single nii.gz file 
        and saves it in the directory of the input file (if not specified).
        In case it cannot be opened beforehand, this method should fix the problem.
        It should not change anything if the file is allright anyway (eqivalent to copying the file). 
    '''
    # Read the image
    image = sitk.ReadImage(in_file_path)

    # Handle naming of the image
    path, file = os.path.split(in_file_path)
    file_name = os.path.splitext(file)[0]
    
    if prefix is None:
        prefix = ''

    if out_dir_path is None:
        out_dir_path = path
    
    out_file_name = '%s%s.gz' % (prefix, file_name)
    out_file_path = os.path.join(out_dir_path, out_file_name)

    # Write the image
    sitk.WriteImage(image, out_file_path)

    print("Repaired " + in_file_path + " -> " + out_file_path)


def acdc_repair_nifties(in_dir_path, out_dir_path, prefix=None, copy_all=False):
    r'''
        - in_dir_path:  Input directory path to the directory where files to be repaired are stored.
        - out_dir_path: Output directory path to a directory where the repaired files will be stored.
        - [prefix]:     Optional prefix for the filename of the repaired image (defalt: None)
        - [copy_all]:   Optional parameter to enable that also non-nifti files are copied (default: False)
        
        The files to be repaired should be .nii.gz files
        
    '''
    print("Starting conversion...")
    
    entries = os.listdir(in_dir_path)
    for i, entry in enumerate(entries):
        entry_path = os.path.join(in_dir_path, entry)

        if os.path.isdir(entry_path):
            print(entry + " is a directory")
            in_sub_dir_path = os.path.join(in_dir_path, entry)
            out_sub_dir_path = os.path.join(out_dir_path, entry)
            os.makedirs(out_sub_dir_path, exist_ok=True)

            files = os.listdir(in_sub_dir_path)
            for j, file in enumerate(files):
                
                file_path = os.path.join(in_sub_dir_path, file)
                if os.path.isdir(file_path):
                    print("Warning: Found an unexpected directory (" + file_path + ")")
                    continue

                p, f = os.path.split(file_path)
                file_name = os.path.splitext(f)[0]
                file_ext = os.path.splitext(f)[1]
                if file_ext != '.gz':
                    print("Warning: Found an unexpected file (" + file_path + ")")
                    if copy_all == True:
                        shutil.copy(file_path, os.path.join(out_sub_dir_path, f))
                    continue

                repair_nifty(file_path, out_dir_path=out_sub_dir_path, prefix=prefix)
        
        elif os.path.isfile(entry_path):
            print(entry + " is a file")
            
            p, f = os.path.split(entry_path)
            file_name = os.path.splitext(f)[0]
            file_ext = os.path.splitext(f)[1]
            if file_ext != '.gz':
                print("Warning: Found an unexpected file (" + entry_path + ")")
                if copy_all == True:
                        shutil.copy(entry_path, os.path.join(out_dir_path, f))
                continue
            
            repair_nifty(entry_path, out_dir_path=out_dir_path, prefix=prefix)
        
        else:
            print("Warning: Entry skipped because it could not be identified as file or directory: " + entry + " (in directory: " + in_dir_path + ")")

    print("Reparation of nifties terminated.")


def split_4d(in_dir_path, out_dir_path=None, key='4d'):
    r'''
        - in_dir_path:      Directory path that specifies the entry point of application.
        - [out_dir_path]:   Optional path that specifies the output directory for the copied directory structure with the split frames
        - [key]:            Keyword that specifies those filenames that will be interpreted as 4d images to be split (default: '4d').

        !!! This method does not work with simple itk 1.2.4. Use instead temporary version 2.2.1 (but it leads to dependancy warning with other library versions) !!!

        This method generates 3d frames for each 4d image existing in the given directory (recognized by the key in the filename)
        and saves them in the same location as the original image.
        By default - if out_dir_path is not set - the direcories with the frames will be stored alongside the split files.
        This method is adjusted to the ACDC dataset naming conventions.
    '''
    for dir_path, dir_names, file_names in os.walk(in_dir_path):
        for file_name in file_names:
            if key in file_name:
                print("key found in: " + dir_path + "\\" + file_name)

                # Construct the names of the file_path to read it
                file_path = os.path.join(dir_path, file_name)
                patient = file_name.split('_')[0]
                patient_dir_name = patient + '_frames'

                # Read the image and get the size of the fourth dimension (#frames)
                image = sitk.ReadImage(file_path)
                numb_frames = image.GetSize()[3]

                # Construct the updated out_dir_path and create sub directory for the frames
                #out_dir_sub_path = out_dir_path
                if out_dir_path is None:
                    out_dir_sub_path = os.path.join(dir_path, patient_dir_name)
                else:
                    out_dir_sub_path = os.path.join(out_dir_path, patient_dir_name)
                
                os.makedirs(out_dir_sub_path, exist_ok=True)
                
                # Loop over the fourth dimension and save each 3D frame as a separate file
                for i in range(numb_frames):
                    frame = image[:, :, :, i]

                    # Generate the output file path     
                    idx = "%03d" % (i+1)
                    out_file_name = f"{patient}_frame{idx}.nii.gz"
                    out_file_path = os.path.join(out_dir_sub_path, out_file_name)

                    # Save the 3D volume as a separate file
                    sitk.WriteImage(frame, out_file_path)

                    print(f"Frame {i+1} saved as ", out_file_path)


def acdc_split_4d(in_dir_path, out_dir_path, copy_all=False):
    r'''
        - in_dir_path:      Directory path that specifies the entry point of application.
        - out_dir_path:     Directory path that specifies the output directory for the images.
        - [copy_all]:       Optional parameter that enables copying of frames and labels (default: False).

        This method searches all 4D-images (recognized by substing '4d' in filenames) in the input directory, splits the image, 
        and saves all the frames according to the 'Medical Decathlon' directory structure in the output directory. 
        If copy_all is set True, additionally existing labels and frames in the dataset are copied into the output directory.
        Notice: This method is adjusted to be applied to the ACDC dataset with its naming convention. Also, this method does not work with SimpleITK version 1.2.4.
    '''
    os.makedirs(os.path.join(out_dir_path, "images_Tr"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "images_Ts"), exist_ok=True)
    if copy_all:
        os.makedirs(os.path.join(out_dir_path, "opt_labels_Tr"), exist_ok=True)
        os.makedirs(os.path.join(out_dir_path, "opt_labels_Ts"), exist_ok=True)
        os.makedirs(os.path.join(out_dir_path, "opt_frames_Tr"), exist_ok=True)
        os.makedirs(os.path.join(out_dir_path, "opt_frames_Ts"), exist_ok=True)
    
    for dir_path, dir_names, file_names in os.walk(in_dir_path):
        for file_name in file_names:
            
            if '4d' in file_name:
                file_path = os.path.join(dir_path, file_name)
                patient_digits = file_name.split('_')[0][7:]
                patient_nr = int(patient_digits)

                image = sitk.ReadImage(file_path)
                numb_frames = image.GetSize()[3]

                if patient_nr <= 100:
                    dest_dir_name = "images_Tr"
                else:
                    dest_dir_name = "images_Ts"

                print(f"\nSplitting image {patient_nr}:")
                
                for i in range(numb_frames):
                    try:
                        frame = image[:, :, :, i]
                    except Exception as e:
                        print(f"Please check the SimpleITK package version. The following error might be due to an outdated version (approved in v2.2.1).\n {e}")
                    
                    frame_digits = "%02d" % (i+1)
                    out_file_name = f"{patient_digits}_{frame_digits}.nii.gz"
                        
                    out_file_path = os.path.join(out_dir_path, dest_dir_name, out_file_name)
                    sitk.WriteImage(frame, out_file_path)
                    print(f"Saved frame {i+1}/{numb_frames}")

            if copy_all:
                if ('gt' in file_name) | ('frame' in file_name):
                    file_path = os.path.join(dir_path, file_name)
                    file_name_split = file_name.split('_')
                    patient_digits = file_name_split[0][7:]
                    frame_digits = file_name_split[1][5:]
                    patient_nr = int(patient_digits)
                    
                    if ('gt' in file_name) & (patient_nr <= 100):
                        dest_dir_name = "opt_labels_Tr"
                    elif ('gt' in file_name) & (patient_nr > 100):
                        dest_dir_name = "opt_labels_Ts"
                    elif ('frame' in file_name) & (patient_nr <= 100):
                        dest_dir_name = "opt_frames_Tr"
                    elif ('frame' in file_name) & (patient_nr > 100):
                        dest_dir_name = "opt_frames_Ts"

                    out_file_name = f"{patient_digits}_{frame_digits}.nii.gz"
                    out_file_path = os.path.join(out_dir_path, dest_dir_name, out_file_name)
                    shutil.copy(file_path, out_file_path)


def dcm_to_nii(in_dir_path, out_dir_path):
    r'''
        - in_dir_path:    Input directory path containing .dcm files to be converted.
        - out_dir_path:   Output directory path for the corresponding .nii.gz converted files.
        
        This method converts all .dcm files in the input directory to .nii.gz files and saves them in the output directory.
    '''
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # Recursively search for DICOM files in the input directory
    for root, dirs, files in os.walk(in_dir_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                # Get the relative path of the current DICOM file
                relative_path = os.path.relpath(root, in_dir_path)
                output_subdir = os.path.join(out_dir_path, relative_path)
                # Create the corresponding subdirectory in the output directory
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                # Set the input and output file paths
                dcm_file = os.path.join(root, file)
                nii_file = os.path.join(output_subdir, file[:-4] + '.nii.gz')
                # Read the DICOM image and write it in NIfTI format
                image = sitk.ReadImage(dcm_file)
                sitk.WriteImage(image, nii_file)
                print(f"Converted: {dcm_file} -> {nii_file}")


def reorient_image(in_file_path, out_dir_path=None, orientation='RAI', prefix=None):
    r'''
        - in_file_path:     Directory path that specifies the entry point of application.
        - [out_dir_path]:   Optional path that specifies the output directory for the converted image.
        - [orientation]:    Orientation ['RAI', 'LPI'] that the images will be converted to (default is 'RAI').
        - [prefix]:         Optional prefix for the converted file name

        This method reorients an image to the specified orientation and overrides the original image by default - if both out_dir_path and prefix are not set.
        (This method does not work on 4d images; This method does works on the acdc images that can not be opened with itk snap)
    '''
    image = sitk.ReadImage(in_file_path)
    
    # Create a reorientation transform to LPI orientation
    reorient_transform = sitk.AffineTransform(3)
    reorient_transform.SetMatrix([1, 0, 0, 0, 1, 0, 0, 0, 1])   # Identity matrix
    reorient_transform.SetCenter([0, 0, 0])                     # Center at the origin

    # Apply the reorientation transform
    reoriented_image = sitk.Resample(image, reorient_transform)

    # Set the image's direction
    if orientation == 'RAI':
        new_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if orientation == 'LPI':
        new_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
    
    reoriented_image.SetDirection(new_direction)

    # Manage directory and file namings
    path, file = os.path.split(in_file_path)
    file_name = os.path.splitext(file)[0]
    file_ext = os.path.splitext(file)[1]

    if prefix is None:
        prefix = ''

    if out_dir_path is None:
        out_dir_path = path
    
    out_file_name = '%s%s.gz' % (prefix, file_name)
    out_file_path = os.path.join(out_dir_path, out_file_name)

    # Save the reorientated image
    sitk.WriteImage(reoriented_image, out_file_path)

    print("Reoriented " + in_file_path + " -> " + out_file_path)


def reorient_all(in_dir_path, out_dir_path=None, orientation='RAI', prefix=None):
    r'''
        - in_dir_path:      Directory path that specifies the entry point of application.
        - [out_dir_path]:   Optional path that specifies the output directory for the copied directory structure with the conversions.
        - [orientation]:    Orientation ['RAI', 'LPI'] that the images will be converted to (default is 'RAI').
        - [prefix]:         Optional prefix for the converted file names

        This method reorients all .nii.gz images in the input directory to the specified orientation and overrides the original images by default - if both out_dir_path and prefix are not set.
        If the out_dir_path is set, a parallel file structure will be created where the conversions will be saved.
        (This method does not work on 4d images; This method does works on the acdc images that can not be opened with itk snap)
    '''
    # Recursively search the directory for all images
    for dir_path, dir_names, file_names in os.walk(in_dir_path):
        for file_name in file_names:
            
            # Skip irelevant files
            if file_name.startswith("."):
                continue
            if not file_name.endswith(".nii.gz"):
                continue
            if '4d' in file_name:
                continue
            
            # Read the image
            in_file_path = os.path.join(dir_path, file_name)
            image = sitk.ReadImage(in_file_path)
    
            # Create a reorientation transform to LPI orientation
            reorient_transform = sitk.AffineTransform(3)
            reorient_transform.SetMatrix([1, 0, 0, 0, 1, 0, 0, 0, 1])   # Identity matrix
            reorient_transform.SetCenter([0, 0, 0])                     # Center at the origin
            reoriented_image = sitk.Resample(image, reorient_transform)

            # Set the image's direction for reorientation
            if orientation == 'RAI':
                new_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            if orientation == 'LPI':
                new_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]

            reoriented_image.SetDirection(new_direction)

            if prefix is None:
                prefix = ''

            if out_dir_path is None:
                out_dir_path = dir_path

            # Manage file and directory naming
            relative_path = os.path.relpath(dir_path, in_dir_path)
            output_subdir = os.path.join(out_dir_path, relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            out_file_name = '%s%s' % (prefix, file_name)
            out_file_path = os.path.join(output_subdir, out_file_name)

            # Save the reorientated image
            sitk.WriteImage(reoriented_image, out_file_path)
            print("Reoriented " + in_file_path + " -> " + out_file_path)



def fuse_dcm_to_nii_with_sub_dir(in_dir_path, out_dir_path):
    r'''
        - in_dir_path:    Input directory path containing the sub-folders with the images (.dcm) to be fused
        - out_dir_path:   Output directory for the fused files (.nii.gz).    
    
        This method can be used if there are multiple folders with .dcm images where all images within each folder need to be fused into one file each.
        It iterates through all sub-directories of the given in_dir_path and fuses all .dcm images via sitk.JoinSeries method.
        The fused images of each sub-directory in the in_dir_path are stored as a copy in their corresponding sub-directory as a .nii.gz file.
    '''
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    for root, dirs, files in os.walk(in_dir_path):
        images = []

        for file in files:
            if file.lower().endswith('.dcm'):
                relative_path = os.path.relpath(root, in_dir_path)
                output_subdir = os.path.join(out_dir_path, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                dcm_file = os.path.join(root, file)
                image = sitk.ReadImage(dcm_file)
                images.append(image)

        if images:
            fused_image = sitk.JoinSeries(images)
            nii_file = os.path.join(output_subdir, "fused_image.nii.gz", )
            
            sitk.WriteImage(fused_image, nii_file)
            print(f"Fused images in {root} -> {nii_file}")


def fuse_dcm_to_nii(in_dir_path, output_dir):
    r'''
        - input_dir:    Input directory path for the images (.dcm) to be fused
        - output_dir:   Output directory for the fused file (.nii.gz).    
    
        This method can be used if there are .dcm images that need to be fused into one .nii.gz file.
        It fuses all .dcm images in the input_dir via sitk.JoinSeries method.
        The fused image is stored as a .nii.gz file in the output_dir.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(in_dir_path):
        images = []

        for file in files:
            if file.lower().endswith('.dcm'):
                relative_path = os.path.relpath(root, in_dir_path)
                output_subdir = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                dcm_file = os.path.join(root, file)
                image = sitk.ReadImage(dcm_file)
                images.append(image)

        if images:
            fused_image = sitk.JoinSeries(images)
            parent_dir = os.path.dirname(output_subdir)
            nii_file = os.path.join(parent_dir, "fused_image.nii.gz")
            
            sitk.WriteImage(fused_image, nii_file)
            print(f"Fused images in {root} -> {nii_file}")


def sb_kaggle_fuse_dcm_to_nii(in_dir_path, out_dir_path):
    r'''
        - in_dir_path:    Input directory path containing the sub-folders with the images (.dcm) to be fused
        - out_dir_path:   Output directory for the fused files (.nii.gz).    
    
        This method can be used if there are multiple folders with .dcm images where all images within each folder need to be fused into one file each.
        It iterates through all sub-directories of the given in_dir_path and fuses all .dcm images via sitk.JoinSeries method to a .nii.gz file which is saved in the out_dir_path.
        Note: This method ignores all "test.dcm" files in the fusion process.
    '''
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    for root, dirs, files in os.walk(in_dir_path):
        images = []

        for file in files:
            if file == 'test.dcm':
                continue
            if file.lower().endswith('.dcm'):
                relative_path = os.path.relpath(root, in_dir_path)
                output_subdir = os.path.join(out_dir_path, relative_path)
                
                dcm_file = os.path.join(root, file)
                image = sitk.ReadImage(dcm_file)
                images.append(image)

        if images:
            fused_image = sitk.JoinSeries(images)
            prefix = output_subdir.split('\\')[-1]
            path = output_subdir.rsplit('\\', 1)[0]
            nii_file = os.path.join(path, prefix + "_fused.nii.gz")
            
            sitk.WriteImage(fused_image, nii_file)
            print(f"Fused images in {root} -> {nii_file}")



def convert_folder_structure(in_dir_path, prefix="patient"):
    r"""
        - in_dir_path:      Directory path that specifies the entry point of application.

        This method can be used to prepare the ACDC data for augmentation. 
        It adjusts the directory structure to fit the JIP-format.
        It performs a conversion of a given directory the following way:
        Example with prefix="patient"
        Before:
            .../input
                    .../001_01.nii.gz
                    .../001_02.nii.gz
                    ...
        After:
            .../input
                    .../patient_001_01/img/img.nii.gz
                    .../patient_001_02/img/img.nii.gz
                    ...
    """
    # Get a list of all files in the original directory
    file_list = os.listdir(in_dir_path)

    for file_name in file_list:
        if file_name.endswith('.nii.gz'):
            # Extract the desired folder name
            folder_name = prefix + "_" + file_name[:-7]

            # Create the new folder structure
            new_folder_path = os.path.join(in_dir_path, folder_name, 'img')
            os.makedirs(new_folder_path, exist_ok=True)

            # Move and rename the file
            original_file_path = os.path.join(in_dir_path, file_name)
            new_file_path = os.path.join(new_folder_path, 'img.nii.gz')
            shutil.move(original_file_path, new_file_path)




def convert_to_rai_orientation(in_dir_path, out_dir_path):

    os.makedirs(out_dir_path, exist_ok=True)

    # Get a list of all NIfTI files in the input directory
    file_list = [file for file in os.listdir(in_dir_path) if file.endswith('.nii.gz')]

    for file_name in file_list:
        # Read the NIfTI image
        input_file_path = os.path.join(in_dir_path, file_name)
        image = sitk.ReadImage(input_file_path)
        original_spacing = image.GetSpacing()
        
        # Reorient the image to RAI orientation
        direction_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        image.SetDirection(direction_matrix)

        # Save the reoriented image to the output directory
        output_file_path = os.path.join(out_dir_path, file_name)
        sitk.WriteImage(image, output_file_path)

        # Set the original spacings back to the reoriented image
        reoriented_image = sitk.ReadImage(output_file_path)
        reoriented_image.SetSpacing(original_spacing)
        sitk.WriteImage(reoriented_image, output_file_path)



def prepare_task_datasets(in_dir_path, out_dir_path, dataset_name="task"):
    os.makedirs(out_dir_path, exist_ok=True)

    # Get a list of all NIfTI files in the input directory
    file_list = [file for file in os.listdir(in_dir_path) if file.endswith('.nii.gz')]

    for file_name in file_list:
        # Read the NIfTI image
        input_file_path = os.path.join(in_dir_path, file_name)
        image = sitk.ReadImage(input_file_path)
        original_spacing = image.GetSpacing()
        
        # Reorient the image to RAI orientation
        direction_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        image.SetDirection(direction_matrix)

        # Save the reoriented image to the output directory
        new_filename = file_name.split('_')[0]
        output_file_path = os.path.join(out_dir_path, file_name)
        sitk.WriteImage(image, output_file_path)

        # Set the original spacings back to the reoriented image
        reoriented_image = sitk.ReadImage(output_file_path)
        reoriented_image.SetSpacing(original_spacing)
        sitk.WriteImage(reoriented_image, output_file_path)




def cut_filenames(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz') and '_' in filename:
            parts = filename.split('_')
            new_filename = f"{parts[0]}.nii.gz"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

def add_prefix(input_directory, prefix="acdc"):
    for folder_name in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder_name)
        if os.path.isdir(folder_path):
            new_folder_name = f"{prefix}_{folder_name}"
            new_folder_path = os.path.join(input_directory, new_folder_name)
            os.rename(folder_path, new_folder_path)

def remove_last_chars(input_directory):
    for folder_name in os.listdir(input_directory):
        if os.path.isdir(os.path.join(input_directory, folder_name)) and folder_name.endswith("_01"):
            new_name = folder_name[:-3]  # Remove the last 3 characters "_01"
            old_path = os.path.join(input_directory, folder_name)
            new_path = os.path.join(input_directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {folder_name} to {new_name}")

def add_postfix(input_directory):
    for folder_name in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder_name)
        if os.path.isdir(folder_path):
            new_folder_name = folder_name + "-01"
            new_folder_path = os.path.join(input_directory, new_folder_name)
            os.rename(folder_path, new_folder_path)
            print(f"Renamed '{folder_name}' to '{new_folder_name}'")

def compare_filenames(directory_a, directory_b):
    files_a = os.listdir(directory_a)
    files_b = os.listdir(directory_b)

    common_files = set(files_a) & set(files_b)

    if common_files:
        print("Common files found:")
        for filename in common_files:
            print(filename)
    else:
        print("No common files found.")

def count_folders(input_directory):
    folder_count = 0
    for item in os.listdir(input_directory):
        item_path = os.path.join(input_directory, item)
        if os.path.isdir(item_path):
            folder_count += 1
    print("folder_count:", folder_count)


def rename_folders(input_folder):
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            new_folder_name = folder_name.replace("adac", "acdc")
            new_folder_path = os.path.join(input_folder, new_folder_name)
            os.rename(folder_path, new_folder_path)
            print(f"Renamed folder: {folder_name} to {new_folder_name}")


def rename_files(input_folder):
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            new_filename = filename.replace("adac", "acdc")
            old_path = os.path.join(input_folder, filename)
            new_path = os.path.join(input_folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} to {new_filename}")



def divide_data(input_folder, train_folder, test_folder, train_percentage):
    random.seed(42)  # Set the random seed for reproducibility
    
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    nii_files = [file for file in os.listdir(input_folder) if file.endswith(".nii.gz")]

    random.shuffle(nii_files)
    total_files = len(nii_files)
    train_count = int(total_files * train_percentage)
    
    train_files = nii_files[:train_count]
    test_files = nii_files[train_count:]

    cnt = 0
    for file in train_files:
        cnt = cnt + 1
        source_path = os.path.join(input_folder, file)
        destination_path = os.path.join(train_folder, file)
        shutil.copy(source_path, destination_path)
    print(f"{cnt} files in Tr")
    cnt = 0
    for file in test_files:
        cnt = cnt + 1
        source_path = os.path.join(input_folder, file)
        destination_path = os.path.join(test_folder, file)
        shutil.copy(source_path, destination_path)
    print(f"{cnt} files in Tr")

    print("\nData division completed.")


def main():

    # Exec...
    get_info(r"")

if __name__ == '__main__':
    main()

