import SimpleITK as sitk
import os
import shutil
import random

def adac_split_4d(in_dir_path, out_dir_path, copy_all=False):
    r'''
        - in_dir_path:      Directory path that specifies the entry point of application.
        - out_dir_path:     Directory path that specifies the output directory for the images.
        - [copy_all]:       Optional parameter that enables copying of frames and labels (default: False).

        This method searches all 4D-images (recognized by substing '4d' in filenames) in the input directory, splits the image, 
        and saves all the frames according to the 'Medical Decathlon' directory structure in the output directory. 
        If copy_all is set True, additionally existing labels and frames in the dataset are copied into the output directory.
        Notice: This method is adjusted to be applied to the ACDC dataset with its naming convention. Also, this method does not work with SimpleITK version 1.2.4.
    '''
    os.makedirs(os.path.join(out_dir_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "imagesTs"), exist_ok=True)
    if copy_all:
        os.makedirs(os.path.join(out_dir_path, "opt_labelsTr"), exist_ok=True)
        os.makedirs(os.path.join(out_dir_path, "opt_labelsTs"), exist_ok=True)
        os.makedirs(os.path.join(out_dir_path, "opt_framesTr"), exist_ok=True)
        os.makedirs(os.path.join(out_dir_path, "opt_framesTs"), exist_ok=True)
    
    for dir_path, dir_names, file_names in os.walk(in_dir_path):
        for file_name in file_names:
            
            if '4d' in file_name:
                file_path = os.path.join(dir_path, file_name)
                patient_digits = file_name.split('_')[0][7:]
                patient_nr = int(patient_digits)

                image = sitk.ReadImage(file_path)
                numb_frames = image.GetSize()[3]

                if patient_nr <= 100:
                    dest_dir_name = "imagesTr"
                else:
                    dest_dir_name = "imagesTs"

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
                        dest_dir_name = "opt_labelsTr"
                    elif ('gt' in file_name) & (patient_nr > 100):
                        dest_dir_name = "opt_labelsTs"
                    elif ('frame' in file_name) & (patient_nr <= 100):
                        dest_dir_name = "opt_framesTr"
                    elif ('frame' in file_name) & (patient_nr > 100):
                        dest_dir_name = "opt_framesTs"

                    out_file_name = f"{patient_digits}_{frame_digits}.nii.gz"
                    out_file_path = os.path.join(out_dir_path, dest_dir_name, out_file_name)
                    shutil.copy(file_path, out_file_path)


def select_frames(in_dir_path, out_dir_path, seed=42):
    r"""
    - in_dir_path:  Directory path that specifies the input directory with files to be copied.
    - out_dir_path: Directory path that specifies the destination point for the copied files.
    - [seed]:       The seed for the random selection a patients frame (default: 42).

    This method selects a random frame for each patient in the in_dir_path.
    The required naming convention is the following: <patientNr>_<frameNr>.nii.gz
    Note: This method also changes the naming convention to: <patientNr>-<frameNr>.nii.gz
    """

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    random.seed(seed)
    copied_files = {}

    for filename in os.listdir(in_dir_path):
        if filename.endswith(".nii.gz"):
            # Extract patient name and frame number from the filename
            parts = filename.split("_")
            patient = parts[0]
            frame = parts[1].split(".")[0]

            # Check if this patient has already been copied
            if patient not in copied_files:
                # Generate a list of all frames for this patient
                frames = [f for f in os.listdir(in_dir_path) if f.startswith(f"{patient}_")]

                # Check if there are frames for this patient
                if frames:
                    # Rename the randomly selected file with <patient>-<frame>.nii.gz format
                    random_frame = random.choice(frames)
                    new_filename = f"{patient}-{frame}.nii.gz"

                    # Copy the file to the output directory with the new filename
                    source_file_path = os.path.join(in_dir_path, random_frame)
                    output_file_path = os.path.join(out_dir_path, new_filename)
                    shutil.copyfile(source_file_path, output_file_path)

                    copied_files[patient] = new_filename

    print("Files copied and renamed successfully.")



def main():

    print("")
    # 1. Split 4d images into frames
    #adac_split_4d(in_dir_path=r"path", 
    #              out_dir_path=r"", copy_all=False)

    # 2. Manually put all images together in one folder
    #    (New dataset split is performed afterwards)

    # 3. For each patient select a random frame
    #select_frames(in_dir_path=r"path", 
    #              out_dir_path=r"path", seed=42)


if __name__ == '__main__':
    main()
