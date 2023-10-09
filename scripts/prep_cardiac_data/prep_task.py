import SimpleITK as sitk
import os
import shutil

def task_reorient(in_dir_path, out_dir_path):
    r'''
        - in_dir_path:      Directory path that specifies the entry point of application.
        - out_dir_path:     Directory path that specifies the output directory for the images.

        This is a preprocess method for the task808 and task809 datasets.
        It copies all .nii.gz images in the in_dir_path, converts them to RAI format, 
        and removes the postfix '_0000' from the image names.
    '''

    if not os.path.exists(in_dir_path):
        print("Input folder does not exist.")
        return

    os.makedirs(out_dir_path, exist_ok=True)

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
        output_file_path = os.path.join(out_dir_path, new_filename + '.nii.gz')
        sitk.WriteImage(image, output_file_path)

        # Set the original spacings back to the reoriented image
        reoriented_image = sitk.ReadImage(output_file_path)
        reoriented_image.SetSpacing(original_spacing)
        sitk.WriteImage(reoriented_image, output_file_path)
    



def main():
    # 1. Reorient images to RAI orientation:
    #task_reorient(in_dir_path=r"path",
    #              out_dir_path=r"path")

if __name__ == '__main__':
    main()
