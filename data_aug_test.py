from data_aug import augment_image_in_four_intensities, random_blur, random_downsample, random_ghosting, random_motion, random_noise, random_spike
import torch
import torchio as tio
import os
import SimpleITK as sitk
import numpy as np 
from mp.data.pytorch.transformation import centre_crop_pad_3d

def augment_data_aug_motion_test(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact motion. 
        It saves the augmented images with and without fft, 
        because the motion classifier uses fft images.
    """

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    motion_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for filename in filenames:

        if 'motion5' in filename: intensities = [5] #[1,2,3,4,5]
        elif 'motion1' in filename or 'motion2' in filename or 'motion3' in filename or 'motion4' in filename: intensities = [5]
        else: continue
        print('motion: ' +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename))

        # Augment image
        
        for val in intensities:
            motion_count[str(val)]=motion_count[str(val)]+1
            # Augment image
            motion = augment_image_in_four_intensities(img, 'motion',val)

            
            x_s = torch.from_numpy(sitk.GetArrayFromImage(motion)).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            motion = centre_crop_pad_3d(x_s, img_size)[0]
            motion = np.transpose(motion, (2,1,0))
            motion_without_fft = motion

            # Transform the augmented image to fft
            fft_trans = np.fft.fftn(motion)
            fft_trans_abs = np.abs(fft_trans)
            motion = fft_trans_abs

            # Save images
            if 'motion1' in filename: val = 1
            elif 'motion2' in filename: val = 2
            elif 'motion3' in filename: val = 3
            elif 'motion4' in filename: val = 4
            elif 'motion5' in filename: val = val
            else: continue
            
            a_filename = filename.split('_')[0] + '_' + 'motion' + str(val) #NEW s.blur

            # Save fft images
            try:
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(motion)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
            except: continue
            
            # Save imaged without fft
            try:
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(motion_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))
            except: continue
            
def augment_data_aug_blur_test(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    blur_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for filename in filenames:
    
        if 'blur5' in filename: intensities = [5] #[1,2,3,4,5]
        elif 'blur1' in filename or 'blur2' in filename or 'blur3' in filename or 'blur4' in filename: intensities = [5]
        else: continue
        print('blur: ' +'/' + str(len(filenames)))

        img = sitk.ReadImage(os.path.join(source_path, filename))

        # Augment image
        
        for val in intensities: 
            blur_count[str(val)]=blur_count[str(val)]+1
            # Augment image
            blur, _ = augment_image_in_four_intensities(img, 'blur',val)

            
            x_s = torch.from_numpy(sitk.GetArrayFromImage(blur)).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            blur = centre_crop_pad_3d(x_s, img_size)[0]
            blur = np.transpose(blur, (2,1,0))
            

            # Save images
            if 'blur1' in filename: val = 1
            elif 'blur2' in filename: val = 2
            elif 'blur3' in filename: val = 3
            elif 'blur4' in filename: val = 4
            elif 'blur5' in filename: val = val
            else: continue
            
            a_filename = filename.split('_')[0] + '_' + 'blur' + str(val) #NEW s.blur

            # Save images
            try:
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(blur)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
            except: continue

def augment_data_aug_ghosting_test(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact ghosting. 
        It saves the augmented images with and without fft, 
        because the ghosting classifier uses fft images.
    """

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    ghosting_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for filename in filenames:

        if 'ghosting5' in filename: intensities = [5] #[1,2,3,4,5]
        elif 'ghosting1' in filename or 'ghosting2' in filename or 'ghosting3' in filename or 'ghostingn4' in filename: intensities = [5]
        else: continue

        print('ghosting: ' +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename))

        # Augment image
        
        for val in intensities: 

            ghosting_count[str(val)]=ghosting_count[str(val)]+1
            # Augment image
            ghosting = augment_image_in_four_intensities(img, 'ghosting',val)

            
            x_s = torch.from_numpy(sitk.GetArrayFromImage(ghosting)).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            ghosting = centre_crop_pad_3d(x_s, img_size)[0]
            ghosting = np.transpose(ghosting, (2,1,0))
            ghosting_without_fft = ghosting

            # Transform the augmented image to fft
            fft_trans = np.fft.fftn(ghosting)
            fft_trans_abs = np.abs(fft_trans)
            ghosting = fft_trans_abs

            # Save images
            if 'ghosting1' in filename: val = 1
            elif 'ghosting2' in filename: val = 2
            elif 'ghosting3' in filename: val = 3
            elif 'ghosting4' in filename: val = 4
            elif 'ghosting5' in filename: val = val
            else: continue
            
            a_filename = filename.split('_')[0] + '_' + 'ghosting' + str(val) #NEW s.blur

            # Save fft images
            try:
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(ghosting)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
            except: continue
            
            # Save imaged without fft
            try:
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(ghosting_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))
            except: continue

def augment_data_aug_spike_test(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact spike. 
        It saves the augmented images with and without fft, 
        because the spike classifier uses fft images.
    """

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    spike_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for filename in filenames:
        
        if 'spike5' in filename: intensities = [5] #[1,2,3,4,5]
        elif 'spike1' in filename or 'spike2' in filename or 'spike3' in filename or 'spike4' in filename: intensities = [5]
        else: continue

        print('spike: ' +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename))

        # Augment image
        
        for val in intensities: 

            spike_count[str(val)]=spike_count[str(val)]+1
            # Augment image
            spike = augment_image_in_four_intensities(img, 'spike',val)

            
            x_s = torch.from_numpy(sitk.GetArrayFromImage(spike)).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            spike = centre_crop_pad_3d(x_s, img_size)[0]
            spike = np.transpose(spike, (2,1,0))
            spike_without_fft = spike

            # Transform the augmented image to fft
            fft_trans = np.fft.fftn(spike)
            fft_trans_abs = np.abs(fft_trans)
            spike = fft_trans_abs

            # Save images
            if 'spike1' in filename: val = 1
            elif 'spike2' in filename: val = 2
            elif 'spike3' in filename: val = 3
            elif 'spike4' in filename: val = 4
            elif 'spike5' in filename: val = val
            else: continue
            
            a_filename = filename.split('_')[0] + '_' + 'spike' + str(val) #NEW s.blur
            print(a_filename)
            
            # Save fft images
            try:
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(spike)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
            except: continue
            
            # Save imaged without fft
            try:
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(spike_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))
            except: continue
      
def augment_data_aug_noise_test(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    noise_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for filename in filenames:

        if 'noise5' in filename: intensities = [5] #[1,2,3,4,5]
        elif 'noise1' in filename or 'noise2' in filename or 'noise3' in filename or 'noise4' in filename: intensities = [5]
        else: continue
        print('noise: ' +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename))

        # Augment image
        
        for val in intensities: 
            noise_count[str(val)]=noise_count[str(val)]+1
            # Augment image
            noise = augment_image_in_four_intensities(img, 'noise',val)

            
            x_s = torch.from_numpy(sitk.GetArrayFromImage(noise)).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            noise = centre_crop_pad_3d(x_s, img_size)[0]
            noise = np.transpose(noise, (2,1,0))
            

            # Save images
            if 'noise1' in filename: val = 1
            elif 'noise2' in filename: val = 2
            elif 'noise3' in filename: val = 3
            elif 'noise4' in filename: val = 4
            elif 'noise5' in filename: val = val
            else: continue
            
            a_filename = filename.split('_')[0] + '_' + 'noise' + str(val) #NEW s.blur

            # Save images
            try:
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(noise)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
            except: continue




