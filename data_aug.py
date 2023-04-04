# ------------------------------------------------------------------------------
# Data augmentation. Only, transformations from the TorchIO library are 
# used (https://torchio.readthedocs.io/transforms/augmentation.html).
# ------------------------------------------------------------------------------

# Imports
import torchio as tio
import SimpleITK as sitk
import os
import torch
import numpy as np 
from mp.data.pytorch.transformation import centre_crop_pad_3d

# Perfom augmentation on dataset
def augment_image_in_four_intensities(image, noise, additional_intensity=0):
    r"""This function takes an image and augments it in 4 intensities for one of 5 artefacts:
        - Blurring
        - Ghosting
        - Motion
        - (Gaussian) Noise
        - Spike
    """
    
    if noise == 'blur':
        if additional_intensity == 5: return image, additional_intensity
        if additional_intensity == 4: blur = random_blur(std=1.5) 
        if additional_intensity == 3: blur = random_blur(std=3)
        if additional_intensity == 2: blur = random_blur(std=4.5)
        if additional_intensity == 1: blur = random_blur(std=5.75)
        return blur(image), additional_intensity

    if noise == 'ghosting':
        if additional_intensity == 5: return image
        if additional_intensity == 4: return random_ghosting(num_ghosts = (2,2), intensity=(0.3,0.3))(image)
        if additional_intensity == 3: return random_ghosting(num_ghosts = (3,3), intensity=(0.54,0.54))(image)
        if additional_intensity == 2: return random_ghosting(num_ghosts = (4,4), intensity=(0.67,0.67))(image)
        if additional_intensity == 1: return random_ghosting(num_ghosts = (5,5), intensity=(0.75,0.75))(image)


    if noise == 'motion':
        if additional_intensity == 5: return image
        if additional_intensity == 4: return random_motion(degrees=(0.37, 0.37), translation=(0.28,0.28), num_transforms=1)(image)
        if additional_intensity == 3: return random_motion(degrees=(0.47,0.47), translation=(0.38,0.38), num_transforms=2)(image)
        if additional_intensity == 2: return random_motion(degrees=(0.54, 0.54), translation=(0.46,0.46), num_transforms=3)(image)
        if additional_intensity == 1: return random_motion(degrees=(0.59, 0.59), translation=(0.51,0.51), num_transforms=4)(image)


    if noise == 'noise':
        if additional_intensity == 5: return image
        if additional_intensity == 4: noise = random_noise(mean=(0,0),std=(70,70))
        if additional_intensity == 3: noise = random_noise(mean=(0,0),std=(101,101)) 
        if additional_intensity == 2: noise = random_noise(mean=(0,0),std=(148,148)) 
        if additional_intensity == 1: noise = random_noise(mean=(0,0),std=(180,180)) 
        return noise(image)

    if noise == 'spike':
        if additional_intensity == 5: return image
        if additional_intensity == 4: return random_spike(num_spikes=(4,4), intensity=(0.75, 0.75))(image)
        if additional_intensity == 3: return random_spike(num_spikes=(5,5), intensity=(0.95, 0.95))(image)
        if additional_intensity == 2: return random_spike(num_spikes=(6,6), intensity=(1.15, 1.15))(image)
        if additional_intensity == 1: return random_spike(num_spikes=(7,7), intensity=(1.25, 1.25))(image)
        return additional_intensity

    
    if noise == 'resolution':
        if additional_intensity == 5: return image
        if additional_intensity == 4: resolution = random_downsample(axes=2, downsampling=(4,4))
        if additional_intensity == 3: resolution = random_downsample(axes=2, downsampling=(6,6))
        if additional_intensity == 2: resolution = random_downsample(axes=2, downsampling=(8,8))
        if additional_intensity == 1: resolution = random_downsample(axes=2, downsampling=(10,10))
        return resolution(image)



# Intensity Functions for data Augmentation

def random_blur(std):
    r"""Blur an image using a random-sized Gaussian filter.
    - std: Tuple (a,b) to compute the standard deviations (𝜎1,𝜎2,𝜎3)
        of the Gaussian kernels used to blur the image along each axis.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    blur = tio.RandomBlur(std=std)
    return blur

def random_ghosting(num_ghosts, intensity):
    r"""Add random MRI ghosting artifact.
    - num_ghosts: Number of ‘ghosts’ n in the image.
    - axes: Axis along which the ghosts will be created. If axes is a
        tuple, the axis will be randomly chosen from the passed values.
        Anatomical labels may also be used.
    - intensity: Positive number representing the artifact strength s
        with respect to the maximum of the k-space. If 0, the ghosts
        will not be visible.
    - restore: Number between 0 and 1 indicating how much of the k-space
        center should be restored after removing the planes that generate
        the artifact.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    ghosting = tio.RandomGhosting(num_ghosts=num_ghosts, intensity=intensity)
    return ghosting

def random_motion(degrees=10, translation=10, num_transforms=2):
    r"""Add random MRI motion artifact.
    - degrees: Tuple (a, b) defining the rotation range in degrees of
        the simulated movements.
    - translation: Tuple (a,b) defining the translation in mm
        of the simulated movements.
    - num_transforms: Number of simulated movements. Larger values generate
        more distorted images.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    motion = tio.RandomMotion(degrees, translation, num_transforms)
    return motion

def random_noise(mean, std):
    r"""Add random Gaussian noise.
    - mean: Mean μ of the Gaussian distribution from which the noise is sampled.
    - std: Standard deviation σ of the Gaussian distribution from which
        the noise is sampled.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    noise = tio.RandomNoise(mean, std)
    return noise

def random_spike(num_spikes, intensity):
    r"""Add random MRI spike artifacts.
    - num_spikes: Number of spikes n present in k-space. Larger values generate
        more distorted images.
    - intensity: Ratio r between the spike intensity and the maximum of
        the spectrum. Larger values generate more distorted images.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    spike = tio.RandomSpike(num_spikes, intensity)
    return spike

def random_downsample(axes=(0, 1, 2), downsampling=(1.5, 5), p=1,
                      seed=None, keys=None):
    r"""Downsample an image along an axis. This transform simulates an
    image that has been acquired using anisotropic spacing, using
    downsampling with nearest neighbor interpolation.
    - axes: Axis or tuple of axes along which the image will be downsampled.
    - downsampling: Downsampling factor m > 1.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    resolution = tio.RandomDownsample(axes, downsampling, p, seed, keys)
    return resolution




def augment_data_aug_motion(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact motion. 
        It saves the augmented images with and without fft, 
        because the motion classifier uses fft images.
    """

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities
    diff = False
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]


    motion_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for filename in filenames:
        if 'motion5' in filename: #NEW s.blur
            print('motion: ' +'/' + str(len(filenames)))
            img = sitk.ReadImage(os.path.join(source_path, filename))
            img_array = sitk.GetArrayFromImage(img)

            # Centercrop and pad the image without artifact
            if diff:
                img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
                img_array_crop = np.transpose(img_array_crop, (2,1,0))

            # Augment image
            rand1 = np.random.randint(1,6)
            rand2 = np.random.randint(1,6)
            
            if motion_count[str(rand1)]<num_img_per_class:
                val1 = rand1
            elif (rand1+1)%6+int((rand1+1)/6) != rand2 and motion_count[str((rand1+1)%6+int((rand1+1)/6))]<num_img_per_class:
                val1 = (rand1+1)%6+int((rand1+1)/6)
            elif (rand1+2)%6+int((rand1+2)/6) != rand2 and motion_count[str((rand1+2)%6+int((rand1+2)/6))]<num_img_per_class:
                val1 = (rand1+2)%6+int((rand1+2)/6)
            elif (rand1+3)%6+int((rand1+3)/6) != rand2 and motion_count[str((rand1+3)%6+int((rand1+3)/6))]<num_img_per_class:
                val1 = (rand1+3)%6+int((rand1+3)/6)
            elif (rand1+4)%6+int((rand1+4)/6) != rand2 and motion_count[str((rand1+4)%6+int((rand1+4)/6))]<num_img_per_class:
                val1 = (rand1+4)%6+int((rand1+4)/6)
            else:
                val1 = None
                
            
            if rand2 !=val1 and motion_count[str(rand2)]<num_img_per_class:
                val2=rand2
            elif (rand2+1)%6+int((rand2+1)/6) != val1 and  motion_count[str((rand2+1)%6+int((rand2+1)/6))]<num_img_per_class:
                val2 = (rand2+1)%6+int((rand2+1)/6)
            elif (rand2+2)%6+int((rand2+2)/6) != val1 and motion_count[str((rand2+2)%6+int((rand2+2)/6))]<num_img_per_class:
                val2 = (rand2+2)%6+int((rand2+2)/6)
            elif (rand2+3)%6+int((rand2+3)/6) != val1 and motion_count[str((rand2+3)%6+int((rand2+3)/6))]<num_img_per_class:
                val2 = (rand2+3)%6+int((rand2+3)/6)
            elif (rand2+4)%6+int((rand2+4)/6) != val1 and motion_count[str((rand2+4)%6+int((rand2+4)/6))]<num_img_per_class:
                val2 = (rand2+4)%6+int((rand2+4)/6)
            else:
                val2 = None

             
            
            if val1 != None:    
                motion_count[str(val1)]=motion_count[str(val1)]+1
                # Augment image
                motion = augment_image_in_four_intensities(img, 'motion',val1)

                
                x_s = torch.from_numpy(sitk.GetArrayFromImage(motion)).unsqueeze_(0)
                # Centercrop and pad all images to the same size
                motion = centre_crop_pad_3d(x_s, img_size)[0]
                motion = np.transpose(motion, (2,1,0))
                motion_without_fft = motion

                # Calculate the difference of the ffts. 
                if diff:
                    img_basic = np.fft.fftn(img_array_crop)
                    fft_shift_basic = np.fft.fftshift(img_basic)
                    fft_trans_abs_basic = np.abs(fft_shift_basic)
                    fft_trans = np.fft.fftn(motion)
                    fft_shift = np.fft.fftshift(fft_trans)
                    fft_trans_abs = np.abs(fft_shift)

                    fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                    fft_trans_log = np.log(fftdiff)
                    diff_img = fft_trans_log

                # Transform the augmented image to fft
                fft_trans = np.fft.fftn(motion)
                fft_trans_abs = np.abs(fft_trans)
                motion = fft_trans_abs

                # Save images
                
                a_filename = filename.split('_')[0] + '_' + 'motion' + str(val1) #NEW s.blur

                # Save fft images
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(motion)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
                
                # Save imaged without fft
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(motion_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))

                # Save the differences
                if diff:
                    #os.makedirs(os.path.join(target_path, 'diff_images'))
                    x = sitk.GetImageFromArray(diff_img)
                    sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))
                    x_fft = sitk.GetImageFromArray(fft_trans_abs)
                    sitk.WriteImage(x_fft, os.path.join(target_path, 'diff_images', a_filename+'_fftabs.nii.gz'))


            if val2 != None:    
                motion_count[str(val2)]=motion_count[str(val2)]+1
                # Augment image
                motion = augment_image_in_four_intensities(img, 'motion',val2)

                
                x_s = torch.from_numpy(sitk.GetArrayFromImage(motion)).unsqueeze_(0)
                # Centercrop and pad all images to the same size
                motion = centre_crop_pad_3d(x_s, img_size)[0]
                motion = np.transpose(motion, (2,1,0))
                motion_without_fft = motion

                # Calculate the difference of the ffts. 
                if diff:
                    img_basic = np.fft.fftn(img_array_crop)
                    fft_shift_basic = np.fft.fftshift(img_basic)
                    fft_trans_abs_basic = np.abs(fft_shift_basic)
                    fft_trans = np.fft.fftn(motion)
                    fft_shift = np.fft.fftshift(fft_trans)
                    fft_trans_abs = np.abs(fft_shift)

                    fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                    fft_trans_log = np.log(fftdiff)
                    diff_img = fft_trans_log

                # Transform the augmented image to fft
                fft_trans = np.fft.fftn(motion)
                fft_trans_abs = np.abs(fft_trans)
                motion = fft_trans_abs

                # Save images
                
                a_filename = filename.split('_')[0] + '_' + 'motion' + str(val2) #NEW s.blur

                # Save fft images
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(motion)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
                
                # Save imaged without fft
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(motion_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))

                # Save the differences
                if diff:
                    x = sitk.GetImageFromArray(diff_img)
                    sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))
                    x_fft = sitk.GetImageFromArray(fft_trans_abs)
                    sitk.WriteImage(x_fft, os.path.join(target_path, 'diff_images', a_filename+'_fftabs.nii.gz'))
            


def augment_data_aug_blur(source_path, target_path, without_fft_path=None, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact blur. 
    """
    def save_img_unchanged(intensity):
        img = sitk.ReadImage(os.path.join(source_path, filename))
        x_s = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze_(0)
        blur = centre_crop_pad_3d(x_s, img_size)[0]
        blur_trans = np.transpose(blur, (2,1,0))
        
        blur_without_fft = blur_trans
        fft_trans = np.fft.fftn(blur_trans)
        fft_trans_abs = np.abs(fft_trans)
        blur = fft_trans_abs
        
        a_filename = filename.split('_')[0] + '_' + 'blur' + str(intensity)
        
        #save without fft
        os.makedirs(os.path.join(target_path, a_filename, 'img'))
        x = sitk.GetImageFromArray(blur_without_fft)
        sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
        
        #save fft
        os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
        x_not_fft = sitk.GetImageFromArray(blur)
        sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))


    def augment_img(original_intensity, additional_intensity):
        img = sitk.ReadImage(os.path.join(source_path, filename))
        blur, additional_intensity = augment_image_in_four_intensities(img, 'blur', additional_intensity)
        new_intensity = additional_intensity - (5 - original_intensity)  
        x_s = torch.from_numpy(sitk.GetArrayFromImage(blur)).unsqueeze_(0)
        blur = centre_crop_pad_3d(x_s, img_size)[0]
        blur_trans = np.transpose(blur, (2,1,0))
        
        blur_without_fft = blur_trans
        fft_trans = np.fft.fftn(blur_trans)
        fft_trans_abs = np.abs(fft_trans)
        blur = fft_trans_abs
        
        a_filename = filename.split('_')[0] + '_' + 'blur' + str(new_intensity)
        
        #save without fft
        os.makedirs(os.path.join(target_path, a_filename, 'img'))
        x = sitk.GetImageFromArray(blur_without_fft)
        sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
        
        #save fft
        os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
        x_not_fft = sitk.GetImageFromArray(blur)
        sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))



    torch.manual_seed(42)
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    count_img = 0
    count_all = 0
    num_img_per_class = 48
    blur_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    # Loop through filenames to augment and save every image
    for filename in filenames:
        count_all += 1
        if 'blur5' in filename: #NEW hinzugefügt wegen handlabeln (wir wollen nur perfekte Bilder verwenden)                    
            print('blur: ' + str(filename+1) +'/' + str(len(filenames)))
            count_img +=1
            rand1 = np.random.randint(1,6)
            rand2 = np.random.randint(1,6)
            rand3 = np.random.randint(1,6)
            rand4 = 10#np.random.randint(1,6)

            if blur_count[str(rand1)]<num_img_per_class:
                val1 = rand1
            elif (rand1+1)%6+int((rand1+1)/6) != rand2 and (rand1+1)%6+int((rand1+1)/6) != rand3 and (rand1+1)%6+int((rand1+1)/6) != rand4 and blur_count[str((rand1+1)%6+int((rand1+1)/6))]<num_img_per_class:
                val1 = (rand1+1)%6+int((rand1+1)/6)
            elif (rand1+2)%6+int((rand1+2)/6) != rand2 and (rand1+2)%6+int((rand1+2)/6) != rand3 and (rand1+2)%6+int((rand1+2)/6) != rand4 and blur_count[str((rand1+2)%6+int((rand1+2)/6))]<num_img_per_class:
                val1 = (rand1+2)%6+int((rand1+2)/6)
            elif (rand1+3)%6+int((rand1+3)/6) != rand2 and (rand1+3)%6+int((rand1+3)/6) != rand3 and (rand1+3)%6+int((rand1+3)/6) != rand4 and blur_count[str((rand1+3)%6+int((rand1+3)/6))]<num_img_per_class:
                val1 = (rand1+3)%6+int((rand1+3)/6)
            elif (rand1+4)%6+int((rand1+4)/6) != rand2 and (rand1+4)%6+int((rand1+4)/6) != rand3 and (rand1+4)%6+int((rand1+4)/6) != rand4 and blur_count[str((rand1+4)%6+int((rand1+4)/6))]<num_img_per_class:
                val1 = (rand1+4)%6+int((rand1+4)/6)
            else:
                val1 = None
                
            if val1 == 5:
                save_img_unchanged(intensity=5)
                blur_count[str(val1)]=blur_count[str(val1)]+1
            elif val1:
                augment_img(original_intensity=5, additional_intensity=val1)
                blur_count[str(val1)]=blur_count[str(val1)]+1



            if rand2 !=val1 and blur_count[str(rand2)]<num_img_per_class:
                val2=rand2
            elif (rand2+1)%6+int((rand2+1)/6) != val1 and (rand2+1)%6+int((rand2+1)/6) != rand3 and (rand2+1)%6+int((rand2+1)/6) != rand4 and blur_count[str((rand2+1)%6+int((rand2+1)/6))]<num_img_per_class:
                val2 = (rand2+1)%6+int((rand2+1)/6)
            elif (rand2+2)%6+int((rand2+2)/6) != val1 and (rand2+2)%6+int((rand2+2)/6) != rand3 and (rand2+2)%6+int((rand2+2)/6) != rand4 and blur_count[str((rand2+2)%6+int((rand2+2)/6))]<num_img_per_class:
                val2 = (rand2+2)%6+int((rand2+2)/6)
            elif (rand2+3)%6+int((rand2+3)/6) != val1 and (rand2+3)%6+int((rand2+3)/6) != rand3 and (rand2+3)%6+int((rand2+3)/6) != rand4 and blur_count[str((rand2+3)%6+int((rand2+3)/6))]<num_img_per_class:
                val2 = (rand2+3)%6+int((rand2+3)/6)
            elif (rand2+4)%6+int((rand2+4)/6) != val1 and (rand2+4)%6+int((rand2+4)/6) != rand3 and (rand2+4)%6+int((rand2+4)/6) != rand4 and blur_count[str((rand2+4)%6+int((rand2+4)/6))]<num_img_per_class:
                val2 = (rand2+4)%6+int((rand2+4)/6)
            else:
                val2 = None
                
            if val2 == 5:
                save_img_unchanged(intensity=5)
                blur_count[str(val2)]=blur_count[str(val2)]+1
            elif val2:
                augment_img(original_intensity=5, additional_intensity=val2)
                blur_count[str(val2)]=blur_count[str(val2)]+1

            
            if rand3 !=val1 and rand3 != val2 and blur_count[str(rand3)]<num_img_per_class:
                val3=rand3
            elif (rand3+1)%6+int((rand3+1)/6) != val1 and (rand3+1)%6+int((rand3+1)/6) != val2 and (rand3+1)%6+int((rand3+1)/6) != rand4 and blur_count[str((rand3+1)%6+int((rand3+1)/6))]<num_img_per_class:
                val3 = (rand3+1)%6+int((rand3+1)/6)
            elif (rand3+2)%6+int((rand3+2)/6) != val1 and (rand3+2)%6+int((rand3+2)/6) != val2 and (rand3+2)%6+int((rand3+2)/6) != rand4 and blur_count[str((rand3+2)%6+int((rand3+2)/6))]<num_img_per_class:
                val3 = (rand3+2)%6+int((rand3+2)/6)
            elif (rand3+3)%6+int((rand3+3)/6) != val1 and (rand3+3)%6+int((rand3+3)/6) != val2 and (rand3+3)%6+int((rand3+3)/6) != rand4 and blur_count[str((rand3+3)%6+int((rand3+3)/6))]<num_img_per_class:
                val3 = (rand3+3)%6+int((rand3+3)/6)
            elif (rand3+4)%6+int((rand3+4)/6) != val1 and (rand3+4)%6+int((rand3+4)/6) != val2 and (rand3+4)%6+int((rand3+4)/6) != rand4 and blur_count[str((rand3+4)%6+int((rand3+4)/6))]<num_img_per_class:
                val3 = (rand3+4)%6+int((rand3+4)/6)
            else:
                val3 = None
                
            if val3 == 5:
                save_img_unchanged(intensity=5)
                blur_count[str(val3)]=blur_count[str(val3)]+1
            elif val3:
                augment_img(original_intensity=5, additional_intensity=val3)
                blur_count[str(val3)]=blur_count[str(val3)]+1




def augment_data_aug_ghosting(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact ghosting. 
        It saves the augmented images with and without fft, 
        because the ghosting classifier uses fft images.
    """
    
    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities
    diff = False
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    # Loop through filenames to augment and save every image
    ghosting_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    for num, filename in enumerate(filenames):
        if 'ghosting5' in filename: 
            print('ghosting: ' + str(num+1) +'/' + str(len(filenames)))
            img = sitk.ReadImage(os.path.join(source_path, filename))        
            img_array = sitk.GetArrayFromImage(img)
            # Centercrop and pad the image without artifact
            if diff:
                img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
                img_array_crop = np.transpose(img_array_crop, (2,1,0))
            # Augment image
            
            rand = np.random.randint(1,6)
            if ghosting_count[str(rand)]<num_img_per_class:
                val1 = rand
            elif ghosting_count[str((rand+1)%6+int((rand+1)/6))]<num_img_per_class:
                val1 = (rand+1)%6+int((rand+1)/6)
            elif ghosting_count[str((rand+2)%6+int((rand+2)/6))]<num_img_per_class:
                val1 = (rand+2)%6+int((rand+2)/6)
            elif ghosting_count[str((rand+3)%6+int((rand+3)/6))]<num_img_per_class:
                val1 = (rand+3)%6+int((rand+3)/6)
            elif ghosting_count[str((rand+4)%6+int((rand+4)/6))]<num_img_per_class:
                val1 = (rand+4)%6+int((rand+4)/6)
            else:
                val1 = None
             
            
            if val1 != None:    
                ghosting_count[str(val1)]=ghosting_count[str(val1)]+1
                ghosting= augment_image_in_four_intensities(img, 'ghosting',val1)

                x_s = torch.from_numpy(sitk.GetArrayFromImage(ghosting)).unsqueeze_(0)
                # Centercrop and pad all images to the same size
                ghosting = centre_crop_pad_3d(x_s, img_size)[0]
                ghosting = np.transpose(ghosting, (2,1,0))
                ghosting_without_fft = ghosting
                # Calculate the difference of the ffts. 
                if diff:
                    img_basic = np.fft.fftn(img_array_crop)
                    fft_shift_basic = np.fft.fftshift(img_basic)
                    fft_trans_abs_basic = np.abs(fft_shift_basic)
                    fft_trans = np.fft.fftn(ghosting)
                    fft_shift = np.fft.fftshift(fft_trans)
                    fft_trans_abs = np.abs(fft_shift)

                    fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                    fft_trans_log = np.log(fftdiff)
                    diff_img = fft_trans_log
                # Transform the augmented image to fft
                fft_trans = np.fft.fftn(ghosting)
                fft_trans_abs = np.abs(fft_trans)
                ghosting = fft_trans_abs
                # Save images
                a_filename = filename.split('_')[0] + '_' + 'ghosting' + str(val1) #NEW s.blur
                
                # Save fft images
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(ghosting)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
                
                # Save imaged without fft
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(ghosting_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))
                # Save the differences
                if diff:
                    #os.makedirs(os.path.join(target_path, 'diff_images'))
                    x = sitk.GetImageFromArray(diff_img)
                    sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))
                    x_fft = sitk.GetImageFromArray(fft_trans_abs)
                    sitk.WriteImage(x_fft, os.path.join(target_path, 'diff_images', a_filename+'_fftabs.nii.gz'))


def augment_data_aug_noise(source_path, target_path, without_fft_path = None, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact noise. 
    """
    
    torch.manual_seed(42)
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    
    noise_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        if 'noise5' in filename:
            print('noise: ' + str(num+1) +'/' + str(len(filenames)))
            img = sitk.ReadImage(os.path.join(source_path, filename))        

            rand = np.random.randint(1,6)
            if noise_count[str(rand)]<num_img_per_class:
                val1 = rand
            elif noise_count[str((rand+1)%6+int((rand+1)/6))]<num_img_per_class:
                val1 = (rand+1)%6+int((rand+1)/6)
            elif noise_count[str((rand+2)%6+int((rand+2)/6))]<num_img_per_class:
                val1 = (rand+2)%6+int((rand+2)/6)
            elif noise_count[str((rand+3)%6+int((rand+3)/6))]<num_img_per_class:
                val1 = (rand+3)%6+int((rand+3)/6)
            elif noise_count[str((rand+4)%6+int((rand+4)/6))]<num_img_per_class:
                val1 = (rand+4)%6+int((rand+4)/6)
            else:
                val1 = None

            if val1 != None:    
                noise_count[str(val1)]=noise_count[str(val1)]+1
                # Augment image
                noise = augment_image_in_four_intensities(img, 'noise',val1)
                x_s = torch.from_numpy(sitk.GetArrayFromImage(noise)).unsqueeze_(0)
                # Centercrop and pad all images to the same size
                noise = centre_crop_pad_3d(x_s, img_size)[0]
                noise = np.transpose(noise, (2,1,0))
                
                noise_without_fft = noise
                fft_trans = np.fft.fftn(noise)
                fft_trans_abs = np.abs(fft_trans)
                noise = fft_trans_abs   
                           
                # Save images
                a_filename = filename.split('_')[0] + '_' + 'noise' + str(val1)
                
                # Save images 
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(noise_without_fft)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
                
                # Save imaged with fft 
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(noise)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))


def augment_data_aug_spike(source_path, target_path, without_fft_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact spike. 
        It saves the augmented images with and without fft, 
        because the spike classifier uses fft images.
    """
    
    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities
    diff = False

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    
    spike_count = {'5':0, '4':0, '3':0, '2':0, '1':0}
    num_img_per_class = 48
    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        if 'spike5' in filename: #NEW s.blur
            print('spike: ' + str(num+1) +'/' + str(len(filenames)))
            

            img = sitk.ReadImage(os.path.join(source_path, filename))        
            img_array = sitk.GetArrayFromImage(img)

            # Centercrop and pad the image without artifact
            if diff:
                img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
                img_array_crop = np.transpose(img_array_crop, (2,1,0))
                
            rand1 = np.random.randint(1,6)
            rand2 = 10#np.random.randint(1,6)

            if spike_count[str(rand1)]<num_img_per_class:
                val1 = rand1
            elif (rand1+1)%6+int((rand1+1)/6) != rand2 and spike_count[str((rand1+1)%6+int((rand1+1)/6))]<num_img_per_class:
                val1 = (rand1+1)%6+int((rand1+1)/6)
            elif (rand1+2)%6+int((rand1+2)/6) != rand2 and spike_count[str((rand1+2)%6+int((rand1+2)/6))]<num_img_per_class:
                val1 = (rand1+2)%6+int((rand1+2)/6)
            elif (rand1+3)%6+int((rand1+3)/6) != rand2 and spike_count[str((rand1+3)%6+int((rand1+3)/6))]<num_img_per_class:
                val1 = (rand1+3)%6+int((rand1+3)/6)
            elif (rand1+4)%6+int((rand1+4)/6) != rand2 and spike_count[str((rand1+4)%6+int((rand1+4)/6))]<num_img_per_class:
                val1 = (rand1+4)%6+int((rand1+4)/6)
            else:
                val1 = None
        
            
            if val1 != None:    
                spike_count[str(val1)]=spike_count[str(val1)]+1
                # Augment image
                spike = augment_image_in_four_intensities(img, 'spike',val1)

                
                x_s = torch.from_numpy(sitk.GetArrayFromImage(spike)).unsqueeze_(0)
                # Centercrop and pad all images to the same size
                spike = centre_crop_pad_3d(x_s, img_size)[0]
                spike = np.transpose(spike, (2,1,0))
                spike_without_fft = spike

                # Calculate the difference of the ffts. 
                if diff:
                    img_basic = np.fft.fftn(img_array_crop)
                    fft_shift_basic = np.fft.fftshift(img_basic)
                    fft_trans_abs_basic = np.abs(fft_shift_basic)
                    fft_trans = np.fft.fftn(spike)
                    fft_shift = np.fft.fftshift(fft_trans)
                    fft_trans_abs = np.abs(fft_shift)

                    fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                    fft_trans_log = np.log(fftdiff)
                    diff_img = fft_trans_log

                # Transform the augmented image to fft
                fft_trans = np.fft.fftn(spike)
                fft_trans_abs = np.abs(fft_trans)
                spike = fft_trans_abs

                # Save images
                
                a_filename = filename.split('_')[0] + '_' + 'spike' + str(val1) #NEW s.blur

                # Save fft images
                os.makedirs(os.path.join(target_path, a_filename, 'img'))
                x = sitk.GetImageFromArray(spike)
                sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
                
                # Save imaged without fft
                os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
                x_not_fft = sitk.GetImageFromArray(spike_without_fft)
                sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))

                # Save the differences
                if diff:
                    #os.makedirs(os.path.join(target_path, 'diff_images'))
                    x = sitk.GetImageFromArray(diff_img)
                    sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))
                    x_fft = sitk.GetImageFromArray(fft_trans_abs)
                    sitk.WriteImage(x_fft, os.path.join(target_path, 'diff_images', a_filename+'_fftabs.nii.gz'))
                    


def augment_data_aug_resolution(source_path, target_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the artefact noise. 
    """
    
    torch.manual_seed(42)
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        if 'resolution5' in filename: #NEW s.blur
            print('resolution: ' + str(num+1) +'/' + str(len(filenames)))
            img = sitk.ReadImage(os.path.join(source_path, filename))        

            # Augment image
            resolution, rand = augment_image_in_four_intensities(img, 'resolution')
            x_s = torch.from_numpy(sitk.GetArrayFromImage(resolution)).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            resolution = centre_crop_pad_3d(x_s, img_size)[0]
            resolution = np.transpose(resolution, (2,1,0))

            # Save images
            a_filename = filename.split('_')[0] + '_' + 'resolution' + str(rand) #NEW s.blur
            os.makedirs(os.path.join(target_path, a_filename, 'img'))
            x = sitk.GetImageFromArray(resolution)
            sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))




def augmentation_segmentation(source_path, target_path, image_type, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for the segmentation. 
        It brings data to the expected format for the segmentation UNet.
        Image_type is 'img' or 'seg'.
    """

    filenames = [x for x in os.listdir(source_path)]
    for num, filename in enumerate(filenames):
        print('segmentation: ' + str(num+1) +'/' + str(len(filenames)))

        if image_type == 'img':
            img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))        
            img_array = sitk.GetArrayFromImage(img)
            # Centercrop and pad all images to the same size
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            for i in range(img_array_crop.shape[2]):
                slice = img_array_crop[:,:,i]
                x = sitk.GetImageFromArray(slice)
                sitk.WriteImage(x, os.path.join(target_path, 'data', 'imgs', filename+'_'+str(i)+'.nii.gz'))
            
        elif image_type == 'seg':
            img = sitk.ReadImage(os.path.join(source_path, filename,'seg', '001.nii.gz'))        
            img_array = sitk.GetArrayFromImage(img)
            img_array = img_array.astype(np.int32)
            # Centercrop and pad all segmentations to the same size
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            for i in range(img_array_crop.shape[2]):
                slice = img_array_crop[:,:,i]
                x = sitk.GetImageFromArray(slice)
                sitk.WriteImage(x, os.path.join(target_path, 'data', 'masks', filename+'_'+str(i)+'.nii.gz'))


# Augments Data for inference
def augmentation_inference(source_path, target_path, img_size=(1, 60, 299, 299)):
    r"""This function augments Data for inference. 
    """

    filenames = [x for x in os.listdir(source_path)]
    for num, filename in enumerate(filenames):
        print('inference augmentation: ' + str(num+1) +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz')) 
        img_array = sitk.GetArrayFromImage(img)
        
        # Centercrop and pad all images to the same size
        img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
        img_array_crop = np.transpose(img_array_crop, (2,1,0))
        
        # Transform the image to fft
        fft_trans = np.fft.fftn(img_array_crop)
        fft_trans_abs = np.abs(fft_trans)
        img_fft = fft_trans_abs

        img_no_fft = sitk.GetImageFromArray(img_array_crop)
        img_with_fft = sitk.GetImageFromArray(img_fft)

        # Save the fft image and the image without fft
        os.makedirs(os.path.join(target_path, filename, 'img'))
        sitk.WriteImage(img_no_fft, os.path.join(target_path, filename, 'img', 'img.nii.gz'))
        sitk.WriteImage(img_with_fft, os.path.join(target_path, filename, 'img', 'fft.nii.gz'))





