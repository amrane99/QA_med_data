import os
import shutil
import SimpleITK as sitk
from names_Task300 import get_Task300_labels 
from names_Task100 import get_Task100_labels
from names_Task200 import get_Task200_labels
from names_Task541 import get_Task541_labels
from names_Task542 import get_Task542_labels
from names_Task06_Train import get_Task06_labels


def merch():
    traintest = 'test'
    datasets = {'a':'Task06_Lung', 'b':'Task100_RadiopediaTrain', 'c':'Task101_RadiopediaTest', 'd':'Task200_MosmedTrain', 
                'e':'Task201_MosmedTest', 'f':'Task300_Challenge', 'g':'Task541_FrankfurtTrainF4', 'h':'Task542_FrankfurtTestF4'}


    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    train = ['a', 'b', 'd', 'f']
    test = ['g', 'h']
    if traintest == 'train':

        use = train
        for set in use:
            source_path = os.path.join('/local/scratch/Racoon_Workflows/Thorax_QA/data_', datasets[set], 'imagesTr')
            if set =='a':
                source_path = '/local/scratch/Racoon_tn/Thorax_QA/data/Task06_Lung/Task06_Lung/imagesTr'
            target_path = os.path.join('/local/scratch/Racoon_tn/Thorax_QA/JIP/data_dirs/temp/train_dirs/input')
            study_names = [x for x in os.listdir(source_path) if '._' not in x and '.nii.gz' in x]
            
            if set == 'f':
                labels = get_Task300_labels()
                #b = 40
                b=199 #NEW für neue augmentierung für Task300
                #if set == 'b': b = 16
                for i in range(b):
                    print(i)
                    name = study_names[i]
                    source_img = os.path.join(source_path, name)
                    for artifact in ['blur', 'noise', 'ghosting', 'resolution', 'motion', 'spike']:
                        shutil.copy(source_img, target_path)
                        target_img = os.path.join(target_path, name)
                        search_name = 'GC_Corona_'+name[:-12]+'_'+artifact
                        try:
                            intensity = int(labels[search_name]*5)
                        except:
                            print(search_name)
                            continue

                        new_name = os.path.join(target_path, set+'-'+name[:-12]+'_'+artifact+str(intensity)+'.nii.gz')
                        os.rename(target_img, new_name)
                    i+=1

            if set == 'b':
                labels = get_Task100_labels()
                #b = 40
                b=16 #NEW für neue augmentierung für Task300
                #if set == 'b': b = 16
                for i in range(b):
                    print(i)
                    name = study_names[i]
                    source_img = os.path.join(source_path, name)
                    for artifact in ['blur', 'noise', 'ghosting', 'resolution', 'motion', 'spike']:
                        shutil.copy(source_img, target_path)
                        target_img = os.path.join(target_path, name)
                        search_name = 'Radiopedia_'+name[:-12]+'_'+artifact
                        try:
                            intensity = int(labels[search_name]*5)
                        except:
                            print(search_name)
                            continue

                        new_name = os.path.join(target_path, set+'-'+name[:-12]+'_'+artifact+str(intensity)+'.nii.gz')
                        os.rename(target_img, new_name)
                    i+=1

            if set == 'd':
                    labels = get_Task200_labels()
                    #b = 40
                    b=40 #NEW für neue augmentierung für Task300
                    #if set == 'b': b = 16
                    for i in range(b):
                        print(i)
                        name = study_names[i]
                        source_img = os.path.join(source_path, name)
                        for artifact in ['blur', 'noise', 'ghosting', 'resolution', 'motion', 'spike']:
                            shutil.copy(source_img, target_path)
                            target_img = os.path.join(target_path, name)
                            search_name = 'Mosmed_'+name[:-12]+'_'+artifact
                            try:
                                intensity = int(labels[search_name]*5)
                            except:
                                print(search_name)
                                continue

                            new_name = os.path.join(target_path, set+'-'+name[:-12]+'_'+artifact+str(intensity)+'.nii.gz')
                            os.rename(target_img, new_name)
                        i+=1
                        
            if set == 'a':
                    labels = get_Task06_labels()
                    b=63 #NEW für neue augmentierung für Task300

                    for i in range(b):
                        print(i)
                        name = study_names[i]
                        source_img = os.path.join(source_path, name)
                        for artifact in ['blur', 'noise', 'ghosting', 'resolution', 'motion', 'spike']:
                            shutil.copy(source_img, target_path)
                            target_img = os.path.join(target_path, name)
                            search_name = 'Train_'+name[:-7]+'_'+artifact
                            try:
                                intensity = int(labels[search_name]*5)
                            except:
                                print(search_name)
                                continue
                
                            name_replaced = name.replace("_", "-")

                            new_name = os.path.join(target_path, set+'-'+name_replaced[:-7]+'_'+artifact+str(intensity)+'.nii.gz')
                            os.rename(target_img, new_name)
                        i+=1
            


    if traintest == 'test':
        use = test
        #use = ['g']
        for set in use:
            source_path = os.path.join('/local/scratch/Racoon_Workflows/Thorax_QA/data_', datasets[set], 'imagesTr')
            if set =='a':
                source_path = '/local/scratch/Racoon_tn/Thorax_QA/data/Task06_Lung/Task06_Lung/imagesTr'
            target_path = os.path.join('/local/scratch/Racoon_tn/Thorax_QA/JIP/data_dirs/temp/test_dirs/input')
            study_names = [x for x in os.listdir(source_path) if '._' not in x and '.nii.gz' in x]
            
            if set == 'h':
                labels = get_Task542_labels()
                b = 10
                #if set == 'b': b = 16
                for i in range(b):
                    print(i)
                    name = study_names[i]
                    source_img = os.path.join(source_path, name)
                    for artifact in ['blur', 'noise', 'ghosting', 'resolution', 'motion', 'spike']:
                        shutil.copy(source_img, target_path)
                        target_img = os.path.join(target_path, name)
                        search_name = 'Frankfurt_test_'+name[:-12]+'_'+artifact
                        try:
                            intensity = int(labels[search_name]*5)
                        except:
                            print(search_name)
                            continue

                        new_name = os.path.join(target_path, set+'-'+name[:-12]+'_'+artifact+str(intensity)+'.nii.gz')
                        os.rename(target_img, new_name)
                    i+=1
                    
            if set == 'g':
                labels = get_Task541_labels()
                b = 40
                #if set == 'b': b = 16
                for i in range(b):
                    print(i)
                    name = study_names[i]
                    source_img = os.path.join(source_path, name)
                    for artifact in ['blur', 'noise', 'ghosting', 'resolution', 'motion', 'spike']:
                        shutil.copy(source_img, target_path)
                        target_img = os.path.join(target_path, name)
                        search_name = 'Frankfurt_train_'+name[:-12]+'_'+artifact
                        try:
                            intensity = int(labels[search_name]*5)
                        except:
                            print(search_name)
                            continue

                        new_name = os.path.join(target_path, set+'-'+name[:-12]+'_'+artifact+str(intensity)+'.nii.gz')
                        os.rename(target_img, new_name)
                    i+=1


def inspect():
    source_path = os.path.join('/local/scratch/Racoon_Workflows/Thorax_QA/data_/Task300_Challenge/imagesTr')
    filenames = [x for x in os.listdir(source_path)]
    for filename in filenames:
        img = sitk.ReadImage(os.path.join(source_path, filename))
        img_array = sitk.GetArrayFromImage(img)



merch()

