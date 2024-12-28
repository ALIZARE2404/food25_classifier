import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import Food101
import random
from torch.utils.data import DataLoader
import os
from pathlib import Path
import shutil

def download_food101(root:str,
                     transform:transforms.Compose,
                     val_split:bool=True,
                     val_pecentage:int=10):
    """download food 101 dataset and create train val and test directory.
    

    Args:
        root : Path where dataset saved.
        transform:torchvision transforms to perform on training and testing data.
        val_split:if true it creats a val directory with val_percentage of images in each category.
        val_percentage : number of images from each category that go to validation set.

    Returns:
        a tuple of (train_data_dir,val_data_dir,test_data_dir)

    """
    
    
    food101_train = Food101(root=root, split='train', transform=transform, download=True)
    food101_test = Food101(root=root, split='test', transform=transform, download=True)
    food_classes=food101_train.classes

    if val_split:
        food_101_path='data/food-101/images'
        train_path = 'data/food-101/train'
        val_path='data/food-101/val'
        test_path = 'data/food-101/test'
        
        food_classes.sort()

        if os.path.isdir(train_path) and os.path.isdir(test_path) and os.path.isdir(val_path):
            print('train, val and test dir exist')
            return(train_path,val_path,test_path)
        else :
            # Create directories for train and test
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)



            for food_class in food_classes:
                src_class_path = os.path.join(food_101_path, food_class)

                if os.path.isdir(src_class_path):
                    # List all images in the class directory
                    images = os.listdir(src_class_path)

                    train_images = random.sample(images,750)  # First 750 for training
                    remained_images = [item for item in images if item not in train_images]   # Remaining for testing
                    val_image_number=int((val_pecentage*1000)/100)
                    val_images=remained_images[:val_image_number]
                    test_images=remained_images[val_image_number:]

                    # Create class directories in train and test folders
                    os.makedirs(os.path.join(train_path, food_class), exist_ok=True)
                    os.makedirs(os.path.join(val_path, food_class), exist_ok=True)
                    os.makedirs(os.path.join(test_path, food_class), exist_ok=True)

                    # Copy training images
                    for image_name in train_images:
                        shutil.copy(os.path.join(src_class_path, image_name),
                                    os.path.join(train_path, food_class))
                    # Copy validation images
                    for image_name in val_images:
                        shutil.copy(os.path.join(src_class_path, image_name),
                                    os.path.join(val_path, food_class))

                    # Copy testing images
                    for image_name in test_images:
                        shutil.copy(os.path.join(src_class_path, image_name),
                                    os.path.join(test_path, food_class))
                    shutil.rmtree(src_class_path)

            print("Train,val and test datasets created successfully.")

            shutil.rmtree(food_101_path)
            shutil.rmtree('data/food-101/meta')
            os.remove("data/food-101/README.txt")
            os.remove("data/food-101/license_agreement.txt")
            os.remove("data/food-101.tar.gz")

            return (train_path,val_path,test_path)
    
    else:
        food_101_path='data/food-101/images'
        train_path = 'data/food-101/train'
        test_path = 'data/food-101/test'
        
        food_classes.sort()

        if os.path.isdir(train_path) and os.path.isdir(test_path):
            print('train, val and test dir exist')
            return(train_path,test_path)
        else :
            # Create directories for train and test
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)



            for food_class in food_classes:
                src_class_path = os.path.join(food_101_path, food_class)

                if os.path.isdir(src_class_path):
                    # List all images in the class directory
                    images = os.listdir(src_class_path)
                    
                    train_images = random.sample(images,750)  # First 750 for training
                    test_images = [item for item in images if item not in train_images]   # Remaining for testing

                    # Create class directories in train and test folders
                    os.makedirs(os.path.join(train_path, food_class), exist_ok=True)
                    os.makedirs(os.path.join(test_path, food_class), exist_ok=True)

                    # Copy training images
                    for image_name in train_images:
                        shutil.copy(os.path.join(src_class_path, image_name),
                                    os.path.join(train_path, food_class))
                   
                    # Copy testing images
                    for image_name in test_images:
                        shutil.copy(os.path.join(src_class_path, image_name),
                                    os.path.join(test_path, food_class))
                    shutil.rmtree(src_class_path)

            print("Train and test datasets created successfully.")

            shutil.rmtree(food_101_path)
            shutil.rmtree('data/food-101/meta')
            os.remove("data/food-101/README.txt")
            os.remove("data/food-101/license_agreement.txt")
            os.remove("data/food-101.tar.gz")

            return (train_path,test_path)
        


def create_food25(root:str):
    
    if os.path.isdir(f'{root}/food-101'):
        # Define paths for original dataset
        original_dataset_path = 'root/food-101'
        new_dataset_path = 'root/food-25'

        # List of first 25 categories (adjust this list based on your actual category names)
        categories = sorted(os.listdir(f'{original_dataset_path}/train'))[:25]

        # Create new directory structure
        for split in ['train', 'val', 'test']:
            new_split_path = os.path.join(new_dataset_path, split)
            os.makedirs(new_split_path, exist_ok=True)  # Create split directory if it doesn't exist

            for category in categories:
                # Define source and destination paths
                src_category_path = os.path.join(original_dataset_path, split, category)
                dst_category_path = os.path.join(new_split_path, category)

                # Copy the category directory to the new location
                if os.path.exists(src_category_path):
                    shutil.copytree(src_category_path, dst_category_path)
                    print(f"Copied {src_category_path} to {dst_category_path}")
                else:
                    print(f"Category {src_category_path} does not exist.")

        print("New dataset created with first 25 categories.")
        return('root/food-25/train','root/food-25/val','root/food-25/test')
        
    else:
        print('please download food101 dataset with download_food101 function.')




    



    