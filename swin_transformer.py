from data import download_food101,create_food25,create_dataloader,save_result
import torchvision.transforms as transforms
import os
from torchvision import models
import torch.nn as nn
from torch.optim import Adam
import engine
import torch


if os.path.isdir('data/food-101'):
    print("food101 dataset downloaded.")
else:
    food_101_train_path,food_101_val_path,food_101_test_path=download_food101(root='data')
#create food25
if os.path.isdir('data/food-25'):
    print("food-25 dataset exists")
    train_path,val_path,test_path='data/food-25/train','data/food-25/val','data/food-25/test'
else:
    train_path,val_path,test_path=create_food25(root='data')

#define train and test transformer
train_transform=transforms.Compose([
    transforms.Resize((246, 246),interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224,)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform=transforms.Compose([
    transforms.Resize((224, 224),interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataloader,val_dataloader,test_dataloader,class_names=create_dataloader(train_path=train_path,
                                                                              val_path=val_path,
                                                                              test_path=test_path,
                                                                              train_transform=train_transform,
                                                                              test_transform=test_transform,
                                                                              batch_size=32,
                                                                              num_workers=os.cpu_count())

#define the model
model=models.swin_v2_b(weights='DEFAULT')
model.head=nn.Linear(in_features=1024,out_features=25,bias=True)

loss_fn=nn.CrossEntropyLoss()
optimizer=Adam(params=model.parameters(),lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device)
#path to save model and its results
save_path='swin_b_food25_v1'
results=engine.train(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=val_dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     epochs=60,
                     device=device,
                     model_name="swin_b_food25_v1.pth",
                     model_saving_dir=save_path)

test_results=engine.eval_model(model=model,
                               dataloader=test_dataloader,
                               loss_fn=loss_fn,
                               num_classes=len(class_names),
                               device=device)


save_result(data_dict=results,
            result_save_dir=save_path,
            result_name="swin_b_food25_v1_train_results")
save_result(data_dict=test_results,
            result_save_dir=save_path,
            result_name="swin_b_food25_v1_eval_results")