import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

 

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_set=datasets.FashionMNIST('./data',train=True,download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False,transform=custom_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    
    if training == False:
        return test_loader
    return train_loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None
    
    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10),
        )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model=model.train()
    opt = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    for epoch in range(T):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            # print statistics
            running_loss += loss.item()
        
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Train Epoch: {epoch}  Accuracy: {correct}/{total}({100 * correct / total:.2f}%) Loss: {running_loss / len(train_loader):.3f}')
       


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model = model.eval()
    opt = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    running_loss = 0.0
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        # print statistics
        running_loss += loss.item()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    if show_loss != False:
        print(f'Loss: {running_loss / len(train_loader):.3f}')
     
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """    
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    prob = F.softmax(model(test_images),dim=1)
    prob = prob[index].cpu().detach().numpy()
    descending_prob = np.copy(prob)
    descending_prob[::-1].sort()
    first_index = np.where(prob == descending_prob[0])[0][0]
    second_index = np.where(prob == descending_prob[1])[0][0]
    third_index = np.where(prob == descending_prob[2])[0][0]
    print(f'{class_names[first_index]}: {100* prob[first_index]:.2f}%')
    print(f'{class_names[second_index]}: {100* prob[second_index]:.2f}%')
    print(f'{class_names[third_index]}: {100* prob[third_index]:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''   
    train_loader  = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader,criterion,5)
    evaluate_model(model,test_loader,criterion)
    evaluate_model(model,test_loader,criterion,False)
    pred_set,_= next(iter(test_loader))
    predict_label(model,pred_set,1)
    
