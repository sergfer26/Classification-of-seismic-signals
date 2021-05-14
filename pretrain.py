import torch 
import pathlib
import pytz
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.autograd import Variable
from models import AutoEncoder
from dataset import LabeledSpectrograms1, LabeledSpectrograms3
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Subset


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


## Defining the reconstruction loss function.
def mseLoss(input, target):
	return torch.sum((input - target).pow(2)) / input.data.nelement()

        
def train(model, trainLoader, optimizer, pbar, valid=False):
    runningLoss = 0.0
    if valid:
        model.eval()
    for _, data in enumerate(trainLoader):
    
        image, labels = data
        x = Variable(image)
        x = x.float()
        x = x.to(device)
        ## Forward Pass.
        encoded, reconstructions = model(x)
	    ## Evaluating loss.
        loss = mseLoss(reconstructions, x)
        runningLoss += loss.item()
	    
        if not valid:
            ## Initialise all parameter gradients to 0.
            optimizer.zero_grad()
	
	        ## Backpropagation.
            loss.backward()
	        ## Optimisation.
            optimizer.step()
        pbar.set_postfix(loss='{:.4f}'.format(loss), acc=0)#'{:.4f}'.format(acc))
        pbar.update(x.shape[0])
    #epochLossList.append(runningLoss / len(trainLoader))
    return runningLoss

    #return epochLossList



if __name__ == '__main__':
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute


    epochs = 50
    dir_spectrograms = 'Rspectrograms/'
    in_channels = 9 # 3 :considers each spectogram as elements, 9: considers 3 spectograms as atributes of a single element

    ## Preparing the data.
    if in_channels == 3:
        DataSet = LabeledSpectrograms1(dir_spectrograms)
    else: 
        DataSet = LabeledSpectrograms3(dir_spectrograms)

    valid_size = 0.2 # parte para validar
    n = len(DataSet)
    indices = list(range(n))
    np.random.shuffle(indices) # revolvemos los indices
    split = int(np.floor(valid_size * n)) 
    train_idx, valid_idx = indices[split:], indices[:split] # seprarmos los indices

    trainDataSet = Subset(DataSet, train_idx) # tomamos un subconjunto de acuerdo a los indices
    valDataSet = Subset(DataSet, valid_idx)

    trainLoader = DataLoader(trainDataSet, shuffle=True, batch_size=16)
    valLoader = DataLoader(valDataSet, batch_size=16, shuffle=True)
    model = AutoEncoder(images=in_channels)

    if torch.cuda.is_available():
        model.encoder.cuda()
        model.decoder.cuda()
        

    optimizer = optim.Adam(model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-5, 
    betas = [0.9, 0.999])
    n_train = len(trainDataSet)
    n_val = len(valDataSet)

    trainLossList = []
    valLossList = []
    ## Training the Auto-Encoder.
    for epoch in range(epochs):
        with tqdm(total = n_train, position=0) as pbar:
            pbar.set_description(f'Epoch {epoch + 1}/'+str(epochs)+' - training')
            runningLoss = train(model, trainLoader, optimizer, pbar)
            trainLossList.append(runningLoss / n_train)

        with tqdm(total = n_val, position=0) as pbar:
            pbar.set_description(f'Epoch {epoch + 1}/'+str(epochs)+' - validation')
            runningLoss = train(model, valLoader, optimizer, pbar, valid=True)
            valLossList.append(runningLoss / n_val)


    
    plt.plot(trainLossList, label='training')
    plt.plot(valLossList, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss, in_channels : ' + str(in_channels))
    plt.legend()
    plt.show()

    path = 'pre_trained_models/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)# directory to store saved models
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    torch.save(model.encoder.state_dict(), path + "/encoder.pth")
    torch.save(model.decoder.state_dict(), path + "/decoder.pth")
