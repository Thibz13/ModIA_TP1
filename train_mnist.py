import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mnist_net import MNISTNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(net, optimizer, loader, epochs,writer):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)

def Test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  
  parser.add_argument('--exp_name', type=str, default='MNIST', help='experiment name')
  parser.add_argument('--bs',type = int,default = 200 , help = 'aaa')
  parser.add_argument('--lr',type = float ,default = 0.1 , help = 'bbb')
  parser.add_argument('--ep',type = int,default = 3 ,help = 'ccc')

  args = parser.parse_args()
  exp_name = args.exp_name
  epochs = args.ep
  batch_size = args.bs
  lr = args.lr

  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])

  # datasets
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  net = MNISTNet()
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(net.parameters(),lr =args.lr)

  writer = SummaryWriter(f'runs/MNIST')

  train(net,optimizer,trainloader,epochs = args.ep,writer = writer)
  test_acc = Test(net,testloader)
  print(f'Test accuracy:{test_acc}')
  torch.save(net.state_dict(), "mnist_net.pth")

  perm = torch.randperm(len(trainset.data))
  images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
  images = images.unsqueeze(1).float().to(device)

  with torch.no_grad():
    embeddings = net.get_features(images)
    writer.add_embedding(embeddings,
                  metadata=labels,
                  label_img=images, global_step=1)

  # save networks computational graph in tensorboard
  writer.add_graph(net, images)

  # save a dataset sample in tensorboard
  img_grid = torchvision.utils.make_grid(images[:64])
  writer.add_image('mnist_images', img_grid)