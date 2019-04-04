import torch
from tensorboardX import SummaryWriter


def model_train(trainloader,net,criterion,optimizer,device,num_epochs,runs_path,weight_path):
   #tensorboard file_path
   writer = SummaryWriter(runs_path)

   #training
   for epoch in range(num_epochs):  
  
       global_i=epoch*len(trainloader)
       running_loss = 0.0
    
       for i, data in enumerate(trainloader, 0):
        
           global_i+=1
        
           # get the inputs
           images, labels = data
           images, labels = images.to(device), labels.to(device)

           # zero the parameter gradients
           optimizer.zero_grad()

           # forward + backward + optimize
           outputs = net(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()

           # print statistics
           running_loss += loss.item()
           if i % 200 == 199:    
               print('[%d, %5d] loss: %.3f' %
                     (epoch + 1, i + 1, running_loss / 200))
               running_loss = 0.0
        
           #tensorboard   (title,y,x)
           if i % 10 == 9:
               writer.add_scalar('train/train_loss', loss.item() , global_i)

   print('Finished Training')
   writer.close()

   #save weight
   torch.save(net.state_dict(),weight_path)