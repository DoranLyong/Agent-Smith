""" (ref) https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter02.1.ipynb
"""
import random 

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import wandb 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch_lr_finder import LRFinder



class D2torchEngine(object): 
    def __init__(self, model, loss_fn, optimizer):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.model.to(self.device)

        self.loss_fn = loss_fn 
        self.optimizer = optimizer 

        # ===
        self.train_loader = None 
        self.val_loader = None 
        self.scheduler = None  # learning_rate scheduler 
        self.is_batch_lr_scheduler = False  # scheduler ON/OFF 
        self.clipping = None # gradient clipping **** (add later)
        self.wandb = None # W&B board 

        # === 
        self.train_losses = [] 
        self.val_losses = [] 
        self.learning_rates = [] 
        self.total_epochs = 0 

        self.visualization = {} 
        self.handles = {}

        # === 
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()


    def to(self, device):
        # === This method allows the user to specify a different device === # 
        
        self.device = device 
        self.model.to(self.device)    

    def set_loaders(self, train_loader, val_loader):
        # === This method allows the user to define which train_loader (and val_loader) to use ===#
        
        self.train_loader = train_loader 
        self.val_loader = val_loader

    def set_wandb(self, wandb): 
        # === This method allows the user to use a W&B instance === # 
        self.wandb = wandb


    def _make_train_step(self): 
        # === build this in higher-order function === # 
        def perform_train_step(input, label):
            self.model.train()  # set train mode 

            yhat = self.model(input) # get score out of the model 
            loss = self.loss_fn(yhat, label) # computes the loss 



            loss.backward() # computes gradients 

            if callable(self.clipping): # ****** study later ***** # 
                self.clipping()

            # Updates parameters using gradients and the learning rate 
            self.optimizer.step() 
            self.optimizer.zero_grad() 
            
            # Returns the loss 
            return loss.item() 
        return perform_train_step

    def _make_val_step(self): 
        # === build this in higher-order function === # 
        def perform_val_step(input, label): 
            self.model.eval() # set eval mode 

            yhat = self.model(input)
            loss = self.loss_fn(yhat, label)

            return loss.item()
        return perform_val_step

    def _mini_batch(self, validation=False): 
        if validation : 
            data_loader = self.val_loader
            step = self.val_step
        else: 
            data_loader = self.train_loader 
            step = self.train_step 
        
        if data_loader is None: 
            print(f"No any dataloader @ validation={validation}")
            return None 
            
        n_batches = len(data_loader)
        
        # === Run loop === # 
        mini_batch_losses = [] 
        for i, (x_batch, y_batch) in enumerate(data_loader): 
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch) # train/val-step
            mini_batch_losses.append(mini_batch_loss)

            if not validation: # only during training! 
                self._mini_batch_schedulers(i/n_batches)# call the learning rate scheduler 
                                                        # at the end of every mini-batch update 
        # Return the avg. loss 
        return np.mean(mini_batch_losses) 

    def set_seed(self, seed:int=42): 
        # === Reproducible === #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        try:
            # === sampling for imbalanced data === # 
            self.train_loader.sampler.generator.manual_seed(seed) # *** (add later)
        except AttributeError:
            pass 

    def train(self, n_epochs, seed=42):
        self.set_seed(seed) # To ensure reproducibility of the training process

        if self.wandb:
            # Tell wandb to watch what the model gets up to: gradients, weights, and more!
            self.wandb.watch(self.model, self.loss_fn, log="all", log_freq=10)
            self.wandb.define_metric("train_loss", summary="min") # (ref) https://docs.wandb.ai/guides/track/log
            self.wandb.define_metric("val_loss", summary="min")

        for epoch in tqdm(range(n_epochs)):
            self.total_epochs += 1

            # === TRAINING === # 
            train_loss = self._mini_batch(validation=False) 
            self.train_losses.append(train_loss)

            # === VALIDATION === # 
            # Set no gradients ! 
            with torch.no_grad(): 
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            self._epoch_schedulers(val_loss)    # learning_rate scheduler
                                                # make sure to set after validation 

            # === If a W&B has been set === # 
            if self.wandb: 
                # Record logs of both losses for each epoch 
                log_dict = {"epoch":epoch, "train_loss":train_loss}

                if val_loss is not None:
                    update_dict = {'val_loss': val_loss} 
                    log_dict.update(update_dict)  # dict() update 

                if self.scheduler is not None: 
                    log_dict.update({'lr_schedule': np.array(self.learning_rates[-1][-1])}) # get the last LR

                self.wandb.log(log_dict)

    def save_checkpoint(self, filename:str): 
        # Builds dictionary with all elements for resuming training
        checkpoint = {  'epoch': self.total_epochs, 
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(), 
                        'train_loss' : self.train_losses, 
                        'val_loss' : self.val_losses
                    }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename:str): 
        checkpoint = torch.load(filename) # Loads dictionary 

        # === Restore state for model and optimizer === # 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.train_losses = checkpoint['train_loss']
        self.val_losses = checkpoint['val_loss']

        # always use TRAIN for resuming training  
        self.model.train()

    def predict(self, input):
        self.model.eval() # Set is to evaluation mode for predictions
        
        input_tensor = torch.as_tensor(input).float() # Takes a Numpy input and make it a float tensor 
        yhat_tensor = self.model(input_tensor.to(self.device)) # Send input to device and uses model for prediction
        
        self.model.train() # Set it back to train mode

        return yhat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend() 
        plt.tight_layout()
        return fig 


    def _metrics(self, name:str):
        """ name can be: 
                - 'accuracy'
            Usage example: 
                - metric_dict['accuracy'] 
        """
        def top1_accuracy(predictions, labels, threshold=0.5):
            n_samples, n_dims = predictions.size() # (num_instances, num_classes)

            if n_dims > 1: 
                # === multiclass classficiation === # 
                _, argmax_idx = torch.max(predictions, dim=1) 
            else: 
                # === binary classficiation === # 
                # we NEED to check if the last layer is a sigmoid (to produce probability)
                if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
                    argmax_idx = (predictions > threshold).long()
                else: 
                    argmax_idx = (torch.sigmoid(predictions) > threshold).long()

            # How many samples got classified correctly for each class 
            result = [] 
            for cls_idx in range(n_dims):
                n_class = (labels == cls_idx).sum().item() # nun_items for each class 
                n_correct = (argmax_idx[labels == cls_idx ] == cls_idx).sum().item() # num_corrects for each class 

#                print(labels == cls_idx) # where is the label ? 
#                print(argmax_idx[labels == cls_idx]) # what is the prediction results on each label 
#                print(argmax_idx[labels == cls_idx ] == cls_idx) # return only the correct answers 

                result.append((n_correct, n_class))
            return torch.tensor(result)
            

        # === Metric Switch === # 
        metric_dict = dict( accuracy=top1_accuracy,
                            )
        return metric_dict[name]


    def correct(self, input, labels):
        self.model.eval() 
        yhat = self.model(input.to(self.device))
        labels = labels.to(self.device)

        # === 
        metric_func = self._metrics('accuracy')

        results = metric_func(yhat, labels)

        return results

    @staticmethod 
    def loader_apply(dataloader, func, reduce='sum'):
        results = [func(inputs, labels) for idx, (inputs, labels) in enumerate(dataloader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum': 
            results = results.sum(axis=0)
        elif reduce == 'mean': 
            results = results.float().mean(axis=0)
        return results     

    # ====================== # 
    #   Learninig Scheduler  #  
    # ====================== # 
    def set_optimizer(self, optimizer): 
        self.optimizer = optimizer

    def set_lr_scheduler(self, scheduler): 
        
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or 
                isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or 
                isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.is_batch_lr_scheduler = True 
            else:
                self.is_batch_lr_scheduler = False


    def _epoch_schedulers(self, val_loss):
        if self.scheduler: 
            if not self.is_batch_lr_scheduler: # if not batch_scheduler 
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step() 

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups'])) 
                self.learning_rates.append(current_lr) # log of learninig_rates 

    def _mini_batch_schedulers(self, frac_epoch): 
        if self.scheduler: 
            if self.is_batch_lr_scheduler: # if batch_scheduler 
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts): 
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else: 
                    self.scheduler.step()

                current_lr = list(map(lambda d:d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr) # log of learninig_rates 

    def lr_range_test(self, end_lr=1e-1, num_iter=100): 
        # === Learning Rate Range Test === # 
        # Using LRFinder 
        assert self.train_loader is not None, "You didn't set trainloader"
        
        fig, ax = plt.subplots(1, 1, figsize=(6,4))

        lr_finder = LRFinder(self.model, self.optimizer, self.loss_fn, device=self.device)
        lr_finder.range_test(self.train_loader, end_lr=end_lr, num_iter=num_iter)
        lr_finder.plot(ax=ax, log_lr=True)

        fig.tight_layout()
        lr_finder.reset()

        return fig 



        



        

                





