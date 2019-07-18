"""
Trainer for the model.
"""

import torch
import torch.functional as F
from torch import nn, optim


class Trainer():
    """
    Trainer to train the model.
    """

    def __init__(self, model, train_data, test_data,
                 criterion, optimizer, scheduler):
        # check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(model, optimizer, criterion, train_dataloader,
              test_dataloader, n_epochs=3, init_val_loss=np.Inf,
              print_every=50, train_device='cuda', init_epoch=1):
        """
        Function to train the model

        Args:
            model: Model used.
            optimizer: Optimizer used.
            criterion: Criterion used
            train_dataloader: Dataloader used
            test_dataloader: Dataloader used
            n_epochs: Max number of epochs.
            init_val_loss: Initial validation loss.
            print_very: print this number of batches.
            train_device: device used for training.
        """
        # Losses array
        losses = {"train": [], "valid": []}
        valid_loss_min = init_val_loss
        # Progress bar
        pbar_epochs = tqdm_notebook(
            range(init_epoch, n_epochs + init_epoch),
            total=n_epochs, ncols=800)
        for epoch in pbar_epochs:
            # keep track of training and validation loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            model.train()
            total_train = len(train_dataloader.dataset)
            pbar_train = tqdm_notebook(enumerate(train_dataloader, 1),
                                       total=total_train // BATCH_SIZE,
                                       ncols=750)
            for c, (img, target) in pbar_train:
                # Move tensors to GPU, if CUDA is available
                img, target = img.to(train_device), target.to(train_device)
                # Clear the gradients of all optimized variables
                optimizer.zero_grad()
                # Batch size
                bs = img.size(0)
                # forward pass
                pred = model(img)
                # calculate the batch loss
                # import ipdb; ipdb.set_trace() # debugging starts here
                # Fix dimension
                pred = pred.permute(0, 2, 3, 1)
                loss = criterion(pred.float(), target.float())
                # backward pass: compute gradient of the loss
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * bs
                # show partial results
                if c % print_every == 0:
                    # print training statistics
                    pbar_train.set_description(
                        'Batch: {:5d}/{:5d} Training Loss: {:2.6f}'.format(
                            c,
                            len(train_dataloader),
                            train_loss / (bs * c)
                        ))

            # Validate model
            valid_loss = validate_model(
                model, criterion, test_dataloader, train_device)
            losses["valid"].append(valid_loss)
            # calculate average losses
            train_loss = train_loss / total_train
            losses["train"].append(train_loss)
            # Save the model if validation loss has decreased
            save_model = False
            if valid_loss < valid_loss_min:
                # update the min_loss
                valid_loss_min = valid_loss
                # Saves the model
                save_model = True
            # Save results
            improve = 'better' if save_model else 'worse'
            save_checkpoint(model, optimizer, scheduler, epoch,
                            losses, "Model_" + str(epoch) + improve + ".model")

            # print training/validation statistics
            output_str = ('Epoch: {:3d}/{:3d}' +
                          ' Training Loss: {:2.6f}' +
                          ' Validation Loss: {:2.6f}' +
                          ' Saving Model: {}')
            pbar_epochs.set_description(output_str.format(
                epoch, n_epochs, train_loss, valid_loss, save_model))

            # Scheduler step
            scheduler.step(valid_loss)

        # Return losses
        return losses

    def validate(model, criterion, data_loader):
        """
        Validate the given model on test data.

        Args:
            model: Model used.
            criterion: Criterion used
            data_loader: Dataloader used
            train_device: device used for validation.

        Return:
            loss: Validation loss
        """
        ######################
        # validate the model #
        ######################
        # Initializa valid loss
        valid_loss = 0.0
        # Move to device
        model = model.to(train_device)
        total_valid = len(data_loader.dataset)
        # Evalutions mode
        model.eval()
        with torch.no_grad():
            for img, target in data_loader:
                # move tensors to GPU if CUDA is available
                img, target = img.to(train_device), target.to(train_device)
                bs = img.size(0)

                # Check cache
                pred = model(img)
                pred = pred.permute(0, 2, 3, 1)
                # calculate the batch loss
                loss = criterion(pred.float(), target.float())

                # update average validation loss
                valid_loss += loss.item() * bs
        # Valid loss
        valid_loss = valid_loss / total_valid
        return valid_loss

    def save_checkpoint(self, model, optimizer, scheduler, epoch, losses,
                        file_name, directory="model"):
        """
        Saves the current model checkpoint

        Args:
            model: Model used.
            optimizer: Optimizer used.
            Scheduler: Scheduler used.
            epoch: Epouch number.
            losses: Dict with the losses.
            file_name: name of the saved file.
            directory: directory to save models.

        """
        # Saves the model
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optim_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(),
                      "epoch": epoch,
                      "train_loss": losses["train"],
                      "valid_loss": losses["valid"]}
        # Created directory
        torch.save(checkpoint, file_name)

    def load_checkpoint(self, model, optimizer, scheduler,
                        losses, file_name, location='cpu'):
        """
        Load all info from last model.

        Args:
            model: Initialized Model.
            optimizer: Initialized Optimizer.
            Scheduler: Initialized Scheduler.
            losses: Initialized Dict with the losses.
            file_name: name of the saved file.
            location: Where to load the model.
        """
        # Loads the model
        checkpoint = torch.load(file_name, map_location=location)

        # Load in given objects
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        losses["train"] = checkpoint["train_loss"]
        losses["valid"] = checkpoint["valid_loss"]
