import click
import torch
from model import MyAwesomeModel
from torch import nn 
from torch import optim
import numpy as np 
from matplotlib import pyplot as plt 

import pickle
from skimage.io import imread, imsave


from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    model.train()
    train_set, _ = mnist()

    ## Your solution here


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 101
    losses = np.zeros((epochs, 1))
    for e in range(epochs):
        print(f"epoch {e + 1} of {epochs}")
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()
            output = model(images)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        losses[e] = running_loss
        # plot
        plt.figure()
        plt.plot(np.arange(1, e + 1), losses[:e])
        plt.savefig("losses.png")
            

        if e % 5 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, "models/checkpoint" + str(e))





@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint)["model_state_dict"])
    model.eval()
    _, test_set = mnist()
    predictions = np.array([])

    for images, labels in test_set:
        with torch.no_grad():
            pred = model(images).cpu().numpy()
        pred = np.argmax(pred, axis = 1)
        predictions = np.append(predictions, pred == labels.cpu().numpy())

    
    print(f"accuracy = {np.mean(predictions)}")


def silly_function():
    print("Hello, this function needs to be deleted")

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
