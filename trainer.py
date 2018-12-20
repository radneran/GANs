import torch
from utils import save_checkpoint


def train_classifier(classifier, optimizer, dataloader, loss_fn, epochs, modelname, verbose=True, softmax=False):
    iterations = 0
    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            samples, labels = [item.cuda() for item in data]
            preds = torch.softmax(classifier(samples).view(-1, 10), 1) if softmax else classifier(samples).view(-1,10)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            iterations+=1
            if verbose:
                print("[%d/%d][%d] Loss: %f"%(epoch, epochs, iterations, loss))
            if iterations % 1000 == 999:
                save_checkpoint(epoch, iterations, classifier, optimizer, modelname)
                if not verbose:
                    print("[%d/%d][%d] Loss: %f"%(epoch, epochs, iterations, loss))
    print("Finished training.")
    save_checkpoint(epochs, iterations, classifier, optimizer, modelname)
    print("Model saved.")