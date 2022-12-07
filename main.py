import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging

from config import get_config
from data import load_dataset
from model import AlexNet, LeNet, GoogleNet, VGG16, ResNet50, EfficientNet

INDEX = 0


class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # Operate the method
        if args.model_name == 'AlexNet':
            self.Mymodel = AlexNet()
        elif args.model_name == 'LeNet':
            self.Mymodel = LeNet()
        elif args.model_name == 'GoogleNet':
            self.Mymodel = GoogleNet()
        elif args.model_name == 'VGG16':
            self.Mymodel = VGG16()
        elif args.model_name == 'ResNet50':
            self.Mymodel = ResNet50()
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        self.args.index += 1
        train_loss, n_correct, n_train = 0, 0, 0
        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # You can check the predicts for the last epoch
            # if (self.args.index > 49):
            #     print(torch.argmax(predicts, dim=1))

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        # Turn on the eval mode
        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    def run(self):
        # Print the parameters of model
        for name, layer in self.Mymodel.named_parameters(recurse=True):
            print(name, layer.shape, sep=" ")

        train_dataloader, test_dataloader = load_dataset(self)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        index = 0
        l_acc, l_trloss, l_epo = [], [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                index = epoch
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        self.logger.info(
            'best loss: {:.4f}, best acc: {:.2f}, best index: {:d}'.format(best_loss, best_acc * 100, index))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        # Draw the training process
        plt.figure(1)
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('./result/' + self.args.model_name + 'acc.png')

        plt.figure(2)
        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.savefig('./result/' + self.args.model_name + 'trloss.png')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Niubility(args, logger)
    nb.run()
