from base import *

class UnlearnFineTune(UnlearnAbstract):
    # https://github.com/unlearning-challenge/starting-kit/blob/main/unlearning-CIFAR10.ipynb
    def unlearn(self, retain_set, forget_set, validation=None):
        """Unlearning by fine-tuning.

        Fine-tuning is a very simple algorithm that trains using only
        the retain set.

        Args:
          retain : torch.utils.data.DataLoader.
            Dataset loader for access to the retain set. This is the subset
            of the training set that we don't want to forget.
          forget : torch.utils.data.DataLoader.
            Dataset loader for access to the forget set. This is the subset
            of the training set that we want to forget. This method doesn't
            make use of the forget set.
          validation : torch.utils.data.DataLoader.
            Dataset loader for access to the validation set. This method doesn't
            make use of the validation set.
        Returns:
          net : updated model
        """
        epochs = 5

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.__model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.__model.train()

        for _ in range(epochs):
            for inputs, targets in retain_set:
                inputs, targets = inputs.to(self.__device), targets.to(self.__device)
                optimizer.zero_grad()
                outputs = self.__model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.__model.eval()
        return self.__model