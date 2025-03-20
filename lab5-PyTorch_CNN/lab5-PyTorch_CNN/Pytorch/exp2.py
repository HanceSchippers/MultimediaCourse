
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # For plotting training curves

from models import resnet20

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = 'cpu'
        print("Using CPU")

    start_epoch = 0
    end_epoch = 14  # Total of 15 epochs
    initial_lr = 0.1

    # Data preprocessing
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ("airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck")

    # Build the model
    print('==> Building model..')
    model = resnet20(num_classes=10)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Optionally load a pre-trained model
    # Uncomment the following lines to load a pre-trained model
    # pretrain_model_path = 'pretrained/pretrain_model.pth'
    # if os.path.exists(pretrain_model_path):
    #     print('==> Loading pre-trained model..')
    #     checkpoint = torch.load(pretrain_model_path)
    #     model.load_state_dict(checkpoint['net'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f'Loaded model from epoch {checkpoint["epoch"]} with accuracy {checkpoint["acc"]:.2f}%')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                          momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler: decrease LR after 5 and 10 epochs
    def adjust_learning_rate(optimizer, epoch):
        if epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
            print('Learning rate adjusted to 0.01')
        elif epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
            print('Learning rate adjusted to 0.001')

    # Training function
    def train(epoch):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # Accumulate loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = train_loss / total
        accuracy = 100. * correct / total
        print('Epoch [%d] Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
              % (epoch, avg_loss, accuracy, correct, total))
        return avg_loss, accuracy

    # Testing function
    def test(epoch):
        print('==> Testing...')
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)  # Accumulate loss
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = test_loss / total
        acc = 100. * correct / total
        print('Epoch [%d] Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
              % (epoch, avg_loss, acc, correct, total))

        # Save checkpoint
        print('Saving checkpoint..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_%d_acc_%.3f.pth' % (epoch, acc))

        return avg_loss, acc

    # Initialize lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Main training loop
    for epoch in range(start_epoch, end_epoch + 1):
        print(f'\nEpoch {epoch + 1}/{end_epoch + 1}')
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Save the final model
    print('==> Saving final model..')
    final_state = {
        'net': model.state_dict(),
        'acc': test_accuracies[-1],
        'epoch': end_epoch,
    }
    torch.save(final_state, './final.pth')

    # Plot and save training curves
    epochs = range(1, end_epoch + 2)  # Epochs start at 1

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = './training_curves.png'
    plt.savefig(plot_path)
    print(f'Training curves saved to {plot_path}')
    plt.show()

    # Print test accuracy trend
    print('\nTest Accuracy Trend:')
    for epoch, acc in enumerate(test_accuracies, start=start_epoch):
        print('Epoch %d: Test Acc = %.3f%%' % (epoch, acc))

if __name__ == '__main__':
    main()


