from torchvision.datasets import CIFAR10

class CIFAR10_albumentation(CIFAR10):
    def __getitem__(self, idx):

        image, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            augmented_noise = self.transform(image=image)
            noised_image = augmented_noise['image']

        if self.target_transform is not None:
            augmented_original = self.target_transform(image=image)
            original_image = augmented_original['image']

        return noised_image, label
