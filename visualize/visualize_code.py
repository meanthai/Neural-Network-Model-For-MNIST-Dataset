import matplotlib.pyplot as plt

def show_image(imgs, labels):
    imgs = imgs.reshape((-1, 28, 28)) # Reshaping the images as they need to be two dimensional images
    plt.figure(figsize=(10, 5))
    
    num_imgs = 10 # number of images to plot
    
    # Plot the first (num_imgs) images of the dataset with their labels
    for i in range(num_imgs):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
