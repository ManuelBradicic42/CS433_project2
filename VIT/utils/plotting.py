def plot_samples(hand, ds, tokenizer):
    hand = hand.flatten()
    for i in range(5):
        token = tokenizer.decode(ds[i][1].to("cpu").numpy())
        hand[i].imshow(ds[i][0][0,:,:], cmap="gray")
        hand[i].axis('off')  # Hide axis
        hand[i].set_title(f"Label: {token}")  # Set title for each image
    return hand