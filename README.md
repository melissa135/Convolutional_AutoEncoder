# Convolutional_AutoEncoder
A convolutional auto-encoder for compressing time sequence data of stocks.    
Similar to full-connection autoencoder(https://github.com/melissa135/Denoise_AutoEncoder), but more suitable for time series data.   

## Network
```
AutoEncoder (
  (encoder): Sequential (
    (0): Conv1d(1, 5, kernel_size=(4,), stride=(4,))
    (1): Tanh ()
    (2): Conv1d(5, 10, kernel_size=(4,), stride=(4,))
    (3): Tanh ()
    (4): Conv1d(10, 5, kernel_size=(3,), stride=(3,))
    (5): Tanh ()
  )
  (decoder): Sequential (
    (0): ConvTranspose1d(5, 10, kernel_size=(3,), stride=(3,))
    (1): Tanh ()
    (2): ConvTranspose1d(10, 5, kernel_size=(4,), stride=(4,))
    (3): Tanh ()
    (4): ConvTranspose1d(5, 1, kernel_size=(4,), stride=(4,))
  )
)
```

## Result
The loss sequence on trainset and testset, shows the less loss and smoother curve comparing to full-connection autoencoder.
![](https://github.com/melissa135/Convolutional_AutoEncoder/blob/master/Figure_1.png)

The original 5-minute K line sequnce and the recovered sequence from compressed vector.
![](https://github.com/melissa135/Convolutional_AutoEncoder/blob/master/vision_10.png)
