# Convolutional_AutoEncoder
A convolutional auto-encoder for compressing time sequence data of stocks.

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
