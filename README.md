A Deep Q-Network in Tensorflow that can play Atari 2600 games! Implement DeepMind paper [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).

## Requirements

- Python 3.5
- [Tensorflow 0.12](https://www.tensorflow.org/get_started/os_setup#anaconda_installation)
- [OpenAI gym](https://github.com/openai/gym#installing-everything)
- Scipy 0.18

## Note

Implement in "NCHW" data format so it can run faster on Tensorflow GPU version, meanwhile it can not run on Tensorflow CPU version.

When set `is_display` to `False`, some viedo clips will be record at `./gym` dir.

When set `is_train` to `False` or set `is_restore_model` to `True`, `MODEL_TIME` must be set correctly. 

run the following command to start.

```python
python main.py
```

## Summary Data

```bash
tensorboard --logdir="./summary"
```
