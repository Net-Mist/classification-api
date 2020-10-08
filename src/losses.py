from src.models.fbnetv2 import ChannelMasking
from submodules.global_dl.training import metrics


def _count_parameters_conv2d(layer):
  # TODO
  raise NotImplementedError('')


def flops_loss(model):
  """loss function defined by number of flops, usefull for Differential Architecture Search

  This function is compatible both with TensorFlow and UpStride engine
  
  Args:
      model: Keras model containing some ChannelMasking layers
  
  Returns:
      float: loss
  """
  loss = 0
  for layer in model.layers:
    if "Conv2D" in str(type(layer)) and "Depthwise" not in str(type(layer)):
      flops = metrics._count_flops_conv2d(layer)
    if type(layer) == ChannelMasking:
      # flops is the number of flops of the channel just before ChannelMasking
      g = layer.g
      for i, prob in enumerate(g):
        cropped_layer_flops = flops * (layer.min + i * layer.step)/layer.max
        loss += cropped_layer_flops * prob
  return loss


def parameters_loss(model):
  loss = 0
  for layer in model.layers:
    if "Conv2D" in str(type(layer)) and "Depthwise" not in str(type(layer)):
      n_params = _count_parameters_conv2d(layer)
    if type(layer) == ChannelMasking:
      # flops is the number of flops of the channel just before ChannelMasking
      g = layer.g
      for i, prob in enumerate(g):
        cropped_layer_n_params = n_params * (layer.min + i * layer.step)/layer.max
        loss += cropped_layer_n_params * prob
  return loss
