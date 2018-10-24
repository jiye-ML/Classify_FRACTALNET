'''
核心操作三个： local drop,  global drop, join

* local drop : 以drop的概率选择每一列
* global drop： 只有一列被选择，其他都为0
* join： 合并不同分支操作，这个时候需要选择local drop 还是 global drop


通过block方式，将三个操作封装成一个操作
'''
import tensorflow as tf

from tensorflow import transpose
from tensorflow.python.ops import nn
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import initializers
import tflearn


def tensor_shape(tensor):
    """Helper function to return shape of tensor."""
    return tensor.get_shape().as_list()

# 将column 掩码
def apply_mask(mask, columns):
    """Uses a boolean mask to zero out some columns.
  
    Used instead of boolean mask so that output has same
    shape as input.
  
    Args:
      mask:boolean tensor.
      columns:columns of fractal block.
    """
    tensor = tf.convert_to_tensor(columns)
    mask = tf.cast(mask, tensor.dtype)
    return transpose(tf.multiply(transpose(tensor), mask))

# global drop path.
def random_column(columns):
    """只有1列不为0

    Args:
      columns: the columns of a fractal block to be selected from.
    """
    num_columns = tensor_shape(columns)[0]
    # 只有一列不为0
    mask = tf.random_shuffle([True] + [False] * (num_columns - 1))
    return apply_mask(mask, columns) * num_columns

# local drop path.
def drop_some(columns, drop_prob=.15):
    """Zeros out columns with probability `drop_prob`.
    """
    num_columns = tensor_shape(columns)[0]
    mask = tf.random_uniform([num_columns]) > drop_prob
    # 需要通过的分支放大比例，这里因为一些分支被修剪之后，需要扩大剩下的权重
    scale = num_columns / tf.reduce_sum(tf.cast(mask, tf.float32))

    return tf.cond(tf.reduce_any(mask),
                   lambda: apply_mask(mask, columns) * scale,
                   lambda: random_column(columns))

#  Used to choose between local and global drop path.
def coin_flip(prob=.5):
    """Random boolean variable, with `prob` chance of being true.
  
    Args:
      prob:float, probability of being True.
    """
    with tf.variable_op_scope([], None, "CoinFlip"):
        coin = tf.random_uniform([1])[0] > prob
    return coin


def drop_path(columns, coin):
    with tf.variable_op_scope([columns], None, "DropPath"):
        # 选择使用local drop，还是global drop
        out = tf.cond(coin,
                      lambda: drop_some(columns),
                      lambda: random_column(columns))
    return out

# 连接操作
def join(columns, coin):
    """Takes mean of the columns, applies drop path if
       `tflearn.get_training_mode()` is True.
  
    Args:
      columns: columns of fractal block.
      is_training: boolean in tensor form. Determines whether drop path
        should be used.
      coin: boolean in tensor form. Determines whether drop path is
       local or global.
    """
    # 如果只有一列返回
    if len(columns) == 1:
        return columns[0]
    with tf.variable_op_scope(columns, None, "Join"):
        columns = tf.convert_to_tensor(columns)
        # 以一定概率选择local drop、 global drop
        columns = tf.cond(tflearn.get_training_mode(),
                          lambda: drop_path(columns, coin),
                          lambda: columns)
        # 最后输出一个分支
        out = tf.reduce_mean(columns, 0)
    return out


def fractal_template(inputs, num_columns, block_fn, block_asc, joined=True, is_training=True, reuse=False, scope=None):
    """Template for making fractal blocks.
  
    Given a function and a corresponding arg_scope `fractal_template`
    will build a truncated fractal with `num_columns` columns.
  
    Args:
      inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
      num_columns: integer, the columns in the fractal.
      block_fn: function to be called within each fractal.
      block_as: A function that returns argscope for `block_fn`.
      joined: boolean, whether the output columns should be joined.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      scope: Optional scope for `variable_scope`.
    """
    def fractal_expand(inputs, num_columns, joined):
        '''Recursive Helper Function for making fractal
        参看原论文中 Figure 1
        
        '''
        with block_asc():
            # 以local drop 或者 global drop 方式合并分支
            output = lambda cols: join(cols, coin) if joined else cols
            # 第一种情况，只是一个fc模块
            if num_columns == 1:
                return output([block_fn(inputs)])

            '''下面部分实现了原论文Figure 1中，f4(z)的部分'''

            # 左边一个卷积得到输出
            left = block_fn(inputs)
            # 右边第一个fc模块
            right = fractal_expand(inputs, num_columns - 1, joined=True)
            # 右边第二个fc模块
            right = fractal_expand(right, num_columns - 1, joined=False)
            cols = [left] + right
        # 合并两个分支
        return output(cols)

    with tf.variable_op_scope([inputs], scope, 'Fractal', reuse=reuse):
        # True or False
        coin = coin_flip()
        net = fractal_expand(inputs, num_columns, joined)

    return net

# 网络的一个block
def fractal_conv2d(inputs,
                   num_columns,
                   num_outputs,
                   kernel_size,
                   joined=True,
                   stride=1,
                   padding='SAME',
                   # rate=1,
                   activation_fn=nn.relu,
                   normalizer_fn=slim.batch_norm,
                   normalizer_params=None,
                   weights_initializer=initializers.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=None,
                   biases_regularizer=None,
                   reuse=None,
                   variables_collections=None,
                   outputs_collections=None,
                   is_training=True,
                   trainable=True,
                   scope=None):
    """Builds a fractal block with slim.conv2d.
    The fractal will have `num_columns` columns, and have
    Args:
      inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
      num_columns: integer, the columns in the fractal.
    """
    locs = locals()
    fractal_args = ['inputs', 'num_columns', 'joined', 'is_training']
    asc_fn = lambda: slim.arg_scope([slim.conv2d],
                                    **{arg: val for (arg, val) in locs.items()
                                       if arg not in fractal_args})
    return fractal_template(inputs, num_columns, slim.conv2d, asc_fn,
                            joined, is_training, reuse, scope)
