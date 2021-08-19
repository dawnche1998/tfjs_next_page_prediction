from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text
import json
import numpy as np
import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory
from tfx.dsl.io import fileio

from tfx import v1 as tfx  # pylint: disable=g-bad-import-order

from tfx_bsl.public import tfxio
file_name = 'trainData.json'
output_name = 'trainDataOutput.json'
with open(file_name) as f:
    session = json.load(f)



def ga_session_to_tensorflow_examples(session):
  """Converts a Google Analytics Session to Tensorflow Examples."""
  examples = []
  for i in range(len(session) - 1):
    features = {
        # Add any additional desired training features here.
        'cur_page': [session[i]['page']['pagePath']],
        'label': [session[i + 1]['page']['pagePath']],
        'session_index': [i],
    }
    
    # examples.append(create_tensorflow_example(features))
    examples.append(create_tensorflow_example(features))
  return examples

def create_tensorflow_example(features):
  """Populate a Tensorflow Example with the given features."""
  result = tf.train.Example()
  for name, value in features.items():
    if not value:
      raise ValueError('each feature must have a populated value list.')
    if isinstance(value[0], int):
      result.features.feature[name].int64_list.value.extend(value)
    elif isinstance(value[0], float):
      result.features.feature[name].float_list.value.extend(value)
    else:
      result.features.feature[name].bytes_list.value.extend(
          [bytes(v, 'utf-8') for v in value])
  return result



result = ga_session_to_tensorflow_examples(session)

with open(output_name, mode='a') as ff:
    json.dump(result, ff)
