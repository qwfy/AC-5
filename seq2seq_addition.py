"""
The task is to train a simple seq2seq model that can do addition.

For example, given a *string* input "1+26",
the model should output a *string* "27".
"""

# %%
import logging
import time
import itertools
import os

import numpy as np
import tensorflow as tf

# %%
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(tf.VERSION)


# %%
# Named tuple will be used to hold tf variable names and hyper parameters,
# so that no strings are passed around.
def make_namedtuple(name, field_names, field_values):
  import collections
  t = collections.namedtuple(name, field_names)
  for fname, fvalue in zip(field_names, field_values):
    setattr(t, fname, fvalue)
  return t


# %%
handle_fields = [
  'input_ids',
  'input_lengths',

  'decoder_input_ids',
  'decoder_input_lengths',
  'max_decode_iterations',

  'target_ids',
  'target_lengths',

  'train_loss',
  'optimize',
  'infer_logits',

  'batch_size',
  'train_loss_summary',
  'val_loss_summary']

# Various tf variable names.
Handles = make_namedtuple('Handles', handle_fields, handle_fields)


# %%
# An encoder is just an RNN.
# We feed it the input sequence, harvest its final state,
# and feed that to the decoder.
def build_encoder(source_vocab_size, source_embedding_dim,
                  rnn_size, rnn_layers,
                  batch_size, handles):

  # Input to the encoder.
  # Each entry is a vocabulary id,
  # and every sequence (i.e. row) is padded to have the same length.
  # (batch_size, sequence_length)
  input_ids = tf.placeholder(tf.int32, (None, None), name=handles.input_ids)

  # Length of each sequence, without counting the padding.
  # (batch_size,)
  sequence_lengths = tf.placeholder(tf.int32, (None,), name=handles.input_lengths)

  # Embedded version of input_ids.
  # (batch_size, max_time, source_embedding_dim)
  # where max_time == tf.shape(input_ids)[1] == tf.reduce_max(sequence_lengths)
  inputs_embedded = tf.contrib.layers.embed_sequence(
    ids=input_ids,
    vocab_size=source_vocab_size,
    embed_dim=source_embedding_dim)

  def build_cell():
    # The cell advance one time step in one layer.
    cell = tf.nn.rnn_cell.LSTMCell(
      num_units=rnn_size,
      initializer=tf.random_uniform_initializer(-0.1, 0.1))
    return cell

  # Conceptially:
  #
  # multi_rnn_cell(input,  [layer1_state,     layer2_state,     ...])
  # ->            (output, [new_layer1_state, new_layer2_state, ...])
  #
  # i.e. advance one time step in each layer.
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
    [build_cell() for _ in range(rnn_layers)])

  zero_state = multi_rnn_cell.zero_state(batch_size, tf.float32)

  # Advance sequence_lengths[i] time steps for layer i, for all i.
  # It is a design choice that we only interested in the final state.
  # (rnn_layers, 2=lstm_state_tuple_size, batch_size, rnn_size)
  _outputs, final_state = tf.nn.dynamic_rnn(
    cell=multi_rnn_cell,
    inputs=inputs_embedded,
    sequence_length=sequence_lengths,
    initial_state=zero_state)

  return final_state


# %%
# The decoder is another RNN, it is a design choice that
# it has the same number of layers and size with the encoder RNN.
#
# The final state of the encoder is passed as the initial state of the decoder,
# and a start of sequence (SOS) symbol, (its embedding vector, to be precise),
# is used as the input at time step 0.
#
# The output at each time step is projected (using a dense layer) to have vocab_size logits.
# When making inference, we sample from the output at time t,
# and use it as the input at time t+1.
#
# When training, we discard the RNN output,
# and feed it the truth corresponding to each time step.
def build_decoder(encoder_state,
                  target_vocab_size, target_embedding_dim,
                  rnn_size, rnn_layers,
                  batch_size, max_decode_iterations, vocab_to_id, handles):

  # Input sequence to the decoder, this is the truth, and only used during training.
  # The first element of each sequence is always the SOS symbol.
  # (batch_size, sequence_length)
  input_ids = tf.placeholder(tf.int32, (None, None), name=handles.decoder_input_ids)

  # Length of input_ids, without counting padding.
  # (batch_size,)
  sequence_lengths = tf.placeholder(tf.int32, (None,), name=handles.decoder_input_lengths)

  input_embeddings = tf.Variable(tf.random_uniform((target_vocab_size, target_embedding_dim)))

  inputs_embedded = tf.nn.embedding_lookup(
    params=input_embeddings,
    ids=input_ids)

  def build_cell():
    cell = tf.nn.rnn_cell.LSTMCell(
      num_units=rnn_size,
      initializer=tf.random_uniform_initializer(-0.1, 0.1))
    return cell

  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
    [build_cell() for _ in range(rnn_layers)])

  # Transform the RNN output to logits, so that we can sample from it.
  projection_layer = tf.layers.Dense(
    units=target_vocab_size,
    activation=None,
    use_bias=False)

  def make_logits(helper, reuse):
    with tf.variable_scope('decoder', reuse=reuse):
      decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=multi_rnn_cell,
        helper=helper,
        initial_state=encoder_state,
        output_layer=projection_layer)

      final_outputs, _final_state, _final_sequence_length = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        maximum_iterations=max_decode_iterations,
        impute_finished=True)

      # (batch_size, max_time, vocab_size)
      return final_outputs.rnn_output

  # The Helper is used to sample from logits,
  # you can swap one Helper with another to get different sampling behaviour.
  #
  # At time t, a TrainingHelper just read data from inputs[:, t],
  # and use it as the input at t .
  train_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs=inputs_embedded,
    sequence_length=sequence_lengths)

  # Greedy, as in GreedyEmbeddingHelper, usually means that a max() is taken.
  # So at time t, output at t-1, which has the shape (vocab_size,),
  # (considering only one sample), is sampled from, and is used as the input at time t,
  # i.e. input at t = lookup_embedding(embedding, argmax(output at t-1))
  infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding=input_embeddings,
    start_tokens=tf.tile(tf.constant([vocab_to_id['<SOS>']]), [batch_size]),
    end_token=vocab_to_id['<EOS>'])

  train_logits = make_logits(train_helper, reuse=False)

  # Use the same variables used in training.
  infer_logits = make_logits(infer_helper, reuse=True)
  tf.identity(infer_logits, name=handles.infer_logits)

  return train_logits


# %%
# Vocabulary information

# +: we only handle addition
# -: addition of negative number
# <PAD>: used to pad the sequence such that they have the same length,
#        (so that they can for a proper np.ndarray or tf.Tensor)
# <SOS>, <EOS>: used to indicates start and end of the sequence
vocabs = [str(i) for i in range(0, 10)] + ['+', '-', '<PAD>', '<SOS>', '<EOS>']

vocab_to_id = {c: i for i, c in enumerate(vocabs)}
id_to_vocab = {i: c for c, i in vocab_to_id.items()}

logger.info(vocabs)
logger.info(vocab_to_id)
logger.info(id_to_vocab)


# %%
# Data preparation

def pad_right(lol, vocab_to_id):
  # Pad right with <PAD>, such that all rows have the same length.
  row_lengths = [len(row) for row in lol]
  max_length = np.max(row_lengths)
  arr = np.ndarray((len(lol), max_length))
  arr.fill(vocab_to_id['<PAD>'])
  for i, length, row in zip(range(len(lol)), row_lengths, lol):
    arr[i, :length] = row
  return arr, row_lengths


def push_sos(arr, vocab_to_id):
  # Prepend each row with <SOS>, and drop the last column,
  # such that the length is unchanged.
  soses = np.ndarray((arr.shape[0], 1))
  soses.fill(vocab_to_id['<SOS>'])
  with_sos = np.concatenate([soses, arr], axis=1)
  no_last = with_sos[:, :-1]
  return no_last

def make_batch(batch_size, lower, upper):
  # Make one batch.
  xys = np.random.randint(low=lower, high=upper, size=(batch_size, 2))
  zs = np.sum(xys, axis=1)

  source_ids = [[vocab_to_id[char] for char in f'{xy[0]}+{xy[1]}']
                for xy in list(xys)]
  target_ids = [[vocab_to_id[char] for char in f'{z}'] + [vocab_to_id['<EOS>']]
                for z in list(zs)]

  padded_source_ids, source_lengths = pad_right(source_ids, vocab_to_id)
  padded_target_ids, target_lengths = pad_right(target_ids, vocab_to_id)
  decoder_input_ids = push_sos(padded_target_ids, vocab_to_id)
  batch = (padded_source_ids, source_lengths,
           padded_target_ids, target_lengths,
           decoder_input_ids)
  return batch


# An iterator that produces feed dicts.
def get_feed(all_batches, graph, handles):
  for (padded_source_ids, source_lengths,
       padded_target_ids, target_lengths,
       decoder_input_ids
       ) in all_batches:
    yield {graph.get_tensor_by_name(f'{handles.batch_size}:0'): len(source_lengths),
           graph.get_tensor_by_name(f'{handles.input_ids}:0'): padded_source_ids,
           graph.get_tensor_by_name(f'{handles.input_lengths}:0'): source_lengths,
           graph.get_tensor_by_name(f'{handles.decoder_input_ids}:0'): decoder_input_ids,
           graph.get_tensor_by_name(f'{handles.decoder_input_lengths}:0'): target_lengths,
           graph.get_tensor_by_name(f'{handles.target_ids}:0'): padded_target_ids,
           graph.get_tensor_by_name(f'{handles.target_lengths}:0'): target_lengths,
           graph.get_tensor_by_name(f'{handles.max_decode_iterations}:0'): 2 * np.max(source_lengths)}


# %% Hyper parameters and alikes

Hparams = make_namedtuple('Hparams', *zip(*[
  # rnn state size and number of layers
  # since we are passing the final encoder state as the initial decoder state,
  # this is shared by both the encoder and the decoder
  # another thought is to use a dense layer to bridge the encoder state and decoder state
  ('rnn_size', 50),
  ('rnn_layers', 3),

  ('source_vocab_size', len(vocabs)),
  ('target_vocab_size', len(vocabs)),

  ('source_embedding_dim', 10),
  ('target_embedding_dim', 10),

  # ('batch_size', 3),
  # ('num_epochs', 2),
  # ('num_train_batches', 4),
  # ('num_val_batches', 4),

  # NB, you don't want to batch_size * num_train_batches close to (data_upper - data_lower),
  # otherwise the model may memorise the universe and we will have no test data.
  ('batch_size', 128),
  ('num_epochs', 480),
  ('num_train_batches', 100),
  ('num_val_batches', 10),

  ('lr', 0.001),
  ('data_lower', -100),
  ('data_upper', 100),

  ('checkpoint_prefix', 'checkpoints/seq2seq_addition/model'),
  ('tensorboard_dir', 'tensorboard')
]))
os.makedirs(os.path.dirname(Hparams.checkpoint_prefix), exist_ok=True)

# %%
train_batches = [
  make_batch(Hparams.batch_size, Hparams.data_lower, Hparams.data_upper)
  for _ in range(Hparams.num_train_batches)]

val_batches = [
  make_batch(Hparams.batch_size, Hparams.data_lower, Hparams.data_upper)
  for _ in range(Hparams.num_val_batches)]


# %%
# Create the graph.
def make_graph(handles, hparams):
  graph = tf.Graph()
  with graph.as_default():
    batch_size = tf.placeholder(tf.int32, (), name=handles.batch_size)

    encoder_final_state = build_encoder(
      source_vocab_size=hparams.source_vocab_size,
      source_embedding_dim=hparams.source_embedding_dim,
      rnn_size=hparams.rnn_size,
      rnn_layers=hparams.rnn_layers,
      batch_size=batch_size,
      handles=handles)

    max_decode_iterations = tf.placeholder(tf.int32, (), name=handles.max_decode_iterations)

    train_logits = build_decoder(
      encoder_state=encoder_final_state,
      target_vocab_size=hparams.target_vocab_size,
      target_embedding_dim=hparams.target_embedding_dim,
      rnn_size=hparams.rnn_size,
      rnn_layers=hparams.rnn_layers,
      batch_size=batch_size,
      max_decode_iterations=max_decode_iterations,
      vocab_to_id=vocab_to_id,
      handles=handles)

    # Labels. So it has EOS tokens in it.
    # Used only during training.
    # (batch_size, target_sequence_length)
    target_ids = tf.placeholder(tf.int32, (None, None), name=handles.target_ids)

    # Length of target_ids, without counting padding.
    # (batch_size,)
    target_lengths = tf.placeholder(tf.int32, (None,), name=handles.target_lengths)

    # Since out target_ids is effectively of variant length,
    # we mask out those padding positions.
    loss_mask = tf.sequence_mask(lengths=target_lengths, dtype=tf.float32)
    train_loss_ = tf.contrib.seq2seq.sequence_loss(
      logits=train_logits,
      targets=target_ids,
      weights=loss_mask)
    train_loss = tf.identity(train_loss_, name=handles.train_loss)

    tf.summary.scalar(handles.train_loss_summary, train_loss)
    tf.summary.scalar(handles.val_loss_summary, train_loss)

    optimizer = tf.train.AdamOptimizer(hparams.lr)
    gradients = optimizer.compute_gradients(train_loss)
    clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                         for grad, var in gradients if grad is not None]
    optimizer.apply_gradients(clipped_gradients, name=handles.optimize)


  return graph

# %%
def restore_model(sess, checkpoint_prefix):
  loader = tf.train.import_meta_graph(checkpoint_prefix + '.meta')
  loader.restore(sess, checkpoint_prefix)


def save_model(sess, checkpoint_prefix):
  saver = tf.train.Saver()
  saver.save(sess, checkpoint_prefix)
  logger.info(f'model saved to {checkpoint_prefix}')


# %%
def train(sess: tf.Session, handles: Handles, hparams: Hparams) -> None:
  with tf.summary.FileWriter(
    logdir=hparams.tensorboard_dir,
    graph=sess.graph
  ) as summary_writer:

    train_loss = sess.graph.get_tensor_by_name(f'{handles.train_loss}:0')
    optimize = sess.graph.get_operation_by_name(handles.optimize)

    train_loss_summary = sess.graph.get_tensor_by_name(f'{handles.train_loss_summary}:0')
    val_loss_summary = sess.graph.get_tensor_by_name(f'{handles.val_loss_summary}:0')

    global_step = 0
    for i_epoch in range(1, hparams.num_epochs + 1):
      time_begin = time.monotonic()
      train_loss_vals = []
      for feed in get_feed(all_batches=train_batches, graph=sess.graph, handles=handles):
        global_step += 1
        train_loss_val, _, summary_val = sess.run([train_loss, optimize, train_loss_summary], feed)
        summary_writer.add_summary(summary_val, global_step=global_step)
        train_loss_vals.append(train_loss_val)

      val_loss_vals = []
      for feed in get_feed(all_batches=val_batches, graph=sess.graph, handles=handles):
        val_loss_val, summary_val = sess.run([train_loss, val_loss_summary], feed)
        summary_writer.add_summary(summary_val, global_step=global_step)
        val_loss_vals.append(val_loss_val)

      train_loss_val = np.mean(train_loss_vals[-len(val_loss_vals):])
      val_loss_val = np.mean(val_loss_vals)

      time_end = time.monotonic()
      logger.info(' '.join([
        f'epoch={i_epoch:0{len(str(hparams.num_epochs))}d}/{hparams.num_epochs}',
        f'train_loss={train_loss_val:.4f}',
        f'val_loss={val_loss_val:.4f}',
        f'duration={time_end-time_begin:.4f}s']))

  save_model(sess, hparams.checkpoint_prefix)


# %%
# Train the model from scratch.
def cold_train(handles: Handles, hparams: Hparams) -> None:
  with tf.Session(graph=make_graph(handles, hparams)) as sess:
    sess.run(tf.global_variables_initializer())
    save_model(sess, Hparams.checkpoint_prefix)
    train(sess, handles, hparams)


cold_train(Handles, Hparams)

# ...
# INFO:__main__:epoch=358/360 train_loss=0.0905 val_loss=0.1683 duratoin=4.9432s
# INFO:__main__:epoch=359/360 train_loss=0.1180 val_loss=0.1358 duratoin=4.8489s
# INFO:__main__:epoch=360/360 train_loss=0.1868 val_loss=0.1850 duratoin=4.6427s
# INFO:__main__:model trained and saved to ckpts/ckpt


# %%
# If the loss is still too big and decreasing,
# we can load the trained model and continue training.
def warm_train(handles: Handles, hparams: Hparams) -> None:
  with tf.Session(graph=tf.Graph()) as sess:
    restore_model(sess, hparams.checkpoint_prefix)
    train(sess, handles, hparams)


warmHparams = Hparams
warmHparams.num_epochs = 120
warm_train(Handles, warmHparams)

# ...
# INFO:__main__:epoch=118/120 train_loss=0.0716 val_loss=0.1247 duratoin=5.0092s
# INFO:__main__:epoch=119/120 train_loss=0.0484 val_loss=0.0832 duratoin=4.9530s
# INFO:__main__:epoch=120/120 train_loss=0.0448 val_loss=0.0738 duratoin=4.9770s
# INFO:__main__:model trained and saved to ckpts/ckpt


# %%
# See how we are doing.

def translate(input_ids_var_length, handles, hparams):
  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    restore_model(sess, hparams.checkpoint_prefix)

    input_ids, input_lengths = pad_right(input_ids_var_length, vocab_to_id)
    feed = {graph.get_tensor_by_name(f'{handles.batch_size}:0'): len(input_ids_var_length),
            graph.get_tensor_by_name(f'{handles.input_ids}:0'): input_ids,
            graph.get_tensor_by_name(f'{handles.input_lengths}:0'): input_lengths,
            graph.get_tensor_by_name(f'{handles.max_decode_iterations}:0'): 2 * np.max(input_lengths)}

    infer_logits = graph.get_tensor_by_name('infer_logits:0')
    logits_val = sess.run(infer_logits, feed)
    target_ids = np.argmax(logits_val, axis=2)

    return target_ids


def lookup(sequences, d):
  return [[d[k] for k in seq] for seq in sequences]


def ids_to_int(ids, id_to_vocab):
  num_str = ''.join([id_to_vocab[x] for x in ids])
  return int(num_str)

# All seen training and validating examples.
def all_seen(batches, vocab_to_id, id_to_vocab):
  seen = []
  for batch in batches:
    seqs = batch[0]
    for seq in seqs:
      filling_left = True
      left, right = [], []
      for id_ in seq:
        if filling_left:
          if id_ == vocab_to_id['+']:
            filling_left = False
          else:
            left.append(id_)
        else:
          if id_ == vocab_to_id['<PAD>']:
            break
          else:
            right.append(id_)
      seen.append((ids_to_int(left, id_to_vocab), ids_to_int(right, id_to_vocab)))
  return seen


def make_unseen(hparams, seen, n):
  made = 0
  res = []
  while made < n:
    while True:
      x, y = np.random.randint(hparams.data_lower, hparams.data_upper, (2,))
      if (x, y) not in seen:
        res.append((x, y))
        made += 1
        break
  return res


seen = all_seen(train_batches + val_batches, vocab_to_id, id_to_vocab)
xys = make_unseen(Hparams, seen, 10) + [
  (1, 1), (12, 12), (23, 32), (0, 1), (0, 0), (50, 50), (98, 98), (99, 99),
  (100, 99), (101, 102), (123, 54), (142, 173), (256, 254)]

batch = [f'{x}+{y}' for x, y in xys]
predict_ids = translate(lookup(batch, vocab_to_id), Handles, Hparams)
for expr, target_ids, (x, y) in zip(batch, predict_ids, xys):
  target_ids_chopped = itertools.takewhile(lambda x: x != vocab_to_id['<EOS>'], target_ids)
  result = ''.join(lookup([target_ids_chopped], id_to_vocab)[0])
  marker = '✓' if str(x + y) == result else '✗'
  print(f'{marker} {expr} = {"<empty>" if result=="" else result}')

# As can be seen from the following output,
# the model is doing OK on similar data that it is trained on, (13 / 17 is correct),
# and got all dissimilar data wrong.

# ...
# INFO:tensorflow:Restoring parameters from ckpts/ckpt
# ✓ 4+29 = 33
# ✓ 71+72 = 143
# ✓ -24+25 = 1
# ✗ 0+77 = 78
# ✓ 59+-27 = 32
# ✓ 70+33 = 103
# ✓ -73+61 = -12
# ✓ -79+-73 = -152
# ✓ 16+34 = 50
# ✓ -41+4 = -37
# ✓ 1+1 = 2
# ✓ 12+12 = 24
# ✓ 23+32 = 55
# ✓ 0+1 = 1
# ✗ 0+0 = 3
# ✓ 50+50 = 100
# ✗ 98+98 = 1949
# ✗ 99+99 = 1959
# ✗ 100+99 = 76
# ✗ 101+102 = -77
# ✗ 123+54 = 48
# ✗ 142+173 = -34
# ✗ 256+254 = 30