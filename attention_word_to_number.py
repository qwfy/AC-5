import logging
import time
import itertools
import os
import string

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib

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
# Vocabulary information

def make_lookup(vocabs):

  vocab_to_id = {c: i for i, c in enumerate(vocabs)}
  id_to_vocab = {i: c for c, i in vocab_to_id.items()}
  lookup = make_namedtuple(
    'VocabLookup',
    field_names=['vocab_to_id', 'id_to_vocab'],
    field_values=[vocab_to_id, id_to_vocab])
  return lookup

support_chars = ['<PAD>', '<SOS>', '<EOS>']

# 'three thousand, two hundred and seven' -> '3207'
source_vocabs = [x for x in string.ascii_lowercase] + [' ', ','] + support_chars
target_vocabs = [x for x in string.digits] + support_chars

source_lookup = make_lookup(source_vocabs)
target_lookup = make_lookup(target_vocabs)


# %%
def int_to_word(n):
  # Convert integer to its English form, where 0 <= n < 1e9
  # e.g. int_to_word(27) == 'twenty seven'
  lookup = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
    15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
    19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty',
    50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'}

  def f(rest, s):
    if rest <= 19:
      return s + [lookup[rest]]
    elif 19 < rest <= 99:
      if rest % 10 == 0:
        return s + [lookup[rest]]
      else:
        ty = lookup[(rest // 10) * 10]
        finger = lookup[rest % 10]
        return s + [ty, finger]
    elif 99 < rest <= 999:
      q = rest // 100
      r = rest % 100
      if r == 0:
        return s + [lookup[q], 'hundred']
      else:
        return s + [lookup[q], 'hundred', 'and'] + f(r, [])
    else:
      if rest >= int(1e9):
        raise ValueError(n)
      elif rest >= int(1e6):
        q = rest // int(1e6)
        r = rest % int(1e6)
        if r == 0:
          return s + f(q, []) + ['million']
        else:
          return s + f(q, []) + ['million,'] + f(r, [])
      elif int(1e3) <= rest < int(1e6):
        q = rest // int(1e3)
        r = rest % int(1e3)
        if r == 0:
          return s + f(q, []) + ['thousand']
        else:
          return s + f(q, []) + ['thousand,'] + f(r, [])
      else:
        raise ValueError(n)

  return ' '.join(f(n, []))


test_ints = [
  0, 1, 7, 17, 20, 30, 45, 99,
  100, 101, 123, 999,
  1000, 1001, 1234, 9999,
  123456, 234578, 999999,
  1000000, 1000001, 9123456,
  999123456, 123000789]

# %%
for n in test_ints:
  print(f'{n:>11,} {int_to_word(n)}')


# %%
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

def make_batch(batch_size, lower, upper, source_lookup, target_lookup):
  # Make one batch.
  target_numbers = np.random.randint(low=lower, high=upper, size=(batch_size,))
  target_strings = [str(n) for n in target_numbers]
  source_strings = [int_to_word(n) for n in target_numbers]

  source_ids = [[source_lookup.vocab_to_id[vocab] for vocab in seq]
                for seq in source_strings]
  target_ids = [[target_lookup.vocab_to_id[vocab] for vocab in seq] + [target_lookup.vocab_to_id['<EOS>']]
                for seq in target_strings]

  padded_source_ids, source_lengths = pad_right(source_ids, source_lookup.vocab_to_id)
  padded_target_ids, target_lengths = pad_right(target_ids, target_lookup.vocab_to_id)
  decoder_input_ids = push_sos(padded_target_ids, target_lookup.vocab_to_id)
  batch = (padded_source_ids, source_lengths,
           padded_target_ids, target_lengths,
           decoder_input_ids)
  return batch, set(target_numbers)

def make_batches(hparams, num_batches, source_lookup, target_lookup):
  batches, seens = [], set()
  for _ in range(num_batches):
    batch, seen = make_batch(
      hparams.batch_size, hparams.data_lower, hparams.data_upper,
      source_lookup=source_lookup,
      target_lookup=target_lookup)
    batches.append(batch)
    seens = seens | seen
  return batches, seens


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

  'train_alignments',
  'train_alignments_summary',
  'attention_cell',

  'batch_size',
  'train_loss_summary',
  'val_loss_summary']

# Various tf variable names.
Handles = make_namedtuple('Handles', handle_fields, handle_fields)


# %%
# An encoder is just an RNN, we feed it the input sequence,
# use its (slightly transformed) final state as the initial state of the decoder,
# and use its outputs when doing attention.

def build_encoder(source_vocab_size, source_embedding_dim,
                  rnn_size, rnn_layers,
                  batch_size, handles):

  # Input to the encoder.
  # Each entry is a vocabulary id,
  # and every sequence (i.e. row) is padded to have the same length.
  # (batch_size, sequence_length)
  input_ids = tf.placeholder(tf.int32, (None, None), name=handles.input_ids)

  # Length of each sequence, without counting the padding,
  # This is also the length of the outputs.
  # (batch_size,)
  sequence_lengths = tf.placeholder(tf.int32, (None,), name=handles.input_lengths)

  # Embedded version of input_ids.
  # (batch_size, max_time, source_embedding_dim)
  # where max_time == tf.shape(input_ids)[1] == tf.reduce_max(sequence_lengths)
  inputs_embedded = contrib.layers.embed_sequence(
    ids=input_ids,
    vocab_size=source_vocab_size,
    embed_dim=source_embedding_dim)

  def build_cell():
    # The cell advance one time step in one layer.
    cell = tf.nn.rnn_cell.LSTMCell(
      num_units=rnn_size,
      initializer=tf.random_uniform_initializer(-0.1, 0.1))
    return cell

  # Conceptually:
  #
  # multi_rnn_cell(input,  [layer1_state,     layer2_state,     ...])
  # ->            (output, [new_layer1_state, new_layer2_state, ...])
  #
  # i.e. advance one time step in each layer.
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
    [build_cell() for _ in range(rnn_layers)])

  zero_state = multi_rnn_cell.zero_state(batch_size, tf.float32)

  # Advance sequence_lengths[i] time steps for layer i, for all i.
  # outputs :: [batch_size, max_time, rnn_size]
  # final_state :: (rnn_layers, 2=lstm_state_tuple_size, batch_size, rnn_size)
  outputs, final_state = tf.nn.dynamic_rnn(
    cell=multi_rnn_cell,
    inputs=inputs_embedded,
    sequence_length=sequence_lengths,
    initial_state=zero_state)

  return sequence_lengths, (outputs, final_state)


# %%
def wrap_with_attention(cell,
                        num_units_memory, attention_size,
                        encoder_outputs, encoder_output_lengths,
                        attention_cell_name):
  """Wrap cell the LuongAttention"""

  # If you simplify enough, an AttentionMechanism is a function with the signature:
  # memory -> query -> alignments
  #
  # memory is usually the encoder's hidden states,
  # it is transformed with a dense layer that has num_units outputs,
  # the result of this transformation is called keys in tensorflow's source code.
  #
  # query is usually the decoder's hidden state at the current time step.
  #
  # Unless you manage memory and/or transforms the query yourself,
  # num_units must be equal to decoder's hidden state size,
  # because internally, this dot (mentioned in Step 2) is used to
  # combine the keys and the decoder hidden state.
  #
  # The score calculated using dot, is passed to probability_fn,
  # which by default is softmax, and turned into a probability distribution.
  #
  # In the default setting, when the entire encoder's outputs are used as memory,
  # LuongAttention implements the global attention with input feeding,
  # and dot product is used as the score function.
  #
  attention_mechanism = contrib.seq2seq.LuongAttention(

    num_units=num_units_memory,

    memory=encoder_outputs,

    # used to mask padding positions
    memory_sequence_length=encoder_output_lengths,

    # convert score to probabilities, default is tf.nn.softmax
    probability_fn=tf.nn.softmax,

    # if memory_sequence_length is not None,
    # then a mask is created from it, and score is transformed using this mask,
    # with true values from the original score,
    # and false value score_mask_value,
    # the default is -inf, when combined with probability_fn softmax,
    # gives padding positions near-zero probabilities.
    # we choose to use the default.
    score_mask_value=None)

  # AttentionWrapper wraps a RNNCell to get another RNNCell,
  # handling attention along the way.
  #
  # data flow:
  #
  # cell_inputs = cell_input_fn(inputs, attentional_hidden_state_at_previous_time)
  # cell_output, next_cell_state = cell(cell_inputs, cell_state)
  # alignments, _unused = attention_mechanism(cell_output, _unused)
  # context = matmul(alignments, masked(attention_mechanism.memory))
  # attention =
  #   if attention_layer
  #   then attention_layer(concat([cell_output, context], 1))
  #   else context
  # output =
  #   if output_attention
  #   then attention
  #   else cell_output
  #
  new_cell = contrib.seq2seq.AttentionWrapper(
    # the original cell to wrap
    cell=cell,

    attention_mechanism=attention_mechanism,

    # size of attentional hidden state size
    attention_layer_size=attention_size,

    # can be used to enable input feeding
    # the default is: lambda inputs, attention: array_ops.concat([inputs, attention], -1)
    # which is input feeding
    cell_input_fn=None,

    # store all alignment history for visualization purpose
    alignment_history=True,

    # output the original cell's output (False),
    # or output the attentional hidden state (True)
    output_attention=True,

    name=attention_cell_name)

  return new_cell


# %%
# The decoder is a RNN with LuongAttention.
#
# The final state of the encoder is wrapped and passed
# as the initial state of the decoder,
# and a start of sequence (SOS) symbol, (its embedding vector, to be precise),
# is used as the input at time step 0.
# (The attentional hidden state is wrapped in the AttentionWrapper's state,
# so we don't have to manually passing the attentional hidden state at time 0,
# i.e. the cell after the wrapping accept the same inputs with the cell before the wrapping)
#
# The output at each time step is projected (using a dense layer)
# to have target_vocab_size logits.
# When making inference, we sample from the output at time t,
# and use it as the input at time t+1.
#
# When training, we discard the RNN output,
# and feed it the truth corresponding to each time step.
def build_decoder(encoder_state,
                  encoder_outputs, encoder_output_lengths,
                  target_vocab_size, target_embedding_dim,
                  rnn_size, rnn_layers,
                  attention_size,
                  batch_size, max_decode_iterations, target_vocab_to_id, handles):

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

  # Notice that we wrap with attention after layering,
  # this is consistent with that described in the flow chart.
  cell_with_attention = wrap_with_attention(
    cell=multi_rnn_cell,
    num_units_memory=rnn_size,
    attention_size=attention_size,
    encoder_outputs=encoder_outputs,
    encoder_output_lengths=encoder_output_lengths,
    attention_cell_name=handles.attention_cell)

  # Transform the RNN output to logits, so that we can sample from it.
  projection_layer = tf.layers.Dense(
    units=target_vocab_size,
    activation=None,
    use_bias=False)

  def make_logits(helper, reuse):
    with tf.variable_scope('decoder', reuse=reuse):
      initial_state = cell_with_attention.zero_state(batch_size=batch_size, dtype=tf.float32)
      initial_state = initial_state.clone(cell_state=encoder_state)
      decoder = contrib.seq2seq.BasicDecoder(
        cell=cell_with_attention,
        helper=helper,
        initial_state=initial_state,
        output_layer=projection_layer)

      final_outputs, final_state, _final_sequence_length = contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        maximum_iterations=max_decode_iterations,
        impute_finished=True)

      # rnn_output :: (batch_size, max_time, vocab_size)
      return final_outputs.rnn_output, final_state

  # The Helper is used to sample from logits,
  # you can swap one Helper with another to get different sampling behaviour.
  #
  # At time t, a TrainingHelper just read data from inputs[:, t],
  # and use it as the input at t.
  train_helper = contrib.seq2seq.TrainingHelper(
    inputs=inputs_embedded,
    sequence_length=sequence_lengths)

  # Greedy, as in GreedyEmbeddingHelper, usually means that a max() is taken.
  # So at time t, output at t-1, which has the shape (vocab_size,),
  # (considering only one sample), is sampled from, and is used as the input at time t,
  # i.e. input at t = lookup_embedding(embedding, argmax(output at t-1))
  infer_helper = contrib.seq2seq.GreedyEmbeddingHelper(
    embedding=input_embeddings,
    start_tokens=tf.tile(tf.constant([target_vocab_to_id['<SOS>']]), [batch_size]),
    end_token=target_vocab_to_id['<EOS>'])

  train_logits, train_final_state = make_logits(train_helper, reuse=False)

  # For visualization
  alignments = train_final_state.alignment_history.stack()
  alignments = tf.transpose(alignments, perm=[1, 0, 2], name=handles.train_alignments)

  # Use the same variables used in training.
  infer_logits, _val_final_state = make_logits(infer_helper, reuse=True)
  tf.identity(infer_logits, name=handles.infer_logits)

  return train_logits, alignments


# %%
# Create the graph.
def build_graph(handles, hparams, target_lookup):
  graph = tf.Graph()
  with graph.as_default():
    batch_size = tf.placeholder(tf.int32, (), name=handles.batch_size)

    input_sequence_lengths, (encoder_outputs, encoder_final_state) = build_encoder(
      source_vocab_size=hparams.source_vocab_size,
      source_embedding_dim=hparams.source_embedding_dim,
      rnn_size=hparams.rnn_size,
      rnn_layers=hparams.rnn_layers,
      batch_size=batch_size,
      handles=handles)

    max_decode_iterations = tf.placeholder(tf.int32, (), name=handles.max_decode_iterations)

    train_logits, train_alignments = build_decoder(
      encoder_state=encoder_final_state,
      encoder_outputs=encoder_outputs,
      encoder_output_lengths=input_sequence_lengths,
      target_vocab_size=hparams.target_vocab_size,
      target_embedding_dim=hparams.target_embedding_dim,
      rnn_size=hparams.rnn_size,
      rnn_layers=hparams.rnn_layers,
      attention_size=hparams.attention_size,
      batch_size=batch_size,
      max_decode_iterations=max_decode_iterations,
      target_vocab_to_id=target_lookup.vocab_to_id,
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
    train_loss_ = contrib.seq2seq.sequence_loss(
      logits=train_logits,
      targets=target_ids,
      weights=loss_mask)
    train_loss = tf.identity(train_loss_, name=handles.train_loss)

    tf.summary.scalar(handles.train_loss_summary, train_loss)
    tf.summary.scalar(handles.val_loss_summary, train_loss)

    # The resulting image should have a white main diagonal,
    # because of the relation of the source and target sequence.
    # Each row in the image corresponds to one target time step,
    # thus have a width of max source time.
    tf.summary.image(
      name=handles.train_alignments_summary,
      tensor=tf.cast(tf.expand_dims(train_alignments * 255, axis=3), dtype=tf.uint8),
      max_outputs=10)

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
def train(sess, handles, hparams, train_batches, val_batches):
  run_id = time.strftime("%Z%Y-%m%d-%H%M%S", time.gmtime())
  logger.info(('run_id', run_id))

  with tf.summary.FileWriter(
    logdir=hparams.tensorboard_dir(run_id),
    graph=sess.graph
  ) as summary_writer:

    train_loss = sess.graph.get_tensor_by_name(f'{handles.train_loss}:0')
    optimize = sess.graph.get_operation_by_name(handles.optimize)

    train_loss_summary = sess.graph.get_tensor_by_name(f'{handles.train_loss_summary}:0')
    val_loss_summary = sess.graph.get_tensor_by_name(f'{handles.val_loss_summary}:0')

    train_alignments_summary = sess.graph.get_tensor_by_name(f'{handles.train_alignments_summary}:0')

    global_step = 0
    for i_epoch in range(1, hparams.num_epochs + 1):
      time_begin = time.monotonic()
      train_loss_vals = []
      for feed in get_feed(all_batches=train_batches, graph=sess.graph, handles=handles):
        global_step += 1

        (train_loss_val, _optimize_val, summary_val, train_alignments_summary_val
         ) = sess.run([
          train_loss, optimize, train_loss_summary, train_alignments_summary
          ], feed)

        summary_writer.add_summary(summary_val, global_step=global_step)
        summary_writer.add_summary(train_alignments_summary_val, global_step=global_step)
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
Hparams = make_namedtuple('Hparams', *zip(*[
  # rnn state size and number of layers
  # since we are passing the final encoder state as the initial decoder state,
  # this is shared by both the encoder and the decoder
  # another thought is to use a dense layer to bridge the encoder state and decoder state
  ('rnn_size', 30),
  ('rnn_layers', 2),

  ('attention_size', 60),

  ('source_vocab_size', len(source_vocabs)),
  ('target_vocab_size', len(target_vocabs)),

  ('source_embedding_dim', int(len(source_vocabs) / 1.5)),
  ('target_embedding_dim', int(len(target_vocabs) / 1.5)),

  # ('batch_size', 3),
  # ('num_epochs', 2),
  # ('num_train_batches', 4),
  # ('num_val_batches', 4),

  ('batch_size', 128),
  ('num_epochs', 30),
  ('num_train_batches', 100),
  ('num_val_batches', 10),

  ('lr', 0.001),
  ('data_lower', 0),
  ('data_upper', 20 * 128 * (100 + 10)),

  ('checkpoint_prefix', 'checkpoints/attention_word_to_number/model'),
  ('tensorboard_dir', lambda run_id: f'tensorboard/attention_word_to_number/{run_id}')
]))
os.makedirs(os.path.dirname(Hparams.checkpoint_prefix), exist_ok=True)


# %%
train_batches, train_seen = make_batches(
  Hparams, Hparams.num_train_batches,
  source_lookup=source_lookup,
  target_lookup=target_lookup)

val_batches, val_seen = make_batches(
  Hparams, Hparams.num_val_batches,
  source_lookup=source_lookup,
  target_lookup=target_lookup)
seen = train_seen | val_seen

logger.info(f'seen_data/all_data = {len(seen)}/{Hparams.data_upper - Hparams.data_lower} = {len(seen)/Hparams.data_upper - Hparams.data_lower:.4f}')


# %%
# Train the model from scratch.
def cold_train(handles, hparams, train_batches, val_batches):
  with tf.Session(graph=build_graph(handles, hparams, target_lookup)) as sess:
    sess.run(tf.global_variables_initializer())
    save_model(sess, Hparams.checkpoint_prefix)
    train(sess, handles, hparams, train_batches, val_batches)


cold_train(Handles, Hparams, train_batches, val_batches)


# %%
# If the loss is still too big and decreasing,
# we can load the trained model and continue training.
def warm_train(handles, hparams, train_batches, val_batches):
  with tf.Session(graph=tf.Graph()) as sess:
    restore_model(sess, hparams.checkpoint_prefix)
    train(sess, handles, hparams, train_batches, val_batches)


warmHparams = Hparams
warmHparams.num_epochs = 5
warm_train(Handles, warmHparams, train_batches, val_batches)


# %%
# See how we are doing.

def translate(input_ids_var_length, handles, hparams, source_lookup):
  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    restore_model(sess, hparams.checkpoint_prefix)

    input_ids, input_lengths = pad_right(input_ids_var_length, source_lookup.vocab_to_id)
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

def make_unseen(hparams, seen, n):
  made = []
  max_loop = n * ((hparams.data_upper - hparams.data_lower) - len(seen))
  loop_counter = 0
  while len(made) < n:
    while True:
      loop_counter += 1
      if loop_counter > max_loop:
        raise Exception('Reached max loop')
      x, = np.random.randint(hparams.data_lower, hparams.data_upper, (1,))
      if x not in seen:
        made.append(x)
        break
  return made


def run_test(hparams, handles, test_ints, seen, source_lookup, target_lookup):
  numbers = make_unseen(hparams, seen, 20) + test_ints
  batch = [int_to_word(x) for x in numbers]
  predict_ids = translate(lookup(batch, source_lookup.vocab_to_id), handles, hparams, source_lookup)
  print('columns are: is correct, is similar data, truth, predicted, source sentence')
  for words, target_ids, x in zip(batch, predict_ids, numbers):
    target_ids_chopped = itertools.takewhile(lambda x: x != target_lookup.vocab_to_id['<EOS>'], target_ids)
    result = ''.join(lookup([target_ids_chopped], target_lookup.id_to_vocab)[0])
    marker = '✓' if str(x) == result else '✗'
    is_similar_data = 'Y' if hparams.data_lower <= x < hparams.data_upper else 'N'
    print(f'{marker} {is_similar_data} {x:>11} {"<empty>" if result=="" else result:<11} {words}')

run_test(Hparams, Handles, test_ints, seen, source_lookup, target_lookup)