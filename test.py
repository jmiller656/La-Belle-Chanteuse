import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from six.moves import cPickle
import numpy as np

#Config Variables
RNN_SIZE = 200
LAYERS = 3
BATCH_SIZE = 1
SEQ_LEN = 1
GRAD_CLIP = 50.0
ITERATIONS = 5001
EPOCHS = 10
ALPHA = 0.002
DECAY_RATE = 0.97

with open("swift_vocab.pkl",'rb') as f:
	chars = cPickle.load(f)
VOCAB_SIZE = len(chars)
vocab = dict(zip(chars,range(len(chars))))
tensor = np.load("swift_data.npy")
#ITERATIONS = int(tensor.size / (BATCH_SIZE * SEQ_LEN))
tensor = tensor[:ITERATIONS*BATCH_SIZE*SEQ_LEN]
xdata =tensor
ydata = np.copy(tensor)
ydata[:-1] = xdata[1:]
ydata[-1] = xdata[0]
x_batches = np.split(xdata.reshape(BATCH_SIZE,-1),ITERATIONS,1)
y_batches = np.split(ydata.reshape(BATCH_SIZE,-1),ITERATIONS,1)

cell = rnn_cell.GRUCell(RNN_SIZE)
cell = rnn_cell.MultiRNNCell([cell]*LAYERS)

input_data = tf.placeholder(tf.int32, [BATCH_SIZE,SEQ_LEN],name="Input")
targets = tf.placeholder(tf.int32, [BATCH_SIZE,SEQ_LEN],name="targets")
initial_state = cell.zero_state(BATCH_SIZE,tf.float32)


with tf.variable_scope('rnnlm'):
	weights = tf.get_variable("softmax_w",[RNN_SIZE,VOCAB_SIZE])
	biases = tf.get_variable("softmax_b",[VOCAB_SIZE])
	with tf.device("/cpu:0"):
		embedding = tf.get_variable("embedding",[VOCAB_SIZE,RNN_SIZE])
		inputs = tf.split(1,SEQ_LEN,tf.nn.embedding_lookup(embedding,input_data))
		inputs = [tf.squeeze(input_,[1]) for input_ in inputs]



def loop(prev,_):
	prev = tf.matmul(prev,weights) + biases
	prev_symbol = tf.stop_gradient(tf.argmax(prev,1))
	return tf.nn.embedding_lookup(embedding,prev_symbol)

outputs,last_state = seq2seq.rnn_decoder(inputs,initial_state,cell,loop_function = loop)
output = tf.reshape(tf.concat(1,outputs),[-1,RNN_SIZE])
logits = tf.matmul(output,weights)+biases
probs = tf.nn.softmax(logits)
loss = seq2seq.sequence_loss_by_example(
	[logits],
	[tf.reshape(targets,[-1])],
	[tf.ones([BATCH_SIZE*SEQ_LEN])],
	VOCAB_SIZE
)
cost = tf.reduce_sum(loss) / BATCH_SIZE / SEQ_LEN
final_state = last_state
lr = tf.Variable(1e-3, trainable = False)
tvars = tf.trainable_variables()
grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars), GRAD_CLIP)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads,tvars))

def sample(sess, chars, vocab, num=200, prime='AYYYYY LMAOOO', sampling_type=2):
        state = cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
	    feed = {input_data: x, initial_state:state}
            [state] = sess.run([final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = ''
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {input_data: x, initial_state:state}
            [prob, state] = sess.run([probs, final_state], feed)
            p = prob[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

saver = tf.train.Saver(tf.all_variables())


with tf.Session() as sess:
	tf.initialize_all_variables().run()
	saver.restore(sess,'Models/chanteuse.ckpt')
	print sample(sess,chars,vocab,5000)

