import tensorflow as tf;
import numpy as np;
import re

# s = 'a    b             c       '
# s = re.sub('\\s+', '', s)
# print(s)

m = 'accuracy:  86.07%; precision:   0.00%; recall:   0.00%; FB05:   0.19'

r = float(re.split('\\s+', m.split(';')[-1].strip())[-1])

meb = np.random.uniform(-0.25, 0.25, (1, 5))
np.set_printoptions(precision=6)
tmp_str = str(meb[0][0:])
tmp_str = tmp_str.replace('[', '').replace(']', '')
tmp_str = re.sub('\\s+', ' ', tmp_str)

UNK_str = '[UNK] ' + tmp_str

print(UNK_str)

################
# c = np.random.random([5, 3])
# b = tf.nn.embedding_lookup(c, [6])
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(c)
#     print('################')
#     print(sess.run(b))
