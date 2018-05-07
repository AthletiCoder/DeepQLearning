import gym
import numpy as np
import tensorflow as tf
from gym import envs
import time
import retro

# env = retro.make(game='Airstriker-Genesis', state='Level1')
env = gym.make('SpaceInvaders-v0')



def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


num_actions = 6
gamma = 0.95
optimizer = tf.train.AdamOptimizer(learning_rate=5e-8)

# Build TF computational graph
tf.reset_default_graph()

# INPUT PREPROCESSING
in_frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
in_frame0 = tf.expand_dims(in_frame, 0)
mid_frame = tf.image.rgb_to_grayscale(in_frame0)
out_frame0 = tf.image.resize_images(mid_frame, [84, 84])
out_frame = tf.squeeze(out_frame0)
# def proc_state()


inp_t = tf.placeholder(shape = [None,84,84,4] , dtype = tf.float32)
act_t_ph = tf.placeholder(tf.int32, [None], name="action")
rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
done_t_ph = tf.placeholder(tf.float32, [None], name="done")

# Q-LEARNING
def q_func(inp,num_act):
    with tf.variable_scope("q_func", reuse=tf.AUTO_REUSE):
        conv_layer1 = tf.layers.conv2d(inputs=inp,filters=16,kernel_size=[8, 8],strides=(4,4),padding="valid",activation=tf.nn.relu)
        conv_layer2 = tf.layers.conv2d(inputs=conv_layer1,filters=32,kernel_size=[4, 4],strides=(2,2),padding="valid",activation=tf.nn.relu)
        flat_conv = tf.reshape(conv_layer2, [-1,9*9*32])
        fc = tf.layers.dense(inputs=flat_conv, units=256, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=fc, units = num_act, activation=tf.nn.relu)
        # [output] = sess.run([out],feed_dict={ inp : input })
        return out

q_t = q_func(inp_t,num_actions)
q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_func")

inp_tp1 = tf.placeholder(shape = [None,84,84,4] , dtype = tf.float32)
q_tp1 = q_func(inp_tp1,num_actions)
q_tp1_best = tf.reduce_max(q_tp1, 1)
q_tp1_best = (1.0 - done_t_ph) * q_tp1_best
q_t_selected_target = rew_t_ph + gamma * q_tp1_best
td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
errors = huber_loss(td_error)
optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)

# Train Network
init = tf.initialize_all_variables()
num_episodes = 1000
e = 0.5

with tf.Session() as sess:
    # sess.run(init)
    saver= tf.train.Saver()
    saver.restore(sess,'./model_breakout.ckpt')
    episodeno = 0
    for i in range(num_episodes):
        episodeno += 1
        print(episodeno)
        s = env.reset()
        s,_,_,info = env.step(1)
        lives = info['ale.lives']
        out_s = sess.run([out_frame],feed_dict = {in_frame:s})
        state = [out_s]
        state = np.stack([out_s]*4,axis=3)
        # print(state.shape) (1,84,84,4)
        d = 0
        iteration = 0
        while d != 1:
            __temp = np.random.rand(1)
            if __temp < e:
                action = env.action_space.sample()  # replace by e-greedy
            else:
                q_out = sess.run([q_t],feed_dict = {inp_t : state})
                action = np.argmax(q_out)

            s,r,d,inf = env.step(action)
            if(inf['ale.lives']<lives):
                r = -10
                lives = inf['ale.lives']
            if r==200:
                r=1000
            out_s = sess.run([out_frame], feed_dict={in_frame: s})
            out_s = np.asarray(out_s)
            out_s = np.expand_dims(out_s,axis=3)
            new_state = np.append(state,out_s,3)
            new_state = np.delete(new_state,0,3)

            # Train the network here
            e,_ = sess.run([errors, optimize_expr],feed_dict = {inp_t : state, inp_tp1 : new_state , act_t_ph:[action] , rew_t_ph: [r] , done_t_ph : [d]})
            state = new_state
        e = e/1.1
        if i % 10 ==1:
            saver.save(sess, './model_breakout.ckpt')

    saver.save(sess,'./model_breakout.ckpt')

    for i in range(1):
        episodeno += 1
        print(episodeno)
        s = env.reset()
        s,_,_,info = env.step(1)
        lives = info['ale.lives']
        out_s = sess.run([out_frame], feed_dict={in_frame: s})
        state = [out_s]
        state = np.stack([out_s]*4, axis=3)
        d = 0
        iteration = 0
        while d != 1:
            q_out = sess.run([q_t], feed_dict={inp_t: state})
            action = np.argmax(q_out)

            s, r, d, inf = env.step(action)
            # if(inf['ale.lives']<lives):
            #     lives = inf['ale.lives']
            #     r = -100
            out_s = sess.run([out_frame], feed_dict={in_frame: s})
            out_s = np.asarray(out_s)
            out_s = np.expand_dims(out_s, axis=3)
            new_state = np.append(state, out_s, 3)
            new_state = np.delete(new_state, 0, 3)
            env.render()
            state = new_state

