import time
import pandas as pd
import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import shutil
def mkdir_clean(directory):
    try:
        shutil.rmtree(directory)
        os.makedirs(directory)
    except:
        os.makedirs(directory)

#%%
#T = pd.read_csv('series/'+'obs_h'+'.csv',header=0,index_col=0,parse_dates=True)
#
#TQ_bool = T.resample('M').count()>2
#nstation = len(TQ_bool.columns)
#TQ_bool_sum = TQ_bool.sum().sort_values(ascending=False)
#TQ_bool = TQ_bool[TQ_bool_sum.index]
#TQ = TQ_bool*np.arange(nstation)
#TQ[~TQ_bool] = np.nan
#
#
#TQ_bool_sum.head()

#%%
station = 'Rockaway_Inlet_near_Floyd_Bennett_Field_NY'
feature = 'obs_tid'
features = feature.split('_')
#station = 'Bergen_Basin_at_Jamaica_Bay_NY'
x_len = 24
y_len = 6
period = x_len+y_len

dir0 = 'tests/{:s}_{:s}_{:d}_{:d}/'.format(station,feature,x_len,y_len)
mkdir_clean(dir0)

orig_stdout = sys.stdout
f = open(dir0+'log.txt', 'w')
sys.stdout = f

print('Station:',station)
print('Feature:',feature)
#f.write('station: {:s}'.format(station))
#f.write('feature: {:s}'.format(feature))
#%% Data
T0 = pd.read_csv('series/'+'obs_h'+'.csv',header=0,index_col=0,parse_dates=True)[station]
T1 = pd.read_csv('series/'+'tid_h'+'.csv',header=0,index_col=0,parse_dates=True)[station]

T = pd.DataFrame({'obs':T0,'tid':T1})
T['sur'] = T0-T1
T.dropna(axis=0,how='any',inplace=True)

T['tdiff'] = np.zeros(len(T))
T.loc[1:,'tdiff'] = (T.index[1:]-T.index[:-1])/pd.Timedelta('1H')
T['isSample'] = T['tdiff'].rolling(period-1).sum()==(period-1)

T['i'] = np.arange(len(T)).astype(int)

#test_end = pd.Timestamp('2017-12-31 23:00:00')
#test_beg = pd.Timestamp('2017-01-01 00:00:00')-pd.Timedelta('{:d}H'.format(period))
#dev_end = pd.Timestamp('2016-12-31 23:00:00')
#dev_beg = pd.Timestamp('2016-01-01 00:00:00')-pd.Timedelta('{:d}H'.format(period))
#train_end = dev_beg-pd.Timedelta('1H')
test_end = pd.Timestamp('2018-01-01 00:00:00')+pd.Timedelta('{:d}H'.format(y_len))
test_beg = pd.Timestamp('2017-01-01 00:00:00')
dev_end = pd.Timestamp('2017-01-01 00:00:00')+pd.Timedelta('{:d}H'.format(y_len))
dev_beg = pd.Timestamp('2016-01-01 00:00:00')
train_end = dev_beg-pd.Timedelta('1H')

id_test = T[T.isSample][test_beg:test_end].i.values
id_dev = T[T.isSample][dev_beg:dev_end].i.values
id_train = T[T.isSample][:train_end].i.values

D = T[features[0]]
X_test_dt = np.array([D[v-period+1:v-y_len+1].index for v in id_test])
Y_test_dt = np.array([D[v-y_len+1:v+1].index for v in id_test])
X_test = np.array([D[v-period+1:v-y_len+1].values for v in id_test])
Y_test = np.array([D[v-y_len+1:v+1].values for v in id_test])
X_dev = np.array([D[v-period+1:v-y_len+1].values for v in id_dev])
Y_dev = np.array([D[v-y_len+1:v+1].values for v in id_dev])
X_train = np.array([D[v-period+1:v-y_len+1].values for v in id_train])
Y_train = np.array([D[v-y_len+1:v+1].values for v in id_train])

D = T[features[1]]
X_test_1 = np.array([D[v-period+1:v-y_len+1].values for v in id_test])
X_dev_1 = np.array([D[v-period+1:v-y_len+1].values for v in id_dev])
X_train_1 = np.array([D[v-period+1:v-y_len+1].values for v in id_train])

X_test = np.concatenate((X_test,X_test_1),axis=1)
X_dev = np.concatenate((X_dev,X_dev_1),axis=1)
X_train = np.concatenate((X_train,X_train_1),axis=1)

print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)
print('X_dev: ', X_dev.shape)
print('Y_dev: ', Y_dev.shape)
print('X_test: ', X_test.shape)
print('Y_test: ', Y_test.shape)
#np.savez(dir0+'/data', X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)



#%% Train

n_train = X_train.shape[0]
n_dev = X_dev.shape[0]
n_test = X_test.shape[0]
n_input = X_train.shape[1]
n_output = Y_train.shape[1]
print('n_train = ', n_train)
print('n_dev = ', n_dev)
print('n_test = ', n_test)
print('n_input = ', n_input)
print('n_output = ', n_output)

n_hidden = [20,10]
n_layer = len(n_hidden)
print('n_layer = ', n_layer)
for i in range(n_layer):
    print('n_hidder{:d} = '.format(i), n_hidden[i])

n_epoch = 1000
batch_size = 500
print('n_epoch = ', n_epoch)
print('batch_size = ', batch_size)
print()

#%% Graph
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
Y = tf.placeholder(tf.float32, shape=(None, n_output), name="Y")
training = tf.placeholder_with_default(False, shape=(), name='training')

batch_norm_momentum = 0.9

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()

    my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)

    my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init,
            activation=tf.nn.relu)

    hidden1 = my_dense_layer(X, n_hidden[0], name="hidden1")
    bn1 = tf.nn.relu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden[1], name="hidden2")
    bn2 = tf.nn.relu(my_batch_norm_layer(hidden2))
    outputs = tf.layers.dense(bn2, n_output, name="outputs", 
                              kernel_initializer=he_init, activation=None)

with tf.name_scope("loss"):
    loss = tf.losses.mean_squared_error(labels=Y,predictions=outputs)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    mse = tf.losses.mean_squared_error(labels=Y,predictions=outputs)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#%% Exercution
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

max_checks_without_progress = 20
checks_without_progress = 0
mse_dev_best = np.infty

with tf.Session() as sess:
    init.run()
    start_time = time.time()
    
    MSE_TRAIN = []
    MSE_DEV = []
    EPOCH = []

    for epoch in range(n_epoch):
        rnd_indices = np.random.permutation(n_train)
        X_train_shuffle = X_train[rnd_indices,:]
        Y_train_shuffle = Y_train[rnd_indices,:]
        for i in range(0,n_train,batch_size):
            X_batch = X_train_shuffle[i:i+batch_size]
            Y_batch = Y_train_shuffle[i:i+batch_size]
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, Y: Y_batch})
        if epoch%10==0:
            mse_train = mse.eval(feed_dict={X:X_batch, Y:Y_batch})
            mse_dev = mse.eval(feed_dict={X:X_dev, Y:Y_dev})
            MSE_TRAIN.append(mse_train)
            MSE_DEV.append(mse_dev)
            EPOCH.append(epoch)
            print(epoch, "Train mse:", mse_train, "Dev mse:", mse_dev, "Best_dev_mse:", mse_dev_best)
        
        if mse_dev < mse_dev_best:
            save_path = saver.save(sess, dir0+"final.ckpt")
            mse_dev_best = mse_dev
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break

    print("Training time: {:0.3f} sec".format(time.time() - start_time))
#    save_path = saver.save(sess, dir1+"final.ckpt")
    
with tf.Session() as sess:
    saver.restore(sess, dir0+"final.ckpt")
    mse_dev = mse.eval(feed_dict={X:X_dev, Y:Y_dev})
    mse_test = mse.eval(feed_dict={X:X_test, Y:Y_test})
    print("Best dev mse: {:.6f}".format(mse_dev))
    print("Test mse: {:.6f}".format(mse_dev))
    
#%% Random check
n_check = 5
check_indices = np.random.permutation(n_test)[:n_check]

check_X_dt = X_test_dt[check_indices]
check_Y_dt = Y_test_dt[check_indices]
check_X = X_test[check_indices]
check_X0,check_X1 = np.split(check_X,2,axis=1)
check_Y = Y_test[check_indices]

check_XY_dt = np.concatenate((check_X_dt,check_Y_dt),axis=1)
check_XY = np.concatenate((check_X0,check_Y),axis=1)

with tf.Session() as sess:
    saver.restore(sess, dir0+"final.ckpt")
    check_outputs = sess.run(outputs,feed_dict={X:check_X, Y:check_Y})
check_mse = ((check_Y-check_outputs)**2).mean(axis=1)
 
fig, axes = plt.subplots(n_check,1, figsize=(9, 15))
for i in range(n_check):
    ax = axes[i]
    ax.plot(check_XY_dt[i],check_XY[i],'k.-',label='obs')
    ax.plot(check_X_dt[i],check_X1[i],'b.-',label='tid')
    ax.plot(check_Y_dt[i],check_outputs[i],'r.-',label='prediction')
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    dt_fmt = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(dt_fmt)
    ax.xaxis.set_minor_locator(hours)
    ax.set_title('mse = {:0.4f}'.format(check_mse[i]))
    ax.legend(loc=2)
ax.set_xlabel('{:s}   {:s}   X:{:d}h   Y:{:d}h   Overall test mse = {:0.4f}m'.format(station,feature,x_len,y_len,mse_test),weight='bold')
fig.tight_layout()
fig.savefig(dir0+'check.png', format='png', dpi=300)
plt.close(fig)

#%% MSE plot
with tf.Session() as sess:
    saver.restore(sess, dir0+"final.ckpt")
    check_outputs = sess.run(outputs,feed_dict={X:X_test, Y:Y_test})
    
    writer = tf.summary.FileWriter(dir0, graph=tf.get_default_graph())
    writer.close()
mse_test_ts = ((Y_test-check_outputs)**2).mean(axis=0)

fig, ax = plt.subplots(1,1, figsize=(9, 9))
ax.plot(mse_test_ts,'k.-')
ax.set_ylabel('mse (m)')
ax.set_xlabel('timestep')
ax.set_title('{:s}\n{:s}   X:{:d}h   Y:{:d}h   Overall test mse = {:0.4f}m'.format(station,feature, x_len,y_len,mse_test),weight='bold')
fig.tight_layout()
fig.savefig(dir0+'mse.png', format='png', dpi=300)
plt.close(fig)

#%%
fig, ax = plt.subplots(1,1, figsize=(9, 9))
ax.plot(EPOCH,MSE_TRAIN,'k.:',label='train')
ax.plot(EPOCH,MSE_DEV,'k.-',label='dev')
ax.legend(loc=1)
ax.set_ylabel('mse (m)')
ax.set_xlabel('epoch')
ax.set_title('{:s}\n{:s}   X:{:d}h   Y:{:d}h   Overall test mse = {:0.4f}m'.format(station,feature, x_len,y_len,mse_test),weight='bold')
fig.tight_layout()
fig.savefig(dir0+'train_dev.png', format='png', dpi=300)
plt.close(fig)

#%% Save
np.savez(dir0+'train_dev_test',
         EPOCH=EPOCH,MSE_TRAIN=MSE_TRAIN,MSE_DEV=MSE_DEV,
         mse_test_ts=mse_test_ts,
         mse_dev=mse_dev,mse_test=mse_test)
#np.load(dir0+'train_dev_test.npz')

sys.stdout = orig_stdout
f.close()