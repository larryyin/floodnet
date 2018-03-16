#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:12:09 2018

@author: larry
"""
import argparse
parser = argparse.ArgumentParser(description='Single station net')
parser.add_argument('--feature', metavar='str', type=str,
                    default='obs_tidext',
                   help='Feature options: "obs", "sur", "obs_tid", "obs_tidext", "sur_tidext"')
parser.add_argument('--xl', metavar='int', type=int,
                   help='x_len, look-back bars')
parser.add_argument('--yl', metavar='int', type=int,
                   help='y_len, prediction bars')
parser.add_argument('--station', metavar='str', type=str,
                   help='Station name')
parser.add_argument('--layers', metavar='int', type=int, nargs='*',
                   help='Hidden layers and hidden units')
args = parser.parse_args()

#print(args.feature)
#print(args.xl)
#print(args.yl)
#print(args.station)
#print(args.layers)

#%%
import time
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import shutil

#%%
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
station = args.station
feature = args.feature
x_len = args.xl
y_len = args.yl
n_hidden = args.layers

# Test
#station = 'Rockaway_Inlet_near_Floyd_Bennett_Field_NY'
##station = 'Bergen_Basin_at_Jamaica_Bay_NY'
#feature = 'sur_tidext'
#x_len = 48
#y_len = 12
#n_hidden = [30,10]

#
period = x_len+y_len

dir0 = 'tests/{:d}_{:d}_{:s}_{:s}/'.format(x_len,y_len,feature,station)
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

if (feature=='obs') | (feature=='sur'):
    D = T[feature]
    X_test_dt = np.array([D[v-period+1:v-y_len+1].index for v in id_test])
    Y_test_dt = np.array([D[v-y_len+1:v+1].index for v in id_test])
    X_test = np.array([D[v-period+1:v-y_len+1].values for v in id_test])
    Y_test = np.array([D[v-y_len+1:v+1].values for v in id_test])
    X_dev = np.array([D[v-period+1:v-y_len+1].values for v in id_dev])
    Y_dev = np.array([D[v-y_len+1:v+1].values for v in id_dev])
    X_train = np.array([D[v-period+1:v-y_len+1].values for v in id_train])
    Y_train = np.array([D[v-y_len+1:v+1].values for v in id_train])
elif feature=='obs_tid':
    features = feature.split('_')
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
elif feature=='obs_tidext':
    features = feature.split('_')
    # obs
    D = T[features[0]]
    X_test_dt = np.array([D[v-period+1:v-y_len+1].index for v in id_test])
    Y_test_dt = np.array([D[v-y_len+1:v+1].index for v in id_test])
    X_test = np.array([D[v-period+1:v-y_len+1].values for v in id_test])
    Y_test = np.array([D[v-y_len+1:v+1].values for v in id_test])
    X_dev = np.array([D[v-period+1:v-y_len+1].values for v in id_dev])
    Y_dev = np.array([D[v-y_len+1:v+1].values for v in id_dev])
    X_train = np.array([D[v-period+1:v-y_len+1].values for v in id_train])
    Y_train = np.array([D[v-y_len+1:v+1].values for v in id_train])
    # tid
    D = T[features[1][:3]]
    X_test_1 = np.array([D[v-period+1:v-y_len+1].values for v in id_test])
    X_dev_1 = np.array([D[v-period+1:v-y_len+1].values for v in id_dev])
    X_train_1 = np.array([D[v-period+1:v-y_len+1].values for v in id_train])
    # tidext
    X_test_2 = np.array([D[v-y_len+1:v+1].values for v in id_test])
    X_dev_2 = np.array([D[v-y_len+1:v+1].values for v in id_dev])
    X_train_2 = np.array([D[v-y_len+1:v+1].values for v in id_train])
    
    X_test = np.concatenate((X_test,X_test_1,X_test_2),axis=1)
    X_dev = np.concatenate((X_dev,X_dev_1,X_dev_2),axis=1)
    X_train = np.concatenate((X_train,X_train_1,X_train_2),axis=1)
elif feature=='sur_tidext':
    features = feature.split('_')
    D = T[features[0]]
    X_test_dt = np.array([D[v-period+1:v-y_len+1].index for v in id_test])
    Y_test_dt = np.array([D[v-y_len+1:v+1].index for v in id_test])
    X_test = np.array([D[v-period+1:v-y_len+1].values for v in id_test])
    Y_test = np.array([D[v-y_len+1:v+1].values for v in id_test])
    X_dev = np.array([D[v-period+1:v-y_len+1].values for v in id_dev])
    Y_dev = np.array([D[v-y_len+1:v+1].values for v in id_dev])
    X_train = np.array([D[v-period+1:v-y_len+1].values for v in id_train])
    Y_train = np.array([D[v-y_len+1:v+1].values for v in id_train])
    # tid
    D = T[features[1][:3]]
    # tidext
    X_test_2 = np.array([D[v-y_len+1:v+1].values for v in id_test])
    X_dev_2 = np.array([D[v-y_len+1:v+1].values for v in id_dev])
    X_train_2 = np.array([D[v-y_len+1:v+1].values for v in id_train])
    
    X_test = np.concatenate((X_test,X_test_2),axis=1)
    X_dev = np.concatenate((X_dev,X_dev_2),axis=1)
    X_train = np.concatenate((X_train,X_train_2),axis=1)
else:
    print('\nNo such feature.\n')

print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)
print('X_dev: ', X_dev.shape)
print('Y_dev: ', Y_dev.shape)
print('X_test: ', X_test.shape)
print('Y_test: ', Y_test.shape)
print()

#%% Train
n_train = X_train.shape[0]
n_dev = X_dev.shape[0]
n_test = X_test.shape[0]
n_input = X_train.shape[1]
n_output = Y_train.shape[1]
print('n_train = ', n_train)
print('n_dev = ', n_dev)
print('n_test = ', n_test)
print()
print('n_input = ', n_input)
print('n_output = ', n_output)

#n_hidden = args.layers
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
    print()
#    save_path = saver.save(sess, dir1+"final.ckpt")
    
with tf.Session() as sess:
    saver.restore(sess, dir0+"final.ckpt")
#    mse_dev = mse.eval(feed_dict={X:X_dev, Y:Y_dev})
#    mse_test = mse.eval(feed_dict={X:X_test, Y:Y_test})
    check_test_outputs,mse_test = sess.run([outputs,mse],feed_dict={X:X_test, Y:Y_test})
    check_dev_outputs,mse_dev = sess.run([outputs,mse],feed_dict={X:X_dev, Y:Y_dev})
    
    writer = tf.summary.FileWriter(dir0, graph=tf.get_default_graph())
    writer.close()
    
rmse_dev = np.sqrt(mse_dev)
rmse_test = np.sqrt(mse_test)
mse_test_ts = ((Y_test-check_test_outputs)**2).mean(axis=0)
rmse_test_ts = np.sqrt(mse_test_ts)
mse_dev_ts = ((Y_dev-check_dev_outputs)**2).mean(axis=0)
rmse_dev_ts = np.sqrt(mse_dev_ts)
corr_dev = np.array([np.corrcoef(v0,v1)[0,1] for v0,v1 in zip(Y_dev,check_dev_outputs)]).mean()
corr_test = np.array([np.corrcoef(v0,v1)[0,1] for v0,v1 in zip(Y_test,check_test_outputs)]).mean()

print("Best dev mse = {:.6f}, rmse = {:.6f}, corr = {:.4f}".format(mse_dev,rmse_dev,corr_dev))
print("Test mse = {:.6f}, rmse = {:.6f}, corr = {:.4f}".format(mse_test,rmse_test,corr_test))

    
#%% Random check
n_check = 5
check_indices = np.random.permutation(n_test)[:n_check]

check_X_dt = X_test_dt[check_indices]
check_Y_dt = Y_test_dt[check_indices]
check_X = X_test[check_indices]
check_Y = Y_test[check_indices]
if (feature=='obs') | (feature=='sur'):
    check_XY_dt = np.concatenate((check_X_dt,check_Y_dt),axis=1)
    check_XY = np.concatenate((check_X,check_Y),axis=1)
elif feature=='obs_tid':
    check_X0,check_X1 = np.split(check_X,2,axis=1)
    check_XY_dt = np.concatenate((check_X_dt,check_Y_dt),axis=1)
    check_XY = np.concatenate((check_X0,check_Y),axis=1)
elif feature=='obs_tidext':
    check_X0,check_X1 = np.split(check_X,[x_len],axis=1)
    check_XY_dt = np.concatenate((check_X_dt,check_Y_dt),axis=1)
    check_XY = np.concatenate((check_X0,check_Y),axis=1)
elif feature=='sur_tidext':
    check_X0,check_X1 = np.split(check_X,[x_len],axis=1)
    check_XY_dt = np.concatenate((check_X_dt,check_Y_dt),axis=1)
    check_XY = np.concatenate((check_X0,check_Y),axis=1)
else:
    print('\nFeature error in random check.\n')

with tf.Session() as sess:
    saver.restore(sess, dir0+"final.ckpt")
    check_outputs = sess.run(outputs,feed_dict={X:check_X, Y:check_Y})
check_mse = ((check_Y-check_outputs)**2).mean(axis=1)
check_rmse = np.sqrt(check_mse)
check_corr = np.array([np.corrcoef(v0,v1)[0,1] for v0,v1 in zip(check_Y,check_outputs)])

fig, axes = plt.subplots(n_check,1, figsize=(9, 15))
for i in range(n_check):
    ax = axes[i]
    if (feature=='obs') | (feature=='sur'):
        ax.plot(check_XY_dt[i],check_XY[i],'k.-',label='obs')
        ax.plot(check_Y_dt[i],check_outputs[i],'r.-',label='pred')
        ax.legend(loc=2)
    elif feature=='obs_tid':
        ax.plot(check_XY_dt[i],check_XY[i],'k.-',label='obs')
        ax.plot(check_X_dt[i],check_X1[i],'b.-',label='tid')
        ax.plot(check_Y_dt[i],check_outputs[i],'r.-',label='pred')
        ax.legend(loc=2)
    elif feature=='obs_tidext':
        ax.plot(check_XY_dt[i],check_XY[i],'k.-',label='obs')
        ax.plot(check_XY_dt[i],check_X1[i],'b.-',label='tid')
        ax.plot(check_Y_dt[i],check_outputs[i],'r.-',label='pred')
        ax.legend(loc=2)
    elif feature=='sur_tidext':
        ax1 = ax.twinx()
        ax.plot(check_XY_dt[i],check_XY[i],'k.-',label='sur')
        ax.plot(check_Y_dt[i],check_outputs[i],'r.-',label='pred')
        ax.legend(loc=2)
        ax.set_ylabel('sur (m)')
        ax1.plot(check_Y_dt[i],check_X1[i],'b.-',label='tid')
        ax1.legend(loc=1)
        ax1.set_ylabel('tid (m)',color='b')
        ax1.tick_params('y',colors='b')
    else:
        print('\nFeature error in random check plots.\n')
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    dt_fmt = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(dt_fmt)
    ax.xaxis.set_minor_locator(hours)
    ax.set_title('rmse = {:.4f}m, corr = {:.4f}'.format(check_rmse[i],check_corr[i]))
ax.set_xlabel('{:s}   {:s}   X:{:d}h   Y:{:d}h\ndev rmse = {:.4f}m, corr = {:.4f}   test rmse = {:.4f}m, corr = {:.4f}'.format(station,feature,x_len,y_len,rmse_dev,corr_dev,rmse_test,corr_test),weight='bold')
fig.tight_layout()
fig.savefig(dir0+'check.png', format='png', dpi=300)
plt.close(fig)

#%% MSE plot
fig, ax = plt.subplots(1,1, figsize=(8, 4))
ax.plot(rmse_dev_ts,'g.-',label='dev')
ax.plot(rmse_test_ts,'k.-',label='test')
ax.set_ylabel('rmse (m)')
ax.set_xlabel('timestep')
ax.set_title('{:s}   {:s}   X:{:d}h   Y:{:d}h\ndev rmse = {:.4f}m, corr = {:.4f}   test rmse = {:.4f}m, corr = {:.4f}'.format(station,feature,x_len,y_len,rmse_dev,corr_dev,rmse_test,corr_test),weight='bold')
ax.legend(loc=2)
fig.tight_layout()
fig.savefig(dir0+'rmse.png', format='png', dpi=300)
plt.close(fig)

fig, ax = plt.subplots(1,1, figsize=(8, 4))
ax.plot(mse_dev_ts,'g.-',label='dev')
ax.plot(mse_test_ts,'k.-',label='test')
ax.set_ylabel('mse')
ax.set_xlabel('timestep')
ax.set_title('{:s}   {:s}   X:{:d}h   Y:{:d}h\ndev mse = {:.4f}, corr = {:.4f}   test mse = {:.4f}, corr = {:.4f}'.format(station,feature,x_len,y_len,mse_dev,corr_dev,mse_test,corr_test),weight='bold')
ax.legend(loc=2)
fig.tight_layout()
fig.savefig(dir0+'mse.png', format='png', dpi=300)
plt.close(fig)
#%%
fig, ax = plt.subplots(1,1, figsize=(8, 4))
ax.plot(EPOCH,MSE_TRAIN,'k.:',label='train')
ax.plot(EPOCH,MSE_DEV,'k.-',label='dev')
ax.legend(loc=1)
ax.set_ylabel('mse')
ax.set_xlabel('epoch')
ax.set_title('{:s}   {:s}   X:{:d}h   Y:{:d}h\ndev mse = {:.4f}, corr = {:.4f}   test mse = {:.4f}, corr = {:.4f}'.format(station,feature,x_len,y_len,mse_dev,corr_dev,mse_test,corr_test),weight='bold')
fig.tight_layout()
fig.savefig(dir0+'train_dev.png', format='png', dpi=300)
plt.close(fig)

#%% Save
np.savez(dir0+'runinfo',
         EPOCH=EPOCH,MSE_TRAIN=MSE_TRAIN,MSE_DEV=MSE_DEV,
         mse_test_ts=mse_test_ts,mse_dev=mse_dev,mse_test=mse_test,
         rmse_test_ts=rmse_test_ts,rmse_dev=rmse_dev,rmse_test=rmse_test,
         corr_dev=corr_dev,corr_test=corr_test,
         x_len=x_len,y_len=y_len,n_hidden=n_hidden,
         station=station,feature=feature)
#np.load(dir0+'train_dev_test.npz')

sys.stdout = orig_stdout
f.close()
