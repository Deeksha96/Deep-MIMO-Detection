import math
import tensorflow as tf
import pylab
import numpy as np
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops

DataSet_x=[]
DataSet_F1=[]
DataSet_F2=[]
NrSamples = 100000
NrSamples_ToTest = 10000
SNR_dB_range=np.arange(0,22,2)
BER=np.zeros(SNR_dB_range.shape,dtype=np.int)
Sigma2 = 1.0  
SNR_min_dB = 0.0
SNR_max_dB = 20.0 
K = 2
N = 2*K
L = 3*K
batch_size=1000
number_of_batches=NrSamples/batch_size
c=0
kk=0
l=1
for runIdx in range(0,NrSamples):
    # generate a random operating SNR
    SNR_dB = SNR_min_dB + (SNR_max_dB - SNR_min_dB)*np.random.rand()
    SNR = 10.0**(SNR_dB/10.0)
    H = ((0.5*float(SNR)/float(K))**(1/2.0))*np.random.randn(N,K)
    x = 2*np.round(np.random.rand(K,1))-1;
    h_prime=np.transpose(H)
    noise = (Sigma2/2.0)**(1/2.0)*np.random.randn(N,1) 
    y = np.matmul(H,x) + noise
    c=c+1
    DataSet_x.append(x)
    DataSet_F1.append(np.matmul(h_prime,y))
    DataSet_F2.append(np.matmul(h_prime,H))    
print("training_data")
kk=0
kk2=0
kk3=0
X=tf.placeholder(tf.float64,shape=(batch_size,2,1))
F1=tf.placeholder(tf.float64,shape=(batch_size,2,1))
F2=tf.placeholder(tf.float64,shape=(batch_size,2,2))

#Test data
TestSet_x=[]
TestSet_F1=[]
TestSet_F2=[]
SNR_dB_range=np.arange(0,22,2)
BER=np.zeros(SNR_dB_range.shape,dtype=np.int)
Sigma2 = 1.0  
SNR_min_dB = 0.0
SNR_max_dB = 20.0 
K = 2
N = 2*K
L = 3*K
number_of_batches_test=NrSamples_ToTest/batch_size
c=0
for runIdx in range(0,NrSamples_ToTest):
    # generate a random operating SNR
    SNR_dB = SNR_min_dB + (SNR_max_dB - SNR_min_dB)*np.random.rand()
    SNR = 10.0**(SNR_dB/10.0)
    H = ((0.5*float(SNR)/float(K))**(1/2.0))*np.random.randn(N,K)
    x = 2*np.round(np.random.rand(K,1))-1;
    h_prime=np.transpose(H)
    noise = (Sigma2/2.0)**(1/2.0)*np.random.randn(N,1)
    y = np.matmul(H,x) + noise
    c=c+1
    TestSet_x.append(x)
    TestSet_F1.append(np.matmul(h_prime,y))
    TestSet_F2.append(np.matmul(h_prime,H))    
kk1=0
kk2=0
kk3=0
with tf.variable_scope("embedding") as scope:
    if kk1 > 0:
      scope.reuse_variables() 
    else:
        W1=tf.get_variable("embedding1",initializer=tf.random_normal([L,8*K,5*K],stddev=0.1,dtype=tf.float64),trainable=True,dtype=tf.float64)
        b1=tf.get_variable("embedding2",initializer=tf.random_normal([L,8*K,1],stddev=0.1,dtype=tf.float64),trainable=True,dtype=tf.float64)
    kk1=1

	
	
with tf.variable_scope("embedding") as scope:
    if kk2 > 0:
      scope.reuse_variables() 
    else:
        W2=tf.get_variable("embedding3",initializer=tf.random_normal([L,K,8*K],stddev=0.1,dtype=tf.float64),trainable=True,dtype=tf.float64)
        b2=tf.get_variable("embedding4",initializer=tf.random_normal([L,K,1],stddev=0.1,dtype=tf.float64),trainable=True,dtype=tf.float64)

with tf.variable_scope("embedding") as scope:
    if kk3 > 0:
      scope.reuse_variables() 
    else:
      W3=tf.get_variable("embedding5",initializer=tf.random_normal([L,2*K,8*K],stddev=0.1,dtype=tf.float64),trainable=True,dtype=tf.float64)
      b3=tf.get_variable("embedding6",initializer=tf.random_normal([L,2*K,1],stddev=0.1,dtype=tf.float64),trainable=True,dtype=tf.float64)

def zk(Hty,xk,HtH,vk,k,name='embedding'):  #(f1,x_hat,f2,v_hat,k)
	inp3=tf.matmul(HtH,xk)
	concat=tf.concat([Hty,xk,inp3,vk],0)
	temp=tf.matmul(W1[k-1],concat)+b1[k-1]
	zk=tf.nn.relu(temp)
	return zk

def psi(x,tt):
	t=tt*tf.ones_like(x)
	relu1=tf.nn.relu((x+t))
	relu2=tf.nn.relu((x-t))
	ab=tf.abs(t)
	out1=tf.div(relu1,ab)
	out2=tf.div(relu2,ab)
	one=tf.ones_like(out1,dtype=tf.float64)
	temp=tf.add(one,out2)
	return tf.subtract(out1,temp)


def xk(z_hat,k,name='embedding'):
	return psi(tf.matmul(W2[k-1],z_hat)+b2[k-1],tt=0.1)

def vk(z_hat,k,name='embedding'):
	return tf.matmul(W3[k-1],z_hat)+b3[k-1]

def x_tilde(HtH,Hty):
	HtH_inv=tf.matrix_inverse(HtH)
	return tf.matmul(HtH_inv,Hty)
def model(x_f,f1_f,f2_f):
    loss_2=0
    pred=[]
    for dsIdx in range(0,batch_size):
        x = x_f[dsIdx]
        f1 = f1_f[dsIdx] ##hty
        f2 = f2_f[dsIdx]##hth
        x_hat=tf.zeros((K,1),dtype = tf.float64)
        v_hat=tf.zeros((2*K,1), dtype = tf.float64)
        x_tilde_num=x_tilde(f2,f1)
        temp_loss=0                
        for k in range(1,L+1):      
            z_hat = zk(f1,x_hat,f2,v_hat,k)
            x_hat=xk(z_hat,k)
            v_hat=vk(z_hat,k)       
            num=math.log(k)*tf.abs(x-x_hat)*tf.abs(x-x_hat)
            denom=tf.abs(tf.subtract(x,x_tilde_num))*tf.abs(tf.subtract(x,x_tilde_num))
            loss_k=tf.div(num,denom)
            loss_k=tf.reduce_sum(loss_k)
            pred.append(x_hat)
            temp_loss=temp_loss+loss_k
        loss_2=loss_2+temp_loss
			############ Combined Loss of all layers ########################
    return loss_2,pred


print("model")
loss_, out=model(X,F1,F2)
loss=loss_/batch_size
tf.summary.histogram("loss",loss)
loss=tf.clip_by_value(loss,clip_value_min=0,clip_value_max=1000)
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
saver=tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
list_of_variables = tf.all_variables()
sess.run(tf.global_variables_initializer())
variable_names= [v.name for v in tf.trainable_variables()]
uninitialized_variables = list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables(list_of_variables)))

init_op = tf.variables_initializer(list_of_variables)
sess.run(tf.variables_initializer(uninitialized_variables))
print(sess.run(tf.report_uninitialized_variables(list_of_variables)))

t=0
with sess.as_default():
                      
    merged=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter("train_loss_f",sess.graph)
    for epoch in range(100,4000):
        loss_ep=0
        random.shuffle(DataSet_x)
        random.shuffle(DataSet_F1)
        random.shuffle(DataSet_F2)
        print("ep")
        for i in range(0,int(number_of_batches)):
            train_step.run(feed_dict={X: DataSet_x[i*batch_size:i*batch_size+batch_size], F1:DataSet_F1[i*batch_size:i*batch_size+batch_size], F2:DataSet_F2[i*batch_size:i*batch_size+batch_size]})
            loss1,summary=sess.run([loss,merged],feed_dict={X: DataSet_x[i*batch_size:i*batch_size+batch_size], F1:DataSet_F1[i*batch_size:i*batch_size+batch_size], F2:DataSet_F2[i*batch_size:i*batch_size+batch_size]})
            loss_ep=loss_ep+loss1
            file=open("loss_f.txt","a")
            file.write("loss"+" "+str(i)+" " +str(loss1)+"\n")
            file.close()
            print("loss"+" "+str(loss1))
            train_writer.add_summary(summary,t)
            t=t+1
        file=open("loss_f.txt","a")
        file.write("loss_ep"+" "+str(loss_ep/number_of_batches)+"\n")
        file.close()
        loss_ep_t=0	
        for j in range(0,int(number_of_batches_test)):
            loss_t, out_t=sess.run([loss,out],feed_dict={X: TestSet_x[j*batch_size:j*batch_size+batch_size], F1:TestSet_F1[j*batch_size:j*batch_size+batch_size], F2:TestSet_F2[j*batch_size:j*batch_size+batch_size]})
            loss_ep_t=loss_ep_t+loss_t
            file=open(str(epoch)+"-"+"test_labels_f.txt","a")
            for z in range(0,batch_size):
                file.write("test_input"+" "+str(TestSet_x[j*batch_size+z][0][0])+" "+str(TestSet_x[j*batch_size+z][1][0])+"\n")
                file.write("test_label"+" "+str(out_t[z][0][0])+" "+str(out_t[z][1][0])+"\n")
            file.close()    		
        file1=open("test_loss_f.txt","a")
        file1.write("test_loss"+" "+str(loss_ep_t/number_of_batches_test)+"\n")
        file1.close()
        print("loss_ep")
        print(loss_ep/number_of_batches)
		

