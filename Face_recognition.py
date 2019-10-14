import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda as L


#Some Variables
img_size=96
img_dims=(96,96,3)
BATCH_SIZE=5
EPOCHS=100
LEARNING_RATE=0.00001

################ Data importing ####################################
#We will take only 6 different faces

data_path=r'C:\Users\Anu\Downloads\Compressed\Face_recognition-master\Face_recognition-master\att_faces'
face_fold_ids=os.listdir(data_path)[0:6]

#Checkpoint Path
ckpt_path=r'C:\Users\Anu\Downloads\Compressed\Face_recognition-master\Face_recognition-master\model_checkpoint'

#First we will create pairs of same faces then different faces
#Each person has 10 images and we will take 8 only and keep 2 for testing our model
print('Working with Dataset')
def get_dataset(face_folder_ids):
    same_person_list=[]
    for f_id in tqdm(face_folder_ids):
        img_ids=os.listdir(os.path.join(data_path,f_id))
        for img in img_ids:
            image_1=os.path.join(data_path,f_id,img)
            #Select second image of same person randomly from the folder
            rand_ix=np.random.choice(img_ids)
            image_2=os.path.join(data_path,f_id,rand_ix)
            same_person_list.append([image_1,image_2])
    
    #Now lets do same process for images of two different peoples
    
    ###Function to get another random person id
    def get_other_folder(fol_ids,f_id):
        f_copy=list(np.copy(fol_ids))
        f_copy.remove(f_id)
        return np.random.choice(f_copy)
        
    different_person_list=[]
    for _ in tqdm(range(len(same_person_list))):
        f_id_1=np.random.choice(face_folder_ids)
        f_id_2=get_other_folder(face_folder_ids,f_id_1)
        #Get random
        img_1=np.random.choice(os.listdir(os.path.join(data_path,f_id_1)))
        img_2=np.random.choice(os.listdir(os.path.join(data_path,f_id_2)))
        image_1=os.path.join(data_path,f_id_1,img_1)
        image_2=os.path.join(data_path,f_id_2,img_2)
        different_person_list.append([image_1,image_2])
        
    image_pair_list=np.concatenate((same_person_list,different_person_list))
    #Now lets generate binary labels and concat our image lists
    labels=np.zeros((len(image_pair_list),1))
    labels[0:len(same_person_list)]=1.0 #first half pairs are of same people
    image_pair_list,labels=shuffle(image_pair_list,labels)
    return image_pair_list,labels

image_pair_list,labels=get_dataset(face_fold_ids)
#Split Dataset into train and 5% validation
t_image_list,v_image_list,t_labels,v_labels=train_test_split(image_pair_list,labels,test_size=0.25)

#lets take out 10% images from validation set which we will use for testing purpose
v_image_list,test_image_list,v_labels,test_labels=train_test_split(v_image_list,v_labels,test_size=0.14)

#Function to Load Images
def read_image(img):
    img=cv2.imread(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_size,img_size))
    img=img.astype('float')/255.0
    return img

def get_images(image_ids): 
    tmp_left_batch=np.zeros((len(image_ids),*img_dims))
    tmp_right_batch=np.zeros((len(image_ids),*img_dims))
    
    for ix,img in enumerate(image_ids):
        img_l=read_image(img[0])
        img_r=read_image(img[1])
        tmp_left_batch[ix]=img_l
        tmp_right_batch[ix]=img_r
    return tmp_left_batch,tmp_right_batch
################################################
    
#####################Placeholders
#Disable eage execution inorder to use placeholder
tf.disable_eager_execution()

inp_1=tf.placeholder(tf.float32,shape=(None,96,96,3),name='left_input')
inp_2=tf.placeholder(tf.float32,shape=(None,96,96,3),name='right_input')
inp_label=tf.compat.v1.placeholder(tf.float32,(None,1),name='target')

###########################################################################
#Model
model_weights=r'C:/Users/Anu/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
print('Creating Model')
def model(inp):
    x=tf.keras.layers.Conv2D(32,(10,10),padding='same',kernel_initializer='he_normal')(inp)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x=tf.keras.layers.Conv2D(64,(6,6),padding='same',kernel_initializer='he_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x=tf.keras.layers.Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x=tf.keras.layers.Flatten()(x)    
    return x

feat_1=model(inp_1)
feat_2=model(inp_2)
l1_distance_layer = L(lambda tensors: K.abs(tensors[0] - tensors[1]))
l1_distance = l1_distance_layer([feat_1, feat_2])
out=tf.keras.layers.Dense(1,activation='sigmoid')(l1_distance)


#Loss,Optimizer,saver
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(inp_label,out))
train_opt=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
saver=tf.train.Saver()

#Train Batches
train_batch_size=int(len(t_image_list)/BATCH_SIZE)
val_batch_size=int(len(v_image_list)/BATCH_SIZE)
print('Total Train Batches: {}'.format(train_batch_size))
print('Total Val Batches: {}'.format(val_batch_size))
        
train_loss_list,val_loss_list=[],[]

###################################################
with tf.Session() as sess:
    tmp_v_loss=[]
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #Training
    for epoch in range(EPOCHS):
        print('EPOCH {}/{}'.format((epoch+1),EPOCHS))
        print('TRAINING..')
        for ix in tqdm(range(train_batch_size)):
            image_ix=t_image_list[ix*BATCH_SIZE:(ix+1)*BATCH_SIZE]
            left_input,right_input=get_images(image_ix)
            train_targets=t_labels[ix*BATCH_SIZE:(ix+1)*BATCH_SIZE]
            
            input_dict={inp_1:left_input,inp_2:right_input,inp_label:train_targets}
            t_loss,_=sess.run([loss,train_opt],feed_dict=input_dict)
        
        for ix in tqdm(range(val_batch_size)):
            image_ix=v_image_list[ix*BATCH_SIZE:(ix+1)*BATCH_SIZE]
            left_input,right_input=get_images(image_ix)
            val_targets=v_labels[ix*BATCH_SIZE:(ix+1)*BATCH_SIZE]
            
            input_dict={inp_1:left_input,inp_2:right_input,inp_label:val_targets}
            v_loss=sess.run(loss,feed_dict=input_dict)
        print('Training Loss: {}'.format(t_loss))
        print('Validation Loss: {}'.format(v_loss))
        train_loss_list.append(t_loss)
        val_loss_list.append(v_loss)
        
        #####Checkpoint
        if epoch==0: #Save first Checkpoint
            saver.save(sess,os.path.join(ckpt_path,'model.ckpt'))
            print('Validation Loss improved from 0 to {}'.format(v_loss))
        elif v_loss<=np.min(val_loss_list):
            saver.save(sess,os.path.join(ckpt_path,'model.ckpt'))
            print('Validation Loss improved from {} to {}'.format(tmp_v_loss,v_loss))
        else:
            print('Validation Loss didnot improve')
        
        tmp_v_loss=v_loss
        
        #Shuffle Dataset after every epoch
        t_image_list,t_labels=shuffle(t_image_list,t_labels)
        v_image_list,v_labels=shuffle(v_image_list,v_labels)
        

###Plotting
loss=train_loss_list
val_loss=val_loss_list
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'b',color='red',label='Training Loss')
plt.plot(epochs,val_loss,'b',color='blue',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.figure()
plt.show()
        

#Test model on unseen test images

#Read test Images
def get_test_images(image_id):
    image_l=read_image(image_id[0])
    image_l=np.expand_dims(image_l,0)
    
    image_r=read_image(image_id[1])
    image_r=np.expand_dims(image_r,0)
    return image_l,image_r

tmp_preds=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,os.path.join(ckpt_path,'model.ckpt'))
    for img_ix in tqdm(test_image_list):
        image_left,image_right=get_test_images(img_ix)
        tmp_preds.extend(sess.run(out,feed_dict={inp_1:image_left,inp_2:image_right}))
predictions=[]
for i in tmp_preds:
    if i>=0.5:
        predictions.append(1)
    else: 
        predictions.append(0)
        
print(roc_auc_score(test_labels,predictions))
confusion_matrix(test_labels,predictions)