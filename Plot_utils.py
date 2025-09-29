#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,roc_auc_score


def Plot_lr_curve(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy'] 
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(122)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
def Plot_confusion_matrix(true_d, pred):
    
    pred_label = []
    true_label = []
    
    for i in range(len(pred)):
        pred_label.append(np.argmax(pred[i]))
        
    for i in range(len(true_d)):
        true_label.append(np.argmax(true_d[i]))
    
    print(pred_label[0:10])
    # Accuracy
    accuracy = accuracy_score(true_label, pred_label)
    print('accuracy : ', np.round(accuracy,3))
    
    # Precision
    precision = precision_score(true_label, pred_label, average='macro')
    print('precision : ', np.round(precision,3))
    
    # Recall
    recall = recall_score(true_label, pred_label,average='macro')
    print('recall : ', np.round(recall,3))
    
    # Specificity
    specificity = recall_score(true_label, pred_label,average='macro')
    print('specificity : ',np.round(specificity,3))
    
    # F1 Score
    F1_score = f1_score(true_label, pred_label,average='macro')
    print('F1_score : ', np.round(F1_score,3))

    #clasification report
    report = classification_report(true_label, pred_label, target_names=['Excitment','Neutral','Stress'])#'Amuse',
    print(report)
    
    #confusion matrix
    confusion = confusion_matrix(true_label, pred_label)
    disp = ConfusionMatrixDisplay(confusion, display_labels=['Excitment','Neutral', 'Stress'])#'Amuse',
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
    #plt.figure(figsize=(10,5))
    cmn = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Excitment', 'Neutral','Stress'], yticklabels=['Excitment', 'Neutral','Stress'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)

