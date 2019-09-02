# TODO attention
# TODO autoencodeur

import pandas as pd
import seaborn as sn
from pandas_ml import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
from preprocess import get_data_loader
from settings import (DEVICE, NB_EPOCHS, HIDDEN_SIZE, ROOT_DATASET, N_LAYERS,
                      LOAD_PREVIOUS, LEARNING_RATE, NUM_WORKERS, BATCH_SIZE)
import torch
from torch import nn
from preprocess import load_data
from archi import SacadeRnn, Classifier, Autoencoder
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
from matplotlib import pyplot as plt
from utils import Checkpoints, load_models


vectorized_persons, voc = load_data(ROOT_DATASET)
train_loader, valid_loader = get_data_loader(
    vectorized_persons, workers=NUM_WORKERS)

check = Checkpoints()
(sacade_rnn,
 classifier,
 autoencoder) = load_models(voc.num_user, load_previous_state=LOAD_PREVIOUS,
                            load_classifier=False)
Cel = nn.CrossEntropyLoss()

optimizerGru = torch.optim.Adam(sacade_rnn.parameters(), lr=LEARNING_RATE)
optimizerClas = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
optimizerAuto = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

sacade_rnn = sacade_rnn.to(DEVICE)
classifier = classifier.to(DEVICE)
autoencoder = autoencoder.to(DEVICE)
Cel = Cel.to(DEVICE)
# ##########
# Training #
# ##########
print("torch version : ", torch.__version__)
print("Device : ", DEVICE)
# # torch.autograd.set_detect_anomaly(True)
writer = SummaryWriter(flush_secs=10)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
y_pred = np.array([])
y_true = np.array([])
for i_epoch in range(NB_EPOCHS):
    print(i_epoch)
    # print(type(train_loader))

    for i_batch, batch in enumerate(train_loader):
        # print(i_batch)
        optimizerGru.zero_grad()
        optimizerClas.zero_grad()

        sessions, lengths, userids = batch
        sessions = sessions.to(DEVICE)
        userids = userids.to(DEVICE)
        # print(sessions.size())
        sessions = autoencoder(sessions)
        # print(sessions.size())

        out = sacade_rnn(sessions, lengths)
        # TODO out = attention(out)
        # TODO? out = sacade_rnn(out)
        # TODO out = attention(out)

        out = classifier(out)
        out = out.to(DEVICE)
        # print(out.size())
        # print(userids.size())
        loss = Cel(out, userids)
        check.addCheckpoint("loss", loss)
        check.save("loss", loss, sacade_rnn, classifier, autoencoder)
        loss.backward()

        optimizerGru.step()
        optimizerClas.step()

        writer.add_scalar("Loss", loss,
                          global_step=i_batch+len(train_loader)*i_epoch)

    score = 0
    sacade_rnn = sacade_rnn.eval()
    classifier = classifier.eval()
    autoencoder = autoencoder.eval()
    for i_batch, batch in enumerate(train_loader):

        sessions, lengths, userids = batch

        sessions = sessions.to(DEVICE)
        userids = userids.to(DEVICE)

        enc_sessions = autoencoder(sessions)
        out_rnn = sacade_rnn(enc_sessions, lengths)
        out_rnn = out_rnn.to(DEVICE)
        out = classifier(out_rnn)
        out = out.to(DEVICE)
        # print(out.size())
        # if i_epoch > 30:
        y_true = np.append(y_true, userids.cpu().data.squeeze().numpy())
        y_pred = np.append(y_pred, torch.argmax(
            out, dim=1).cpu().data.squeeze().numpy())
        score += torch.sum(torch.argmax(out, dim=1) == userids)
        # notGood = list((torch.argmax(out, dim=1) != userids).cpu())

        if torch.sum(torch.argmax(out, dim=1) == userids) != BATCH_SIZE:
            val, ind = torch.topk(F.softmax(out, dim=-1), 3, dim=1)
            sort, _ = torch.sort(F.softmax(out, dim=-1), dim=-1)
            sort = sort.cpu().data
            # print("Probabilite sortie : \n", sort.squeeze())
            print(voc.index2title[int(sessions[-1][0][1])])
            print("Trois plus grande prob :")
            for v in val.squeeze().data:
                print(f"{round(float(v.cpu().numpy()),3):>10}", end=" ")
            print("\n")
            for u in ind.squeeze().cpu().data:
                print(f"{voc.index2user[int(u)]:>10}", end=" ")
            print("\n   ", voc.index2user[int(userids)], " <= Real")
    print(f"{score}/{len(train_loader)*BATCH_SIZE} => ", end=" ")
    score = float(score)/float(len(train_loader)*BATCH_SIZE)
    writer.add_scalar("Score", score, global_step=i_epoch)
    print(f"score : {score} ")
    sacade_rnn = sacade_rnn.train()
    classifier = classifier.train()
    autoencoder = autoencoder.train()

writer.close()

confusion_matrix = confusion_matrix(y_true, y_pred)
classes = [voc.index2user[u] for u in unique_labels(y_true, y_pred)]
confusion_matrix = confusion_matrix.astype(
    'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
sn.heatmap(confusion_matrix, annot=True, robust=True, fmt='.1f',
           xticklabels=classes, yticklabels=classes)
# sn.clustermap(confusion_matrix)
plt.show()
