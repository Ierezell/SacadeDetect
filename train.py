# TODO attention
# TODO autoencodeur

import datetime

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch import nn
from tqdm import tqdm

from preprocess import get_data_loader, load_data
from pretty_conf_mat import plot_confusion_matrix_from_data
from settings import (BATCH_SIZE, CONFIG, DEVICE, HIDDEN_SIZE, LEARNING_RATE,
                      LOAD_PREVIOUS, N_LAYERS, NB_EPOCHS, NUM_WORKERS,
                      PATH_WEIGHTS_AUTOENCODER, PATH_WEIGHTS_CLASSIFIER,
                      PATH_WEIGHTS_RNN, ROOT_DATASET)
from utils import Checkpoints, load_models, print_parameters

# plt.ion()

plt.figure()
wandb.init(project="SacadeDetect",
           name=f"test-{datetime.datetime.now().replace(microsecond=0)}",
           resume=LOAD_PREVIOUS,
           config=CONFIG)

vectorized_persons, voc = load_data(ROOT_DATASET)
train_loader, valid_loader = get_data_loader(
    vectorized_persons, workers=NUM_WORKERS)

check = Checkpoints()
(sacade_rnn,
 classifier,
 autoencoder) = load_models(voc.num_user, load_previous_state=LOAD_PREVIOUS,
                            load_classifier=False)

wandb.watch((sacade_rnn, classifier, autoencoder))

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
print("Nombre d'eleves : ", voc.num_user)
print("Nombre de donnees : ", voc.num_user)
# print("Nombre d'eleve reduit : ", voc.user2index.keys())

print_parameters(sacade_rnn)
print_parameters(classifier)
print_parameters(autoencoder)

y_pred = np.array([])
y_true = np.array([])

for i_epoch in range(NB_EPOCHS):
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

        out = classifier(out)
        out = out.to(DEVICE)
        # print(out.size())
        # print(userids.size())
        loss = Cel(out, userids)
        check.addCheckpoint("loss", loss)
        check.save("loss", loss, sacade_rnn, classifier, autoencoder)
        loss.backward()
        wandb.log({"loss": loss})
        optimizerGru.step()
        optimizerClas.step()

    # #############
    # FIN D'EPOCH #
    # #############
    score = 0
    sacade_rnn = sacade_rnn.eval()
    classifier = classifier.eval()
    autoencoder = autoencoder.eval()
    print(f"EVAL ! Epoch {i_epoch}")
    with torch.no_grad():
        for i_batch, batch in enumerate(valid_loader):
            sessions, lengths, userids = batch

            sessions = sessions.to(DEVICE)
            userids = userids.to(DEVICE)

            enc_sessions = autoencoder(sessions)
            out_rnn = sacade_rnn(enc_sessions, lengths)
            out_rnn = out_rnn.to(DEVICE)
            out = classifier(out_rnn)
            out = out.to(DEVICE)

            y_true = np.append(y_true, userids.cpu().data.squeeze().numpy())
            y_pred = np.append(y_pred, torch.argmax(
                out, dim=1).cpu().data.squeeze().numpy())
            score += torch.sum(torch.argmax(out, dim=1) == userids)

        print(f"{score}/{len(valid_loader)*BATCH_SIZE} => ", end=" ")
        score = float(score)/float(len(valid_loader)*BATCH_SIZE)

        print(f"score : {score} ")
        classes = [voc.index2user[u] for u in unique_labels(y_true, y_pred)]
        p = plot_confusion_matrix_from_data(y_true, y_pred, classes)

        wandb.log({"Score": score})
        wandb.log({"Confusion matrix": [wandb.Image(plt, caption="conf mat")]})

        wandb.save(PATH_WEIGHTS_AUTOENCODER)
        wandb.save(PATH_WEIGHTS_CLASSIFIER)
        wandb.save(PATH_WEIGHTS_RNN)

        sacade_rnn = sacade_rnn.train()
        classifier = classifier.train()
        autoencoder = autoencoder.train()

        # notGood = list((torch.argmax(out, dim=1) != userids).cpu())
        # if torch.sum(torch.argmax(out, dim=1) == userids) != BATCH_SIZE:
        #     val, ind = torch.topk(F.softmax(out, dim=-1), 3, dim=1)
        #     sort, _ = torch.sort(F.softmax(out, dim=-1), dim=-1)
        #     sort = sort.cpu().data
        #     # print("Probabilite sortie : \n", sort.squeeze())
        #     print(voc.index2title[int(sessions[-1][0][1])])
        #     print("Trois plus grande prob :")
        #     for v in val.squeeze().data:
        #         print(f"{round(float(v.cpu().numpy()),3):>10}", end=" ")
        #     print("\n")
        #     for u in ind.squeeze().cpu().data:
        #         print(f"{voc.index2user[int(u)]:>10}", end=" ")
        #     print("\n   ", voc.index2user[int(userids)], " <= Real")

        # conf_mat = confusion_matrix(y_true, y_pred)
        # df_conf_mat = pd.DataFrame(conf_mat, index=classes, columns=classes)
        # # conf_mat = conf_mat.astype(
        # #     'float') / conf_mat.sum(axis=1)[:, np.newaxis]

        # sn.heatmap(df_conf_mat, annot=True)
        # # sn.clustermap(confusion_matrix)
        # plt.show()
        # wandb.log({"Conf Mat": s})
