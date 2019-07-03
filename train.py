
from torch.utils.tensorboard import SummaryWriter
from preprocess import get_data_loader
from settings import (DEVICE, NB_EPOCHS, HIDDEN_SIZE,
                      LEARNING_RATE, NUM_WORKERS, BATCH_SIZE)
import torch
from torch import nn
from preprocess import load_data
from archi import SacadeRnn, Classifier

vectorized_persons, voc = load_data()
train_loader = get_data_loader(vectorized_persons, workers=NUM_WORKERS)

sacade_rnn = SacadeRnn(62, voc.num_user, n_layers=2)
classifier = Classifier(voc.num_user)
Cel = nn.CrossEntropyLoss()

optimizerGru = torch.optim.Adam(sacade_rnn.parameters(), lr=LEARNING_RATE)
optimizerClas = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

sacade_rnn = sacade_rnn.to(DEVICE)
classifier = classifier.to(DEVICE)
Cel = Cel.to(DEVICE)
# ##########
# Training #
# ##########
print("torch version : ", torch.__version__)
print("Device : ", DEVICE)
# # torch.autograd.set_detect_anomaly(True)
writer = SummaryWriter(flush_secs=10)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(len(train_loader), "batches of ", BATCH_SIZE, " elements")
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        # print(i_batch)
        optimizerGru.zero_grad()
        optimizerClas.zero_grad()

        sessions, lengths, userids = batch

        sessions = sessions.to(DEVICE)
        userids = userids.to(DEVICE)

        out = sacade_rnn(sessions, lengths)

        out = out.to(DEVICE)

        out = classifier(out)
        out = out.to(DEVICE)

        loss = Cel(out, userids)
        loss.backward()

        optimizerGru.step()
        optimizerClas.step()

        writer.add_scalar("Loss", loss,
                          global_step=i_batch+len(train_loader)*i_epoch)

    score = 0
    for i_batch, batch in enumerate(train_loader):
        sessions, lengths, userids = batch

        sessions = sessions.to(DEVICE)
        userids = userids.to(DEVICE)

        out = sacade_rnn(sessions, lengths)
        out = out.to(DEVICE)
        out = classifier(out)
        out = out.to(DEVICE)

        score += torch.sum(torch.argmax(out, dim=1) == userids)
    print(f"{score}/{len(train_loader)*BATCH_SIZE} => ", end=" ")
    score = float(score)/float(len(train_loader)*BATCH_SIZE)
    writer.add_scalar("Score", score,
                      global_step=i_epoch)
    print(f"score : {score} ")
writer.close()
