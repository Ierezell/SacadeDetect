import torch

from data import Donnees
from torch.utils.data import Dataset, DataLoader
from settings import BATCH_SIZE, DEVICE
import numpy as np


def load_data(pathJson="./Dataset/events.json"):
    donnes = Donnees(pathJson)
    donnes.load_json()
    donnes.remove_mobile()
    donnes.remove_docs()
    donnes.remove_small()
    donnes.create_dict_persons()
    donnes.create_voc()
    donnes.to_numeral()
    donnes.to_vector()
    return donnes.vectorized_persons, donnes.voc


class keyboardLoader(Dataset):
    def __init__(self, datas):
        super(keyboardLoader, self).__init__()
        self.datas = datas

    def __getitem__(self, index):
        data = torch.tensor(
            np.array(self.datas[index][1], dtype=np.float), dtype=torch.float)
        userid = torch.tensor(
            np.array(self.datas[index][0], dtype=np.float), dtype=torch.long)
        return data, userid

    def __len__(self):
        return len(self.datas)


def collate_pad(batch):
    lengths = []
    datas = []
    userids = []
    for tenseur, userid in batch:
        lengths.append(tenseur.size(0))
        userids.append(userid)
        datas.append(tenseur)
    datas = torch.nn.utils.rnn.pad_sequence(datas)
    userids = torch.stack(userids)
    return datas, lengths, userids
# in dataloader : collate_fn=collate_pad


def get_data_loader(datas, split_percent=0.8, workers=0):

    datas = keyboardLoader(datas=datas)
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    pin = False if DEVICE.type == 'cpu' else True

    train_size = int(split_percent * len(datas))
    test_size = len(datas) - train_size
    train_datas, valid_datas = torch.utils.data.random_split(datas,
                                                             [train_size,
                                                              test_size])

    train_loader = DataLoader(train_datas, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pad, num_workers=workers,
                              pin_memory=pin)

    valid_loader = DataLoader(valid_datas, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pad, num_workers=workers,
                              pin_memory=pin)

    return train_loader, valid_loader


# def get_notebooks_person(person_id):
#     pass


# def zeroPadding(l: List[List[int]], fillvalue=PAD_token) -> List[List[int]]:
#     """Convert list of list of ints to a matrix padded to get same dimmension
#     Arguments:
#         l {List[List[str]]} -- list of our senteces (with differents sizes)
#     Keyword Arguments:
#         fillvalue -- value to fill with (default: {PAD_token})
#     Returns:
#         list[list[int]] -- matrix with sentences padded
#     """
#     return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# def maskMatrix(l: List[List[int]]) -> List[List[bool]]:
#     """Gives the boolean matrix to know if element i,j is paddeding or not
#     Arguments:
#         l {List[List[str]]} -- list of our sentences converted first to ints
#     Returns:
#         List[List[bool]] -- matrix to know where padding is
#     """
#     mask = []
#     for i, seq in enumerate(l):
#         mask.append([])
#         for token in seq:
#             if token == PAD_token:
#                 mask[i].append(0)
#             else:
#                 mask[i].append(1)
#     return mask

# # Returns padded input sequence tensor and lengths


# def toInputTensor(voc, l: List[List[str]]):
#     """Get list of sentences (list of words) and return a matrix with these
#     sentences padded and converted to numbers. The fonction zeroPadding is
#     applying a transpose !
#     Arguments:
#         voc {Voc} -- The class for our vocabulary
#         l {List[List[str]]} -- list of sentences (list of words)
#     Returns:
#         padVar -- matrix with padded sentences converted to int
#         lengths -- max length of the longest sentence in our list
#     """
#     indexes_batch = [voc.sentenceToIndex(sentence) for sentence in l]
#     lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
#     padList = zeroPadding(indexes_batch)
#     padVar = torch.LongTensor(padList)
#     return padVar, lengths

# # Returns padded target sequence tensor, padding mask, and max target length


# def toOutputTensor(voc, l: List[List[str]]):
#     """Get list of sentences (list of words) and return a matrix with these
#     sentences padded and converted to numbers. The fonction zeroPadding is
#     applying a transpose !
#     Arguments:
#         voc {Voc} -- The class for our vocabulary
#         l {List[List[str]]} -- list of sentences (list of words)
#     Returns:
#         padVar -- matrix with padded sentences converted to int
#         mask -- matrix to know which element is padding
#         max_target_len -- max length of the longest sentence in our list
#     """
#     indexes_batch = [voc.sentenceToIndex(sentence)
#                      for sentence in l]
#     max_target_len = max([len(indexes) for indexes in indexes_batch])
#     padList = zeroPadding(indexes_batch)
#     mask = maskMatrix(padList)
#     mask = torch.ByteTensor(mask)
#     padVar = torch.LongTensor(padList)
#     return padVar, mask, max_target_len

# # Returns all items for a given batch of pairs


# def batch2TrainData(voc, pair_batch: List[Tuple[List[str], List[str]]]):
#     """Take batch of pairs sentence and return them in a good format
#     (matrix of numbers padded)
#     ex : [("Hello how are you ?", "I'm good thanks"),
#             ("You're stupid","thanks")]
#         =>
#         inp = [[564, 45, 82, 123],
#                 [46, 123, 0, 0]]
#         length = [4, 2]
#         output = [[357,195,65],
#                     [889, 0, 0]]
#         mask = [[1,1,1],
#                 [1, 0, 0]]
#         max_target_length = 3
#     Arguments:
#         voc -- Instance of the Voc class
#         pair_batch {List[Tuple[List[str], List[str]]]} -- list of the pairs of
#                                                             dialog
#     Returns:
#         inp -- 2D Tensor with the int matrix of input sentences,
#                 organized by batch
#         lengths -- 1D Tensor with lenght of each input sentence
#         output -- 2D Tensor with the int matrix of output senteces,
#                     organized by batch
#         mask -- 2D Tensor with the bool matrix which tell where is the padding
#         max_target_len -- int to give output sentence maximal length
#     """
#     pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
#     input_batch, output_batch = [], []
#     for pair in pair_batch:
#         input_batch.append(pair[0])
#         output_batch.append(pair[1])
#     inputTensor, lengths = toInputTensor(voc, input_batch)
#     outputTensor, mask, max_target_len = toOutputTensor(voc, output_batch)
#     return inputTensor, lengths, outputTensor, mask, max_target_len
