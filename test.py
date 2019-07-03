from data import Donnees
from vocabulary import Voc

donnes = Donnees("./Dataset/events.json")
donnes.load_json()
donnes.remove_mobile()
donnes.create_dict_persons()
donnes.create_voc()
donnes.to_numeral()
donnes.to_vector()

# for uid, data in donnes.vectorized_persons[0:2]:
#     for vector in data:
#         print(len(vector))
# print(donnes.vectorized_persons["joi0dO"][0])
# users = [
#     "joi0dO", "4K6Pdc", "F2HTo1", "viexVP", "7yNCFc", "oYfKAn", "PYDaQc",
#     "u4smTD", "QUHXF6", "3lpz49", "tuFwoS", "2WWuJx", "vgaxU4", "fwvBVd",
# ]
# print()
# print()
# print()
# print()
# for u in users:
#     if len(donnes.vectorized_persons[u][0]) < 20:
#         print(u)
#         print()
#         print(donnes.vectorized_persons[u][0])
#         print()
# printprint(donnes.vectorized_persons["joi0dO"][0])(
#     donnes.show_vectorized_persons())

# print("num event ", donnes.voc.num_event)
# print("\n")
# print("event 2 index ", donnes.voc.event2index)
# print("\n")
# print("index 2 event", donnes.voc.index2event)
# print("\n")
# print("count events ", donnes.voc.count_events)
# print("\n")

# print("num type ", donnes.voc.num_type)
# print("\n")
# print("types 2 index ", donnes.voc.type2index)
# print("\n")
# print("index 2 types", donnes.voc.index2type)
# print("\n")
# print("count types ", donnes.voc.count_type)
# print("\n")

# print("num title ", donnes.voc.num_title)
# print("\n")
# print("titles 2 index ", donnes.voc.title2index)
# print("\n")
# print("index 2 titles", donnes.voc.index2title)
# print("\n")
# print("count titles ", donnes.voc.count_title)
# print("\n")

# print(donnes.voc.key2index)s
