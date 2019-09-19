# ########
# TOKENS #
# ########
PAD_token = 0  # Used for padding short sequence
SOS_token = 1  # Start-of-sensequencetence token
EOS_token = 2  # End-of-sequence token


class Voc:
    def __init__(self):
        self.user2index = {}
        self.index2user = {}
        self.num_user = 0

        self.event2index = {}
        self.index2event = {}
        self.count_events = {}
        self.num_event = 0

        self.type2index = {}
        self.index2type = {}
        self.count_type = {}
        self.num_type = 0

        self.title2index = {}
        self.index2title = {}
        self.count_title = {}
        self.num_title = 0

        self.key2index = {}
        self.index2key = {}
        self.count_keys = {}
        self.num_keys = 0

        self.button2index = {}
        self.index2button = {}
        self.count_buttons = {}
        self.num_buttons = 0

        self.lenght_event = {}

        self.vector = {}
        self.masks = {}

    def infos_to_index(self, dict_persons):
        for userid, list_session in dict_persons.items():
            if userid not in self.user2index:
                self.user2index[userid] = self.num_user
                self.index2user[self.num_user] = userid
                self.num_user += 1
            else:
                self.count_type[session["type"]] += 1
            for session in list_session:
                if session["type"] not in self.type2index:
                    self.type2index[session["type"]] = self.num_type
                    self.count_type[session["type"]] = 1
                    self.index2type[self.num_type] = session["type"]
                    self.num_type += 1
                else:
                    self.count_type[session["type"]] += 1

                if session["title"] not in self.title2index:
                    self.title2index[session["title"]] = self.num_title
                    self.count_title[session["title"]] = 1
                    self.index2title[self.num_title] = session["title"]
                    self.num_title += 1
                else:
                    self.count_title[session["title"]] += 1

                for event in session["events"]:
                    if event["type"] not in self.event2index:
                        self.event2index[event["type"]] = self.num_event
                        self.count_events[event["type"]] = 1
                        self.index2event[self.num_event] = event["type"]
                        self.num_event += 1
                        self.lenght_event[event["type"]] = len(event.values())
                    else:
                        self.count_events[event["type"]] += 1

                    if event["type"] == "keydown" or event["type"] == "keyup":
                        try:
                            k = event["key"]
                        except KeyError:
                            # print(event, session["pk"], userid)
                            event["key"] = "UKWN"
                        if event["key"] not in self.key2index:
                            self.key2index[event["key"]] = self.num_keys
                            self.count_keys[event["key"]] = 1
                            self.index2key[self.num_keys] = event["key"]
                            self.num_keys += 1
                        else:
                            self.count_keys[event["key"]] += 1
                    if event["type"] == "click":
                        if event["button"] not in self.button2index:
                            self.button2index[event["button"]
                                              ] = self.num_buttons
                            self.count_buttons[event["button"]] = 1
                            self.index2button[self.num_buttons
                                              ] = event["button"]
                            self.num_buttons += 1
                        else:
                            self.count_buttons[event["button"]] += 1
