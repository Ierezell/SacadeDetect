from vocabulary import Voc
import json
import itertools
from settings import TIME_LIMIT, NB_EVENT_LIMIT, MIN_EVENT_SIZE
import math


class Donnees:
    def __init__(self, path_json):
        self.path_json = path_json
        self.dict_sessions = []
        self.dict_persons = {}
        self.vectorized_persons = []
        self.voc = Voc()
        self.time_previous_event = 0
        self.time_previous_same_event = {}

    def load_json(self):
        """
        Charge le Json dans la variable dict_sessions
        liste de dico : {pk, titre, [{eventments}]}
        """
        with open(self.path_json, 'r') as file_json:
            self.dict_sessions = json.load(file_json)
            # Load la partie events car c'est une string
            for session in self.dict_sessions:
                session["events"] = json.loads(session["events"])

    def remove_mobile(self):
        for session in self.dict_sessions.copy():
            if session["browser"] == "mobile":
                self.dict_sessions.remove(session)

    def remove_docs(self):
        for session in self.dict_sessions.copy():
            if session["type"] not in ["quiz", "exercise", "exams"]:
                self.dict_sessions.remove(session)

    def remove_small(self):
        for session in self.dict_sessions.copy():
            # print(len(session["events"]))
            if len(session["events"]) < MIN_EVENT_SIZE:
                self.dict_sessions.remove(session)

    def keep_only(self, list_type):
        for session in self.dict_sessions.copy():
            if session["type"] not in list_type:
                self.dict_sessions.remove(session)

    def create_dict_persons(self):
        """
        Cree le dictionnaire {userid1:[{session1}, {session2} ...]}
        """
        for session in self.dict_sessions:
            for s in self.split_session(session):
                userid = s["userid"]
                session_without_id = s
                session_without_id.pop("userid")
                # print(session_without_id["events"], "\n")
                self.dict_persons.setdefault(userid,  []
                                             ).append(session_without_id)

    def create_voc(self):
        assert bool(self.dict_persons), """
        Dictionary of persons must be filled before creating the vocabulary
        Use donnes.create_dict_persons before using this function! """
        self.voc.infos_to_index(self.dict_persons)

    def to_numeral(self):
        assert bool(self.dict_persons), """Dictionary of persons must be filled
        Use donnes.create_dict_persons before using this function! """
        assert bool(self.voc.event2index), """Vocablary must be filled
        Use donnes.create_voc before using this function! """
        for _, list_session in self.dict_persons.items():
            for session in list_session:
                for event in session["events"]:
                    if event["type"] == "click":
                        event["button"] = self.voc.button2index[
                            event["button"]]
                        if "cell" not in event.keys():
                            event["cell"] = 0
                    elif (event["type"] == "keydown" or
                          event["type"] == "keyup"):
                        event["key"] = self.voc.key2index[event["key"]]
                    event["type"] = self.voc.event2index[event["type"]]

    def to_vector(self):
        self.vectorized_persons = []
        for userid, list_session in self.dict_persons.items():
            for session in list_session:
                vectorized_session = []
                notebook_title = self.voc.title2index[session["title"]]
                notebook_type = self.voc.type2index[session["type"]]
                for event in session["events"]:
                    vectorized_event = self._get_vector(
                        event, notebook_title, notebook_type)
                    # print(len(vectorized_event))
                    # if vectorized_event is not None:
                    vectorized_session.append(vectorized_event)
                # print(vectorized_session)
                num_user = self.voc.user2index[userid]
                self.vectorized_persons.append((num_user, vectorized_session))

    def _get_vector(self, event, notebook_title, notebook_type):
        name_event = self.voc.index2event[event["type"]]
        # TODO garder le dico des ttp
        # TODO Ajouter un ttp de levenement davant
        # plus un ttp de levenement du meme type d'avant
        try:
            event['ts'] = math.log(event['ts'])
        except ValueError:
            # timstamp was 0 so cannot do log and will keep 0
            pass

        time_previous_same_event = event["ts"] - \
            self.time_previous_same_event.get(event["type"], event["ts"])
        self.time_previous_same_event[event["type"]] = event["ts"]

        time_previous_event = event["ts"] - self.time_previous_event
        self.time_previous_event = event["ts"]
        # print("event : ", time_previous_event)
        # print("same : ", time_previous_same_event)
        # print("selfsame : ", self.time_previous_same_event)
        # print("self : ", self.time_previous_event)
        # print()

        vector = {
            "notebook_type": [notebook_type],
            "notebook_title": [notebook_title],
            "mouse_move": [1] + [0]*6,  # *self.voc.lenght_event["mousemove"]
            "mouse_leave_enter": [1] + [1] + [0]*5,
            "mouse_up_down": [1] + [1] + [0]*6,
            "wheel": [1] + [0]*7,  # *self.voc.lenght_event["wheel"],
            "click": [1] + [0]*4,  # *self.voc.lenght_event["click"],
            "key_up_down": [1] + [1] + [0]*4,
            "focus_in_out": [1] + [1] + [0]*3,
            "hidden_visible": [1] + [1] + [0]*2,
            "load_unload": [1] + [1] + [0]*2,  # *self.voc.lenght_event["load"]
            "copy_cut_paste": [1] + [1] + [1] + [0] * 3,
            "time_previous_event": [0],
            "time_previous_same_event": [0],
        }
        masks = {
            "mousemove": [0],
            "mouseenter": [0, 1],
            "mouseleave": [1, 0],
            "mousedown": [0, 1],
            "mouseup": [1, 0],
            "wheel": [0],
            "click": [0],
            "keyup": [0, 1],
            "keydown": [1, 0],
            "focusin": [0, 1],
            "focusout": [1, 0],
            "hidden": [0, 1],
            "visible": [1, 0],
            "load": [0, 1],
            "unload": [1, 0],
            "copy": [0, 1, 1],
            "paste": [1, 0, 1],
            "cut": [1, 1, 0],
        }
        event.pop("i", None)

        # TIME PREVIOUS EVENT
        vector["time_previous_event"] = [time_previous_event]
        # TIME PREVIOUS SAME EVENT
        vector["time_previous_same_event"] = [time_previous_same_event]
        # MOUSEMOVE
        if name_event == "mousemove":
            vector["mouse_move"] = masks[name_event] + list(event.values())

        # MOUSELEAVE / ENTER
        if name_event == "mouseleave" or name_event == "mouseenter":
            vector["mouse_leave_enter"] = masks[name_event] + \
                list(event.values())

        # MOUSEUP / DOWN
        if name_event == "mouseup" or name_event == "mousedown":
            vector["mouse_up_down"] = masks[name_event] + list(event.values())

        # WHEEL
        if name_event == "wheel":
            vector["wheel"] = masks[name_event] + list(event.values())

        # CLICK
        if name_event == "click":
            vector["click"] = masks[name_event] + list(event.values())

        # KEYUP /DOWN
        if name_event == "keyup" or name_event == "keydown":
            vector["key_up_down"] = masks[name_event] + list(event.values())

        # FOCUSOUT / IN
        if name_event == "focusout" or name_event == "focusin":
            vector["focus_in_out"] = masks[name_event] + list(event.values())

        # HIDDEN / VISIBLE
        if name_event == "hidden" or name_event == "visible":
            vector["hidden_visible"] = masks[name_event] + list(event.values())

        # LOAD / UNLOAD
        if name_event == "load" or name_event == "unload":
            vector["load_unload"] = masks[name_event] + list(event.values())

        # PASTE / COPY / CUT
        if (name_event == "paste" or name_event == "copy" or
                name_event == "cut"):
            vector["copy_cut_paste"] = masks[name_event] + list(event.values())

        vec = list(itertools.chain.from_iterable(vector.values()))
        return vec

    def split_hidden(self, session):
        splitted_session = []
        header = session.copy()
        evenements = header.pop("events")
        isHidden = False
        previousVisible = 0
        indexVisible = 0
        for i, e in enumerate(evenements):
            if e["type"] == "hidden":
                isHidden = True
            if e["type"] == "visible" and isHidden:
                isHidden = False
                indexVisible = i
                for event in evenements[previousVisible:indexVisible]:
                    if event["type"] == "keydown":
                        cutted_session = header.copy()
                        cutted_session["events"] = evenements[previousVisible:
                                                              indexVisible]
                        splitted_session.append(cutted_session)
                        break
                previousVisible = i

        for event in evenements[indexVisible:]:
            if event["type"] == "keydown":
                cutted_session = header.copy()
                cutted_session["events"] = evenements[indexVisible:]
                splitted_session.append(cutted_session)
                break
        # print("Session : ", session)
        # for kak in splitted_session:
            # print("Splitted  : ",  kak)
        return splitted_session
        # return [session]

    # def split_hidden(self, session):
    #     splitted_session = []
    #     header = session.copy()
    #     evenements = header.pop("events")
    #     isHidden = False
    #     previousVisible = 0
    #     indexVisible = 0
    #     for i, e in enumerate(evenements):
    #         if e["type"] == "hidden":
    #             isHidden = True
    #         if e["type"] == "visible" and isHidden:
    #             isHidden = False
    #             indexVisible = i
    #             if len(evenements[previousVisible:indexVisible]) > NB_EVENT_LIMIT:
    #                 cutted_session = header.copy()
    #                 cutted_session["events"] = evenements[previousVisible:
    #                                                       indexVisible]
    #                 splitted_session.append(cutted_session)
    #             previousVisible = i

    #     if len(evenements[indexVisible:]) > NB_EVENT_LIMIT:
    #         cutted_session = header.copy()
    #         cutted_session["events"] = evenements[indexVisible:]
    #         splitted_session.append(cutted_session)
    #     # print("Session : ", session)
    #     # for kak in splitted_session:
    #         # print("Splitted  : ",  kak)
    #     return splitted_session

    def split_session(self, session, time_limit=TIME_LIMIT):
        # liste de disctionaires
        # [ {pk, type, title,user,browser,created,modified, events : [{},{}]} ]
        splitted_hiden = self.split_hidden(session)
        # for k in splitted_hiden:
        # print(k, "\n\n\n")
        splitted_session = []
        for sess in splitted_hiden:
            header = sess.copy()
            evenements = header.pop("events")
            previousTime = evenements[0]['ts']
            previousCut = 0
            for i, e in enumerate(evenements):
                timeEvent = e["ts"]
                if timeEvent - previousTime > time_limit:
                    # print("cut  ", i)
                    cutted_session = header.copy()
                    cutted_session["events"] = evenements[previousCut: i]
                    if len(evenements[previousCut:i]) > NB_EVENT_LIMIT:
                        splitted_session.append(cutted_session)
                    previousCut = i
                previousTime = timeEvent

            cutted_session = header.copy()
            cutted_session["events"] = evenements[previousCut:]
            if len(evenements[previousCut:]) > NB_EVENT_LIMIT:
                splitted_session.append(cutted_session)
        # for k in splitted_session:
        #     print(len(k['events']))

        return splitted_session
        # return [session]
