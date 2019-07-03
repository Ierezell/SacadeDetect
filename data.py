from vocabulary import Voc
import json
import itertools


class Donnees:
    def __init__(self, path_json):
        self.path_json = path_json
        self.dict_sessions = []
        self.dict_persons = {}
        self.vectorized_persons = []
        self.voc = Voc()
        self.time_previous_event = {}

    def load_json(self):
        with open(self.path_json, 'r') as file_json:
            self.dict_sessions = json.load(file_json)
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

    def keep_only(self, list_type):
        for session in self.dict_sessions.copy():
            if session["type"] not in list_type:
                self.dict_sessions.remove(session)

    def create_dict_persons(self):
        for session in self.dict_sessions:
            session_without_id = session.copy()
            session_without_id.pop("userid")
            self.dict_persons.setdefault(session["userid"], []
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
                    if vectorized_event is not None:
                        vectorized_session.append(vectorized_event)
                # print(vectorized_session)
                num_user = self.voc.user2index[userid]
                self.vectorized_persons.append((num_user, vectorized_session))

    def _get_vector(self, event, notebook_title, notebook_type):
        name_event = self.voc.index2event[event["type"]]
        # time_to_previous_event = event["ts"] - \
        # self.time_previous_event.get(event["type"], event["ts"])
        # vector = {
        #     "notebook_type": [notebook_type],
        #     "notebook_title": [notebook_title],
        #     "mouse_move": [1] + [0]*self.voc.lenght_event["mousemove"],
        #     "mouse_leave_enter":
        #     [1] + [1] + [0]*self.voc.lenght_event["mouseleave"],
        #     "mouse_up_down": [1] + [1] + [0]*self.voc.lenght_event["mouseup"],
        #     "wheel": [1] + [0]*self.voc.lenght_event["wheel"],
        #     "click": [1] + [0]*self.voc.lenght_event["click"],
        #     "key_up_down": [1] + [1] + [0]*self.voc.lenght_event["keyup"],
        #     "focus_in_out": [1] + [1] + [0]*self.voc.lenght_event["focusin"],
        #     "hidden_visible": [1] + [1] + [0]*self.voc.lenght_event["hidden"],
        #     "load_unload": [1] + [1] + [0]*self.voc.lenght_event["load"],
        #     "copy_cut_paste": [1] + [1] + [1] +
        #     [0]*self.voc.lenght_event["copy"],
        # }

        vector = {
            "notebook_type": [notebook_type],
            "notebook_title": [notebook_title],
            "mouse_move": [1] + [0]*6,
            "mouse_leave_enter": [1] + [1] + [0]*5,
            "mouse_up_down": [1] + [1] + [0]*6,
            "wheel": [1] + [0]*7,
            "click": [1] + [0]*4,
            "key_up_down": [1] + [1] + [0]*4,
            "focus_in_out": [1] + [1] + [0]*3,
            "hidden_visible": [1] + [1] + [0]*2,
            "load_unload": [1] + [1] + [0]*2,
            "copy_cut_paste": [1] + [1] + [1] + [0]*3,
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
        # print("AAAAAA", len(list(itertools.chain.from_iterable(vector.values()))))
        event.pop("i", None)
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
