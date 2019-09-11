import json
import numpy as np
from datetime import datetime
from settings import MIN_EVENT_SIZE


class Stats:
    def __init__(self, pathJson):
        self.ref_time = datetime.fromtimestamp(0)
        self.dict_sessions = []
        self.pathJson = pathJson

        self.count_titles = {}
        self.count_types = {}
        self.count_browser = {}
        self.count_userid = {}
        self.count_events = {}

        self.temps_sessions = np.array([])

        self.count_btn_cell = {}
        self.ttp_mousedown = []

        self.length_move = []
        self.time_move = []
        self.ttp_move = []

        self.count_cell_enter = {}
        self.ttp_enter = []

        self.time_hidden = []

        self.count_cell_focus = {}
        self.ttp_focus = []

        self.dt_wheel = []
        self.dy_wheel = []
        self.dl_wheel = []

        self.count_md = {}
        self.ttp_wheel = []

        self.count_btn_click = {}
        self.ttp_click = []

        self.count_copy = {}
        self.count_cut = {}
        self.count_paste = {}
        self.ttp_copy = []
        self.ttp_cut = []
        self.ttp_paste = []

        with open(self.pathJson, 'r') as file_json:
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

    def remove_small(self):
        for session in self.dict_sessions.copy():
            # print(len(session["events"]))
            if len(session["events"]) < MIN_EVENT_SIZE:
                self.dict_sessions.remove(session)

    def get_titles(self):
        for session in self.dict_sessions:
            self.count_titles[session["title"]] = self.count_titles.get(
                session["title"], 0) + 1

    def get_types(self):
        for session in self.dict_sessions:
            self.count_types[session["type"]] = self.count_types.get(
                session["type"], 0) + 1

    def get_browser(self):
        for session in self.dict_sessions:
            self.count_browser[session["browser"]] = self.count_browser.get(
                session["browser"], 0) + 1

    def get_userid(self):
        for session in self.dict_sessions:
            self.count_userid[session["userid"]] = self.count_userid.get(
                session["userid"], 0) + 1

    def get_events(self):
        for session in self.dict_sessions:
            for event in session["events"]:
                self.count_events[event["type"]] = self.count_events.get(
                    event["type"], 0) + 1

    def get_time_session(self):
        self.temps_sessions = np.array([])
        for session in self.dict_sessions:
            if (session["events"][0]["type"] == "load" and
                    session["events"][-1]["type"] == "unload"):
                self.temps_sessions = np.append(self.temps_sessions,
                                                session["events"][-1]["ts"])

    def get_mousedown_infos(self):
        for session in self.dict_sessions:
            ts_previous_click = 0
            for event in session["events"]:
                if event["type"] == "mousedown":
                    dict_buttons = self.count_btn_cell.get(event["cell"], {})
                    dict_buttons[event["btn"]] = dict_buttons.get(
                        event["btn"], [])
                    dict_buttons[event["btn"]].append([event["x"], event["y"]])
                    self.count_btn_cell[event["cell"]
                                        ] = dict_buttons
                    if event["ts"] - ts_previous_click < 0:
                        # print("ttp mousedown negatif !  :  ",
                            #   session["userid"], session["title"], event,
                            #   event["ts"], ts_previous_click)
                        pass
                    else:
                        self.ttp_mousedown.append(
                            event["ts"] - ts_previous_click)
                        ts_previous_click = event["ts"]

    def get_mouse_move_infos(self):
        for session in self.dict_sessions:
            ts_previous_move = 0
            for event in session["events"]:
                if event["type"] == "mousemove":
                    if event["dl"] != None and event["dt"] != None:
                        self.length_move.append(event["dl"])
                        self.time_move.append(event["dt"])

                    if event["ts"] - ts_previous_move < 0:
                        # print("ttp mousemove negatif !  :  ",
                        #       session["userid"], session["title"], event,
                        #       event["ts"], ts_previous_move)
                        pass
                    else:
                        self.ttp_move.append(event["ts"] - ts_previous_move)
                        ts_previous_move = event["ts"]

    def get_mouse_enter_infos(self):
        for session in self.dict_sessions:
            ts_previous_enter = 0
            for event in session["events"]:
                if event["type"] == "mouseenter":
                    self.count_cell_enter[event["cell"]] = self.count_cell_enter.get(
                        event["cell"], [])
                    self.count_cell_enter[event["cell"]].append(
                        [event["x"], event["y"]])

                    if event["ts"] - ts_previous_enter < 0:
                        # print("ttp mouseenter negatif !  :  ",
                        #       session["userid"], session["title"], event,
                        #       event["ts"], ts_previous_enter)
                        pass
                    else:
                        self.ttp_enter.append(event["ts"] - ts_previous_enter)
                        ts_previous_enter = event["ts"]

    def get_hidden_infos(self):
        for session in self.dict_sessions:
            hidden = False
            time_hidden = 0
            for event in session["events"]:
                if event["type"] == "hidden":
                    if hidden == False:
                        time_hidden = event["ts"]
                        hidden = True
                    else:
                        # print(session["userid"], session["title"])
                        # print("Double hidden !")
                        # raise AssertionError("Double hidden !")
                        pass
                elif event["type"] == "visible":
                    if hidden == True:
                        self.time_hidden.append(event["ts"]-time_hidden)
                        hidden = False
                    else:
                        # print(session["userid"], session["title"])
                        # print("Double visible !")
                        # raise AssertionError("Double Visible !")
                        pass
        print("\n")

    def get_focus_infos(self):
        for session in self.dict_sessions:
            ts_previous_focus = 0
            for event in session["events"]:
                if event["type"] == "focusin":
                    self.count_cell_focus[event["cell"]] = self.count_cell_focus.get(
                        event["cell"], 0) + 1

                    if event["ts"] - ts_previous_focus < 0:
                        # print("ttp focus negatif !  :  ",
                        #       session["userid"], session["title"], event,
                        #       event["ts"], ts_previous_focus)
                        pass
                    else:
                        self.ttp_focus.append(event["ts"] - ts_previous_focus)
                        ts_previous_focus = event["ts"]

    def get_wheel_infos(self):
        for session in self.dict_sessions:
            ts_previous_wheel = 0
            for event in session["events"]:
                if event["type"] == "wheel":
                    self.count_md[event["md"]] = self.count_md.get(
                        event["md"], 0) + 1
                    self.dt_wheel.append(event["dt"])
                    self.dy_wheel.append(event["dy"])
                    self.dl_wheel.append(event["dl"])

                    if event["ts"] - ts_previous_wheel < 0:
                        # print("ttp wheel negatif !  :  ",
                        #       session["userid"], session["title"], event,
                        #       event["ts"], ts_previous_wheel, event["ts"]-ts_previous_wheel)
                        pass
                    else:
                        self.ttp_wheel.append(event["ts"] - ts_previous_wheel)
                        ts_previous_wheel = event["ts"]

    def get_click_infos(self):
        for session in self.dict_sessions:
            ts_previous_click = 0
            for event in session["events"]:
                if event["type"] == "click":
                    if "cell" in event.keys():
                        # self.count_btn_click[f"Exec_{event['cell']}"] = \
                        #     self.count_btn_click.get(
                        #         f"Exec_{event['cell']}", 0)+1
                        pass

                    else:
                        self.count_btn_click[event["button"]] = \
                            self.count_btn_click.get(
                            event["button"], 0)+1

                    if event["ts"] - ts_previous_click < 0:
                        #print("ttp click negatif !  :  ", session["userid"], session["title"], event, event["ts"], event["ts"]-ts_previous_click)
                        pass
                    else:
                        self.ttp_click.append(event["ts"] - ts_previous_click)
                        ts_previous_click = event["ts"]

    def get_copy_paste_infos(self):
        for session in self.dict_sessions:
            ts_previous_copy = 0
            ts_previous_paste = 0
            ts_previous_cut = 0
            for event in session["events"]:
                if event["type"] == "copy":
                    self.count_copy[event["cell"]] = self.count_copy.get(
                        event["cell"], 0)+1
                    self.ttp_copy.append(event["ts"] - ts_previous_copy)
                    ts_previous_copy = event["ts"]

                if event["type"] == "cut":
                    self.count_cut[event["cell"]] = self.count_cut.get(
                        event["cell"], 0)+1
                    self.ttp_cut.append(event["ts"] - ts_previous_cut)
                    ts_previous_cut = event["ts"]

                if event["type"] == "paste":
                    self.count_paste[event["cell"]] = self.count_paste.get(
                        event["cell"], 0)+1
                    self.ttp_paste.append(event["ts"] - ts_previous_paste)
                    ts_previous_paste = event["ts"]

    def show_titles(self):
        print(f"{'Titres des notebooks':<40} Nombre de sessions")
        for title, num in sorted(self.count_titles.items()):
            print(f"{title:_<50} : {num}")
        print("\n\n")

    def show_types(self):
        print(f"{'Type des notebooks':<20} Nombre de sessions")
        for title, num in sorted(self.count_types.items()):
            print(f"{title:_<30} : {num}")
        print("\n\n")

    def show_browser(self):
        print(f"{'Type de navigateur':<20} Nombre de sessions")
        for title, num in sorted(self.count_browser.items()):
            print(f"{title:_<30} : {num}")
        print("\n\n")

    def show_userid(self):
        print(f"{'User_Id':<8} Nombre de sessions")
        for title, num in sorted(self.count_userid.items()):
            print(f"{title:_<10} : {num}")
        print("\n\n")

    def show_events(self):
        for e1, e2 in [("load", "unload"), ("hidden", "visible"),
                       ("focusin", "focusout"), ("mousedown", "mouseup"),
                       ("mouseenter", "mouseleave"), ("keydown", "keyup")]:
            print(f"{e1:_<20} : {self.count_events[e1]:<4} | ",
                  f"{self.count_events[e2]:<4} : {e2:_>20}")
        print("\n")
        for event in ["mousemove", "wheel", "click", "paste", "copy", "cut", ]:
            print(f"{event:_<12} : {self.count_events[event]}")
        print("\n")

    def show_time_session(self):
        print("Temps passe sur les notebooks")
        median = datetime.fromtimestamp(
            np.median(self.temps_sessions)/1000)-self.ref_time
        mean = datetime.fromtimestamp(
            np.mean(self.temps_sessions)/1000)-self.ref_time
        maxi = datetime.fromtimestamp(
            np.max(self.temps_sessions)/1000)-self.ref_time
        mini = datetime.fromtimestamp(
            np.min(self.temps_sessions)/1000)-self.ref_time
        std = datetime.fromtimestamp(
            np.std(self.temps_sessions)/1000)-self.ref_time
        somme = datetime.fromtimestamp(
            np.sum(self.temps_sessions)/1000)-self.ref_time
        print(f" {'Median :':_<10} {median}")
        print(f" {'Mean :':_<10} {mean}")
        print(f" {'Maxi :':_<10} {maxi}")
        print(f" {'Mini :':_<10} {mini}")
        print(f" {'std :':_<10} {std}")
        print(f" {'Sum :':_<10} {somme}")
        print("\n")

    def show_mousedown_infos(self):
        for title, value in sorted(self.count_btn_cell.items()):
            click_tot = 0
            moyenne_pos_tot = np.array([0.0, 0.0])
            count_btn = {}
            for button, list_pos in value.items():
                click_tot += len(list_pos)
                count_btn["g" if button == 0 else "d"] = click_tot
                moyenne_pos_tot += np.mean(np.array(list_pos), axis=0)
            print(f" Nombre de clicks cell {title:<2} : {click_tot:<4} | ",
                  f"{str(sorted(count_btn.items())):<22} | ",
                  f"x_moy, y_moy : {moyenne_pos_tot.round(2)}")
        print("\n")
        print(f" {'Temps moyen entre deux mousedown :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_mousedown)/1000) - self.ref_time)
        print(f" {'Temps median entre deux mousedown :':_<40}",
              datetime.fromtimestamp(np.median(self.ttp_mousedown)/1000) - self.ref_time)
        print(f" {'Temps max entre deux mousedown :':_<40}",
              datetime.fromtimestamp(np.max(self.ttp_mousedown)/1000) - self.ref_time)
        print(f" {'Temps min entre deux mousedown :':_<40}",
              datetime.fromtimestamp(np.min(self.ttp_mousedown) / 1000) - self.ref_time)
        print(f" {'Variance temps entre deux mousedown :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_mousedown)/1000)-self.ref_time)
        print("\n")

    def show_mouse_move_infos(self):
        l = np.array(self.length_move)
        t = (np.array(self.time_move)+1e-10)/1000
        v = np.divide(np.mean(l), np.mean(t)).round(2)
        print(f" {'distance moyenne move :':_<40}", np.mean(l).round(2))
        print(f" {'temps moyen move :':_<40}", np.mean(t).round(2))
        print(f" {'vitesse moyenne move :':_<40}", v)
        print("\n")
        print(f" {'Temps moyen entre deux move :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_move)/1000) - self.ref_time)
        print(f" {'Temps median entre deux move :':_<40}",
              datetime.fromtimestamp(np.median(self.ttp_move)/1000) - self.ref_time)
        print(f" {'Temps max entre deux move :':_<40}",
              datetime.fromtimestamp(np.max(self.ttp_move)/1000) - self.ref_time)
        print(f" {'Temps min entre deux move :':_<40}",
              datetime.fromtimestamp(np.min(self.ttp_move)/1000)-self.ref_time)
        print(f" {'Variance temps entre deux move :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_move)/1000)-self.ref_time)
        print("\n")

    def show_mouse_enter_infos(self):
        for cell, list_pos in sorted(self.count_cell_enter.items()):
            print(f"nombre enter cellule {cell:<2} : {len(list_pos):<4} | ",
                  f" x_moy y_moy  {np.mean(list_pos, axis=0).round(2)}")
        print("\n")
        print(f" {'Temps moyen entre deux enter :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_enter)/1000) - self.ref_time)
        print(f" {'Temps median entre deux enter :':_<40}",
              datetime.fromtimestamp(np.median(self.ttp_enter)/1000) - self.ref_time)
        print(f" {'Temps max entre deux enter :':_<40}",
              datetime.fromtimestamp(np.max(self.ttp_enter)/1000) - self.ref_time)
        print(f" {'Temps min entre deux enter :':_<40}",
              datetime.fromtimestamp(np.min(self.ttp_enter) / 1000) - self.ref_time)
        print(f" {'Variance temps entre deux enter :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_enter)/1000)-self.ref_time)
        print("\n")

    def show_hidden_infos(self):
        print(f" {'Temps moyen d un hidden :':_<40}",
              datetime.fromtimestamp(np.mean(self.time_hidden)/1000) - self.ref_time)
        print(f" {'Temps median d un hidden :':_<40}",
              datetime.fromtimestamp(np.median(self.time_hidden)/1000) - self.ref_time)
        print(f" {'Temps max d un hidden :':_<40}",
              datetime.fromtimestamp(np.max(self.time_hidden)/1000) - self.ref_time)
        print(f" {'Temps min d un hidden :':_<40}",
              datetime.fromtimestamp(np.min(self.time_hidden) / 1000) - self.ref_time)
        print(f" {'Variance temps d un hidden :':_<40}",
              datetime.fromtimestamp(np.std(self.time_hidden)/1000)-self.ref_time)
        print("\n")

    def show_focus_infos(self):
        for cell, value in self.count_cell_focus.items():
            print(f" Nombre de focus cellule  {cell:<2} : {value}")
        print("\n")
        print(f" {'Temps moyen entre deux focus :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_focus)/1000) - self.ref_time)
        print(f" {'Temps median entre deux focus :':_<40}",
              datetime.fromtimestamp(np.median(self.ttp_focus)/1000) - self.ref_time)
        print(f" {'Temps max entre deux focus :':_<40}",
              datetime.fromtimestamp(np.max(self.ttp_focus)/1000) - self.ref_time)
        print(f" {'Temps min entre deux focus :':_<40}",
              datetime.fromtimestamp(np.min(self.ttp_focus) / 1000) - self.ref_time)
        print(f" {'Variance temps entre deux focus :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_focus) / 1000) - self.ref_time)
        print("\n")

    def show_wheel_infos(self):
        l = np.array(self.dl_wheel)  # ou dy ? les deux marchent
        t = (np.array(self.dt_wheel)+1e-10)/1000
        v = np.divide(np.mean(l), np.mean(t)).round(2)
        print(f" {'distance moyenne wheel :':_<40}", np.mean(l).round(2))
        print(f" {'temps moyen wheel :':_<40}", np.mean(t).round(2))
        print(f" {'vitesse moyenne wheel :':_<40}", v)
        print("\n")

        for title, value in self.count_md.items():
            print(f" Nombre de wheel type {title} : {value}")
        print("\n")
        print(f" {'Temps moyen entre deux wheel :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_wheel)/1000) - self.ref_time)
        print(f" {'Temps median entre deux wheel :':_<40}",
              datetime.fromtimestamp(np.median(self.ttp_wheel)/1000) - self.ref_time)
        print(f" {'Temps max entre deux wheel :':_<40}",
              datetime.fromtimestamp(np.max(self.ttp_wheel)/1000) - self.ref_time)
        print(f" {'Temps min entre deux wheel :':_<40}",
              datetime.fromtimestamp(np.min(self.ttp_wheel) / 1000) - self.ref_time)
        print(f" {'variance temps entre deux wheel :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_wheel) / 1000) - self.ref_time)
        print("\n")

    def show_click_infos(self):
        for title, value in self.count_btn_click.items():
            print(f" Nombre de {title:_<10} : {value}")
        print("\n")
        print(f" {'Temps moyen entre deux click :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_click)/1000) - self.ref_time)
        print(f" {'Temps median entre deux click :':_<40}",
              datetime.fromtimestamp(np.median(self.ttp_click)/1000) - self.ref_time)
        print(f" {'Temps max entre deux click :':_<40}",
              datetime.fromtimestamp(np.max(self.ttp_click)/1000) - self.ref_time)
        print(f" {'Temps min entre deux click :':_<40}",
              datetime.fromtimestamp(np.min(self.ttp_click) / 1000) - self.ref_time)
        print(f" {'Variance temps entre deux click :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_click) / 1000) - self.ref_time)
        print("\n")

    def show_copy_paste_infos(self):
        for cell, nb_copy in self.count_copy.items():
            print(f" Nombre de paste cell {cell:_<10} : {nb_copy}")
        for cell, nb_cut in self.count_cut.items():
            print(f" Nombre de paste cell {cell:_<10} : {nb_cut}")
        for cell, nb_paste in self.count_paste.items():
            print(f" Nombre de paste cell {cell:_<10} : {nb_paste}")
        print("\n")
        print(f" {'Temps moyen entre deux copy :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_copy)/1000) - self.ref_time)
        print(f" {'Temps moyen entre deux cut :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_cut)/1000) - self.ref_time)
        print(f" {'Temps moyen entre deux paste :':_<40}",
              datetime.fromtimestamp(np.mean(self.ttp_paste) / 1000) - self.ref_time)
        print(f" {'variance temps entre deux paste :':_<40}",
              datetime.fromtimestamp(np.std(self.ttp_paste)/1000) - self.ref_time)
        print("\n")


stat = Stats("./Dataset/events.json")
stat.remove_mobile()
stat.remove_docs()
stat.remove_small()
stat.get_titles()
stat.get_types()
stat.get_browser()
stat.get_userid()
stat.get_events()


stat.get_time_session()
stat.get_click_infos()
stat.get_copy_paste_infos()
stat.get_focus_infos()
stat.get_hidden_infos()
stat.get_mouse_enter_infos()
stat.get_mouse_move_infos()
stat.get_mousedown_infos()
stat.get_wheel_infos()

stat.show_titles()
stat.show_types()
stat.show_browser()
stat.show_userid()
stat.show_events()

stat.show_time_session()
stat.show_click_infos()
stat.show_focus_infos()
stat.show_wheel_infos()
stat.show_hidden_infos()
stat.show_mousedown_infos()
stat.show_copy_paste_infos()
stat.show_mouse_move_infos()
stat.show_mouse_enter_infos()
