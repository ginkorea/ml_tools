import stadium
import reinforcement as re


class Lab:

    def __init__(self):
        self.q = stadium.World("q")
        self.sarsa = stadium.World("sarsa")
        self.open = True
        self.experiment = None
        while self.open:
            self.get_user_input()

    def get_user_input(self):
        try:
            selection = int(input("select an track: int(0 - 3)"))
            mode = input("select q or sarsa: str()")
            iterations = int(input("select the number of iterations"))
        except TypeError:
            pass
        if selection == 3:
            self.open = False
        else:
            if mode == "q":
                world = self.q
            else:
                world = self.sarsa
            track = world.tracks[selection]
            Experiment(track, mode, iterations)


class Experiment:

    def __init__(self, track, mode, total_iterations):
        self.track = track
        self.speeds = []
        self.filename = self.track.name + "-" + mode + "-learning-" + str(total_iterations) + ".csv"
        i = 0
        while i < total_iterations:
            track.car.seek_starting_line()
            window, canvas, racer = track.render()
            track.car.drive_course(window, canvas)
            # self.speeds.append(speed)
            i += 1
            track.car.finished = False
            window.destroy()
        track.table.df.to_csv(self.filename)







