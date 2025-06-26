from random import uniform
from time import time

import numpy
from PIL import Image
import math


# Preface.
# Everyone who enters here should know: There is no further God.💀💀💀💀
# Yes, I know I left 8 warnings, 3 war crimes and 2 coups here.
# However, it may help someone in understanding the basics of neural network programming and training.
# UPD1.0: After refactoring I can say that overall I like what I wrote. However, I don't think this
# code should be an example of how to do it, not everything here is done in the most optimal way I'll
# leave this for version 2.0 where i plan to add parallel computing (not the best thing for Python, but it should help).
# UPD3.0: So, I don't think this code really needs parallel computing, because it's just an educational library and
# (mostly) I'm lazy as shit. Now we have LeakyReLu, L1 and L2 regularization, more stable, more efficient, random value
# normalization. Removed: zip saving, reason: ugly as shit and compression ratio less 30% (too low).


def ne__rconv(s):
    if not s:
        return 0
    return ne__rconv(s[:-1]) * 93 + (ord(s[-1]) - 34)


def ne__rconvfl(s):
    if "!!" in s:
        p = s.find("!!")
        return -float(str(ne__rconv(s[:p])) + "." + str(ne__rconv(s[p + 2:]))[::-1])
    elif "!" in s:
        p = s.find("!")
        return float(str(ne__rconv(s[:p])) + "." + str(ne__rconv(s[p + 1:]))[::-1])
    else:
        return ne__rconv(s)

class Neuro:
    def __init__(self, width, speed):
        self.w = []
        self.ou = []
        su = 0
        for _ in range(len(width)):
            if width[_] <= 0:
                raise Exception("Invalid width value")
            self.ou.append([0] * width[_])
        for _ in range(1, len(width)):
            self.w.append([])
            for __ in range(width[_]):
                su += width[_ - 1]
                self.w[_ - 1].append([0] * width[_ - 1])
        if speed <= 0 or speed > 1:
            raise Exception("invalid speed value")
        self.speed = speed
        self.width = width
        self.elements_to_count = su
        self.correction_array = []
        self.accuracy = 0

    @staticmethod
    def activation_func(s):  # activation function definition Leaky ReLu
        if s >= 0:
            return s
        return s*0.09

    def fill(self, mute):
        if not mute:
            print("Start filling...")
        kc = 0
        t = time()
        for i in range(len(self.width) - 1):
            for j in range(self.width[i + 1]):
                for k in range(self.width[i]):
                    __ = uniform(-math.sqrt(6/(self.width[i]*1.0081)), math.sqrt(6/(self.width[i]*1.0081)))
                    if __ == 0:
                        __ += 0.001
                    kc += 1
                    self.w[i][j][k] = __  # filling the weights array with random values (UPD3: now i use normalization)
                if (not mute) and (time() - t >= 60):
                    print(kc, "/", self.elements_to_count)
                    t = time()
        if not mute:
            print("Done filling")

    def save(self, fl_nm, mute):
        if not mute:
            print("Start saving...")
        kc = 0
        t = time()
        f = open(fl_nm, "w")
        f.write("text" + "\n")  # signal that normal(text) storage is being used
        for i in range(len(self.width) - 1):
            for j in range(self.width[i + 1]):
                for k in range(self.width[i]):
                    f.write(str(self.w[i][j][k]) + "\n")
                    kc += 1
                if (not mute) and (time() - t >= 60):
                    print(kc, "/", self.elements_to_count)
                    t = time()
        f.close()
        if not mute:
            print("Done saving")

    def load(self, fl_nm, mute):
        if not mute:
            print("Start loading...")
        kc = 0
        t = time()
        try:
            f = open(fl_nm, "r")
        except Exception:
            raise Exception("Unable to load file. The file may not exist.")
        ___ = f.readline()
        if "zip" in ___:  # If you used the compressed saving method (deprecated)
            for i in range(len(self.width) - 1):
                for j in range(self.width[i + 1]):
                    for k in range(self.width[i]):
                        self.w[i][j][k] = ne__rconvfl(f.readline())
                        kc += 1
                    if (not mute) and (time() - t >= 60):
                        print(kc, "/", self.elements_to_count)
                        t = time()
        elif "text" in ___:  # If you used the standard saving method
            for i in range(len(self.width) - 1):
                for j in range(self.width[i + 1]):
                    for k in range(self.width[i]):
                        self.w[i][j][k] = float(f.readline())
                        kc += 1
                    if (not mute) and (time() - t >= 60):
                        print(kc, "/", self.elements_to_count)
                        t = time()
        else:
            f.close()
            raise Exception("Unable to load file. Unknown data type")
        f.close()
        if not mute:
            print("Done loading")

    # -----------------------------------------------In the previous version of the code (which did not work)
    #                                               I classified neurons into internal and axons. Now I got
    #                                               rid of this for brevity, readability and common sense
    #                                               in general.(UPD1)
    def get_result(self, inp, mute):
        if not mute:
            print("Start counting...")
        t = time()
        kc = 0
        count = self.elements_to_count + self.width[0]
        if len(inp) != self.width[0]:
            raise Exception("Incorrect input array width")
        for _ in range(len(inp)):
            try:
                self.ou[0][_] = float(inp[_])
                kc += 1
            except Exception:
                raise Exception("Incorrect data type of input array. It must be float (0.0 to 1.0)")
        if (not mute) and (time() - t >= 60):
            print(kc, "/", count)
            t = time()
        for i in range(len(self.width) - 1):
            for j in range(self.width[i + 1]):
                su = 0
                for k in range(self.width[i]):
                    su += self.ou[i][k] * self.w[i][j][k]
                    kc += 1
                self.ou[i + 1][j] = self.activation_func(su)
                if (not mute) and (time() - t >= 60):
                    print(kc, "/", count)
                    t = time()
        if not mute:
            print("Done counting...")
        return self.ou[-1]

    # ---------------------------------------------backpropagation ¯\_(ツ)_/¯
    #                                              (I don’t know anything other than this algorithm,
    #                                              so I’ll be glad if someone tells me the intricacies.)

    #                                              Upd1: I wrote this code more than six months ago,
    #                                              2 months ago I debugged it, now I don’t remember anything,
    #                                              (UPD2: I remembered while refactoring code, but it's impossible 
    #                                              to explain it in code)
    #                                              maybe I’ll comment on this part of the code later. In any case,
    #                                              before doing this, it’s better for you to read smart
    #                                              literature on this topic and not the nonsense of a stupid monkey (me)

    #                                              All I can say now: it definitely works (UPD2: LIE), if you came here
    #                                              looking for my mistake, then with a 99% chance it’s definitely not
    #                                              here, I checked it on 3 neural network models

    #                                              UPD2: Well, last time there REALLY was a mistake. I fixed it, but now
    #                                              I'm not sure of anything anymore.

    #                                              UPD3: So, another day, another fix, but now, maybe, it works.
    def backpropagation(self, output_a, mute, L1=False, L2=False):
        if not mute:
            print("Start backpropagation...")
        t = time()
        kc = 0
        count = self.elements_to_count * 2
        if len(output_a) != self.width[-1]:
            raise Exception("Incorrect output array width")
        if L1 and L2:
            raise Exception("Both L1 and L2 regularization can not be used.")
        self.correction_array = []
        correction_array1 = []
        for i in range(len(output_a)):
            if self.ou[-1][i] < 0:
                correction_array1.append(-(self.ou[-1][i] - output_a[i]) * 0.09)
            else:
                correction_array1.append(-(self.ou[-1][i]-output_a[i]))
            # getting the first layer of correction values. Starts from the output.
            kc += 1
        self.correction_array.append(correction_array1)
        _i = 0
        for i in range(len(self.width) - 2, -1, -1):
            correction_array1 = []
            for j in range(self.width[i]):
                su = 0
                for k in range(self.width[i + 1]):
                    if L1:
                        su += self.correction_array[_i][k] * self.w[i][k][j]*abs(self.w[i][k][j])
                        # getting correction values for inner layers with L1 regularization.
                    elif L2:
                        su += self.correction_array[_i][k] * self.w[i][k][j] * self.w[i][k][j] * self.w[i][k][j]
                        # getting correction values for inner layers with L2 regularization.
                    else:
                        su += self.correction_array[_i][k] * self.w[i][k][j]
                        # getting correction values for inner layers without regularization.
                    kc += 1
                if self.ou[i][j] < 0:
                    de = su*0.09 #Leaky ReLu derivative with output <0
                else:
                    de = su #Leaky ReLu derivative with output >0
                correction_array1.append(de)
                if (not mute) and (time() - t >= 60):
                    print(kc, "/", count)
                    t = time()
            self.correction_array.append(correction_array1)
            _i += 1
        for i in range(len(self.width) - 1):
            for j in range(self.width[i+1]):
                for k in range(self.width[i]):
                    _ = self.correction_array[len(self.width)-2-i][j]
                    self.w[i][j][k] += _ * self.speed * self.ou[i][k]
        su = 0
        for i in range(len(output_a)):
            su += 1 - abs(output_a[i] - self.ou[-1][i]) / max(output_a[i], self.ou[-1][i],
                                                              abs(output_a[i] - self.ou[-1][i]), 1)
        self.accuracy = (su / len(output_a)) ** 2
        if not mute:
            print("accuracy:", self.accuracy)
            print("Done backpropagation")
            if numpy.isnan(su):
                raise Exception("NaN in out array. Maybe coefficients blows up, try using L1 or L2 regularization.")
                #Sometimes this may mean you need a compression layer.

    # ---------------------------------------------converting the image and getting the result
    #                                              TODO:(Then the strength began to leave me and I did it with crutches,
    #                                              good luck with debugging (^_<)〜☆)
    def image_get_result(self, flnm, height, width, mute):  # -------------------------blank for reading pictures
        if not mute:
            print("Start image converting...")
        t = time()
        kc = 0
        count = width * height * 3
        inp = []
        im = Image.open(flnm).resize((width, height))
        n = numpy.array(im)
        for i in range(width):
            for j in range(height):
                su = 0
                for k in range(3):
                    su += abs(255 - n[i, j, k])  # converting an image to negative black and white
                    kc += 1
                inp.append(su / 765)
            if (not mute) and (time() - t >= 60):
                print(kc, "/", count)
                t = time()
        if not mute:
            print("Done image converting")
        return self.get_result(inp, mute)
