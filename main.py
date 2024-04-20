from random import randint
from time import time

import numpy
from PIL import Image


# Preface.
# Everyone who enters here should know: There is no further God.💀💀💀💀
# Yes, I know I left 8 warnings, 3 war crimes and 2 coups here.
# However, it may help someone in understanding the basics of neural network programming and training.
# UPD1.0: After refactoring I can say that overall I like what I wrote. However, I don't think this
# code should be an example of how to do it, not everything here is done in the most optimal way I'll
# leave this for version 2.0 where i plan to add parallel computing (not the best thing for Python, but it should help).

def ne__conv(a):
    if a == 0:
        return ""
    return ne__conv(a // 93) + chr(a % 93 + 34)  # here I use 93-digit number system to convert it to ASCII characters,
    #                                              which reduces the number of characters used to store


def ne__convfl(a):
    s = str(abs(a))
    if "." in s:
        if "e" in s:
            p = s.find("-")
            t = int(s[p + 1:])  # I catch the scientific type of notation float and designate negative numbers as “!!”.
            s = s[:p - 1]  # This is the reason why negative integers are stored as float.
            s = s.replace(".", "")
            s = s[::-1] + "0" * (t - 1)
            if a < 0:
                return ne__conv(0) + "!!" + ne__conv(int(s))
            else:
                return ne__conv(0) + "!" + ne__conv(int(s))
        else:
            p = s.find(".")
            if a < 0:
                return ne__conv(int(s[:p])) + "!!" + ne__conv(int(s[p + 1:][::-1]))  # I expand the fractional part to
            else:  # preserve the 0 before the first
                return ne__conv(int(s[:p])) + "!" + ne__conv(int(s[p + 1:][::-1]))  # significant digit.
    else:
        if a < 0:
            return ne__conv(abs(a)) + "!!"
        else:
            return ne__conv(abs(a))


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

    @staticmethod
    def activation_func(s):  # activation function definition
        if s > 70:
            s = 70  # overfill protection (figure pulled out of thin air)
        if s < -70:
            s = -70
        return 1 / (1 + 2.71 ** (-2 * s))

    def fill(self, mute):
        if not mute:
            print("Start filling...")
        kc = 0
        t = time()
        for i in range(len(self.width) - 1):
            for j in range(self.width[i + 1]):
                for k in range(self.width[i]):
                    __ = randint(-50, 50) / randint(100, 500)
                    if __ == 0:
                        __ += 0.01
                    kc += 1
                    self.w[i][j][k] = __  # filling the weights array with random values
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
        if "zip" in ___:  # If you used the compressed saving method
            for i in range(len(self.width) - 1):
                for j in range(self.width[i + 1]):
                    for k in range(self.width[i]):
                        self.w[i][j][k] = ne__rconvfl(f.readline())  # outstanding move (check comments later)
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
    #                                               in general.
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
    def backpropagation(self, output_a, mute):
        if not mute:
            print("Start backpropagation...")
        t = time()
        kc = 0
        count = self.elements_to_count * 2
        if len(output_a) != self.width[-1]:
            raise Exception("Incorrect output array width")
        correction_array = []
        correction_array1 = []
        for i in range(len(output_a)):
            correction_array1.append(self.ou[-1][i] * (1 - self.ou[-1][i]) * (output_a[i] - self.ou[-1][i]))
            # getting the first layer of correction values. Starts from the output.
            kc += 1
        correction_array.append(correction_array1)
        _i = 0
        for i in range(len(self.width) - 2, -1, -1):
            correction_array1 = []
            for j in range(self.width[i]):
                su = 0
                for k in range(self.width[i + 1]):
                    su += correction_array[_i][k] * self.w[i][k][j]  # getting correction values for inner layers.
                    kc += 1
                de = self.ou[i][j] * (1 - self.ou[i][j]) * su
                correction_array1.append(de)
                if (not mute) and (time() - t >= 60):
                    print(kc, "/", count)
                    t = time()
            correction_array.append(correction_array1)
            _i += 1
        _ = len(self.width)
        for i in range(_ - 1):
            for j in range(self.width[i + 1]):
                for k in range(self.width[i]):
                    self.w[i][j][k] += correction_array[_ - 2 - i][j] * self.speed * self.ou[i][
                        k]  # adjusting values of weights
                    kc += 1
                if (not mute) and (time() - t >= 60):
                    print(kc, "/", count)
                    t = time()
        if not mute:
            su = 0
            for i in range(len(output_a)):
                su += 1 - abs(output_a[i] - self.ou[-1][i])
            print("accuracy:", (su / len(output_a)) ** 2)
            print("Done backpropagation")

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

    # ----------------------------------------------converting variable to characters for compressed storage
    #                                               (Lossless compression is used)
    #                                               Now I am going "to pull a rabbit out of the hat"
    #                                               with recursion for faster saving in a compressed format,
    #                                               for this I need a functions before declaration Neuro().
    # ----------------------------------------------zip saving (2 times slower)
    def zip_save(self, fl_nm, mute):
        if not mute:
            print("Start saving...")
        kc = 0
        t = time()
        f = open(fl_nm, "w")
        f.write("zip" + "\n")  # signal that compressed storage is being used
        for i in range(len(self.width) - 1):
            for j in range(self.width[i + 1]):
                for k in range(self.width[i]):
                    f.write(ne__convfl(self.w[i][j][k]) + "\n")  # outstanding move here!
                    kc += 1
                if (not mute) and (time() - t >= 60):
                    t = time()
                    print(kc, "/", self.elements_to_count)
        f.close()
        if not mute:
            print("Done saving")
