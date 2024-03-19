from random import randint
from time import time

import numpy
from PIL import Image

# Preface.
# Everyone who enters here should know: There is no further God.💀💀💀💀
# Yes, I know I left 8 warnings, 3 war crimes and 2 coups here.
# However, it may help someone in understanding the basics of neural network programming and training.

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
            s = s[:p - 1]  #      This is the reason why negative integers are stored as float.
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
            else:  #                                                                   preserve the 0 before the first
                return ne__conv(int(s[:p])) + "!" + ne__conv(int(s[p + 1:][::-1]))  #  significant digit.
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


# ------------------------------------------------->Scary things start here<------------------------------------------

class Neuro():
    def __init__(self, wide, height, dpt, speed, mute):
        if wide > 0:
            self.wide = wide
        else:
            raise Exception("Invalid wide value")
        if height > 0:
            self.height = height
        else:
            raise Exception("Invalid height value")
        self.lenm = wide * height  # len array
        if dpt > 0:
            self.dpt = dpt  # defining the depth of a neural network
        else:
            raise Exception("Invalid depth value")
        self.spd = speed  # ------------------------defining a speed of backpropagation (between 0 and 1)
        self.w = []  # defining an array of weights
        self.ou = []  # defining the output array
        self.vhm = []  # defining the input array
        self.mute = mute  # mute

    # ---------------------------------------------activation function
    # ---------------------------------------------The sigmoid is used as the activation function.
    #                                              Keep in mind that in order to change the activation
    #                                              function you will need to change the error detection
    #                                              function (backpropagation algorithm) (see "bpn" function)
    @staticmethod
    def ne(s):  # activation function definition
        if s > 50:
            s = 50  # overfill protection (figure pulled out of thin air)
        if s < -50:
            s = -50
        return 1 / (1 + 2.71 ** (-2 * s))

    # ---------------------------------------------initializing weights
    def fill_m(self):
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)  # from here on "uuu" means the number of elements to process
        if not self.mute:
            print("start filling...")
            print(uuu, "elements")
        t = time()
        self.w = []
        kc = 0
        i: int
        for i in range(self.dpt):
            a1 = []
            j: int
            for j in range(self.lenm + 1):
                a2 = []
                k: int
                for k in range(self.lenm + 1):
                    __ = randint(-50, 50) / randint(100, 500)
                    if __ == 0:
                        __ += 0.01
                    a2.append(__)  # filling the weights array with random values
                    kc += 1
                a1.append(a2)
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
            self.w.append(a1)
        if not self.mute:
            print("done filling")

    # ---------------------------------------------initializing array (I just create an array of the required sizes)
    def init_m(self):
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)  # "uuu" means the number of elements to process
        if not self.mute:
            print("start init...")
            print(uuu, "elements")
        t = time()
        self.w = []
        kc = 0
        for i in range(self.dpt):
            a1 = []
            for j in range(self.lenm + 1):
                a2 = []
                for k in range(self.lenm + 1):
                    a2.append(0)
                    kc += 1
                a1.append(a2)
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
            self.w.append(a1)
        if not self.mute:
            print("done init")

    # ---------------------------------------------saving weights (Standard way to save values, use for small networks.
    #                                              Up to 2 times faster than compressed.)
    def save_m(self, fl_nm):
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)  # "uuu" means the number of elements to process
        if not self.mute:
            print("start saving...")
            print(uuu, "elements")
        kc = 0
        t = time()
        f = open(fl_nm, "w")
        f.write("text" + "\n")  # signal that normal(text) storage is being used
        i: int
        for i in range(self.dpt):
            j: int
            for j in range(self.lenm + 1):
                k: int
                for k in range(self.lenm + 1):
                    f.write(str(self.w[i][j][k]) + "\n")
                    kc += 1
            if time() - t > 60 and not self.mute:
                print(kc, "/", uuu)
                t = time()
        f.close()
        if not self.mute:
            print("done saving")

    # ---------------------------------------------loading weights
    def load_m(self, fl_nm):
        try:
            f = open(fl_nm, "r")
        except Exception:
            raise Exception("Unable to load file. The file may not exist.")

        ___ = f.readline()
        if "zip" in ___: # If you used the compressed saving method
            uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)  # "uuu" means the number of elements to process
            if not self.mute:
                print("start loading...")
                print(uuu, "elements")
            kc = 0
            t = time()
            for i in range(self.dpt):
                for j in range(self.lenm + 1):
                    for k in range(self.lenm + 1):
                        self.w[i][j][k] = ne__rconvfl(f.readline())  # outstanding move (check comments later)
                        kc += 1
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
        elif "text" in ___: # If you used the standard saving method
            uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)  # "uuu" means the number of elements to process
            if not self.mute:
                print("start loading...")
                print(uuu, "elements")
            kc = 0
            t = time()
            for i in range(self.dpt):
                for j in range(self.lenm + 1):
                    for k in range(self.lenm + 1):
                        self.w[i][j][k] = float(f.readline())
                        kc += 1
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
        else:
            f.close()
            raise Exception("Unable to load file. Unknown data type")
        f.close()
        if not self.mute:
            print("done loading")

    # ---------------------------------------------getting results
    def getv(self, ina):
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)
        if not self.mute:
            print("start...")
            print(uuu, "elements")
        kc = 0
        t = time()  # n = 0
        self.vhm = [0] * (self.lenm + 1)
        i: int
        for i in range(len(ina)):
            self.vhm[i] = float(ina[i])
        # self.vhm[-1]=randint(-1,1) #-------------You can use this as a bias neuron
        self.ou = []
        i: int
        for i in range(self.dpt):
            a1 = []
            j: int
            for j in range(self.lenm + 1):
                a1.append(0)
            self.ou.append(a1)
        i: int
        for i in range(self.lenm + 1): # For the input data array
            s = 0
            j: int
            for j in range(self.lenm + 1):
                s += self.vhm[j] * self.w[0][i][j] # Calculation of the sum from all “axons” of a neuron
                kc += 1
            self.ou[0][i] = self.ne(s) # Maintaining the state of the neuron
            if time() - t > 60 and not self.mute:
                print(kc, "/", uuu)
                t = time()
        i: int
        for i in range(1, self.dpt): # directly within the network itself
            j: int
            for j in range(self.lenm + 1):
                s = 0
                k: int
                for k in range(self.lenm + 1):
                    s += self.ou[i - 1][k] * self.w[i][j][k]
                    kc += 1
                self.ou[i][j] = self.ne(s)
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
        if not self.mute:
            print("done")
        return self.ou[-1]

    # ---------------------------------------------backpropagation ¯\_(ツ)_/¯
    #                                              (I don’t know anything other than this algorithm,
    #                                              so I’ll be glad if someone tells me the intricacies.)

    #                                              Upd: I wrote this code more than six months ago,
    #                                              2 months ago I debugged it, now I don’t remember anything,
    #                                              maybe I’ll comment on this part of the code later. In any case,
    #                                              before doing this, it’s better for you to read smart
    #                                              literature on this topic and not the nonsense of a stupid monkey (me)

    #                                              All I can say now: it definitely works, if you came here looking
    #                                              for my mistake, then with a 99% chance it’s definitely not here,
    #                                              I checked it on 3 neural network models
    def bpn(self, oua):
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1) * 2  # "uuu" means the number of elements to process
        if not self.mute:
            print("start backpropagation...")
            print(uuu, "elements")
        kc = 0
        t = time()
        ts = t
        hm = [0] * (self.lenm + 1)
        i: int
        for i in range(len(oua)):
            hm[i] = float(oua[i])
        dem = []
        dem1 = [0] * (self.lenm + 1)
        i: int
        for i in range(self.lenm + 1):
            de = self.ou[-1][i] * (1 - self.ou[-1][i]) * (hm[i] - self.ou[-1][i])
            dem1[i] = de
            kc += 1
        dem.append(dem1)
        nn = 0
        i: int
        for i in range(self.dpt - 2, -1, -1):
            dem1 = []
            j: int
            for j in range(self.lenm + 1):
                su = 0
                j1: int
                for j1 in range(self.lenm + 1):
                    su += dem[nn][j1] * self.w[i + 1][j1][j]
                    kc += 1
                de = self.ou[i][j] * (1 - self.ou[i][j]) * su
                dem1.append(de)
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
            dem.append(dem1)
            nn += 1
        i: int
        for i in range(self.dpt):
            j: int
            for j in range(self.lenm + 1):
                k: int
                for k in range(self.lenm + 1):
                    self.w[i][j][k] += dem[self.dpt - 1 - i][j] * self.spd * self.ou[i - 1][k]
                    kc += 1
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
        if not self.mute:
            print("done", time() - ts, "seconds")
            su = 0
            i: int
            for i in range(self.lenm + 1):
                su += 1 - abs(hm[i] - self.ou[-1][i])
            print("accuracy:", su / (self.lenm + 1) * su / (self.lenm + 1))

    # ---------------------------------------------converting the image and getting the result
    #                                              TODO:(Then the strength began to leave me and I did it with crutches,
    #                                              good luck with debugging (^_<)〜☆)
    def imgg(self, flnm):  # -------------------------blank for creating pictures
        ar = []
        im = Image.open(flnm).resize((self.wide, self.height))
        n = numpy.array(im)
        i: int
        for i in range(self.height):
            j: int
            for j in range(self.wide):
                su = 0
                for k in range(3):
                    su += abs(255 - n[i, j, k])
                ar.append(su / 765)
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)
        if not self.mute:
            print("start counting...")
            print(uuu, "elements")
        kc = 0
        t = time()
        self.vhm = [0] * (self.lenm + 1)
        i: int
        for i in range(len(ar)):
            self.vhm[i] = ar[i]
        # self.vhm[-1]=randint(-1,1)
        self.ou = []
        i: int
        for i in range(self.dpt):
            a1 = []
            j: int
            for j in range(self.lenm + 1):
                a1.append(0)
            self.ou.append(a1)
        i: int
        for i in range(self.lenm + 1):
            s = 0
            j: int
            for j in range(self.lenm + 1):
                s += self.vhm[j] * self.w[0][i][j]
                kc += 1
            self.ou[0][i] = self.ne(s)
            if time() - t > 60 and not self.mute:
                print(kc, "/", uuu)
                t = time()
        i: int
        for i in range(1, self.dpt):
            j: int
            for j in range(self.lenm + 1):
                s = 0
                k: int
                for k in range(self.lenm + 1):
                    s += self.ou[i - 1][k] * self.w[i][j][k]
                    kc += 1
                self.ou[i][j] = self.ne(s)
                if time() - t > 60 and not self.mute:
                    print(kc, "/", uuu)
                    t = time()
        if not self.mute:
            print("done")
        return self.ou[-1]

    # ----------------------------------------------converting variable to characters for compressed storage
    #                                               (Lossless compression is used)
    #                                               Now I am going "to pull a rabbit out of the hat"
    #                                               with recursion for faster saving in a compressed format,
    #                                               for this I need a functions before declaration Neuro().
    # ----------------------------------------------zip saving (2 times slower)
    def zip_save_m(self, fl_nm):
        f = open(fl_nm, "w")
        uuu = self.dpt * (self.lenm + 1) * (self.lenm + 1)
        if not self.mute:
            print("start saving...")
            print(uuu, "elements")
        f.write("zip" + "\n")  # signal that compressed storage is being used
        kc = 0
        t = time()
        i: int
        for i in range(self.dpt):
            j: int
            for j in range(self.lenm + 1):
                k: int
                for k in range(self.lenm + 1):
                    f.write(ne__convfl(self.w[i][j][k]) + "\n")  # outstanding move here!
                    kc += 1
            if time() - t > 60 and not self.mute:
                print(kc, "/", uuu)
                t = time()
        f.close()
        if not self.mute:
            print("done saving")
