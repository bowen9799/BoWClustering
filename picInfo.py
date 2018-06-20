# -*- coding:UTF-8 -*-
class PicInfo:
    def __init__(self, Name, Path,Probability,NoDish):
        self.Name = Name
        self.Path = Path
        self.Probability = Probability
        self.NoDish = NoDish

    def displayName(self):
        print("Name:"+self.Name)

    def displayPath(self):
        print("Path:"+self.Path)

    def displayProbability(self):
        print("Probability:"+self.Probability)

    def displayNoDish(self):
        print("NoDish:"+self.NoDish)

    def PicInfo(self):
        print("Name:"+self.Name+"Path:"+self.Path+",Probability:"+self.Probability+",NoDish:"+self.NoDish)