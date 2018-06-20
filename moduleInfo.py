# -*- coding:UTF-8 -*-
class ModuleInfo:
    def __init__(self, ID, Coarse_Grained_ModulePath,Fine_Grained_ModulePath,Clustering_ModulePath):
        self.ID = ID
        self.Coarse_Grained_ModulePath = Coarse_Grained_ModulePath
        self.Fine_Grained_ModulePath = Fine_Grained_ModulePath
        self.Clustering_ModulePath = Clustering_ModulePath

    def displayID(self):
        print("ID:"+self.ID)

    def displayCoarseModulePath(self):
        print("Coarse_Grained_ModulePath:"+self.Coarse_Grained_ModulePath)

    def displayFineGrainedModulePath(self):
        print("Fine_Grained_ModulePath:"+self.Fine_Grained_ModulePath)

    def displayClusteringModulePath(self):
        print("Clustering_ModulePath:"+self.Clustering_ModulePath)

    def ModuleInfo(self):
        print("ID:"+self.ID+"Coarse_Grained_ModulePath:"+self.Coarse_Grained_ModulePath+",Fine_Grained_ModulePath:"+self.Fine_Grained_ModulePath+",Clustering_ModulePath:"+self.Clustering_ModulePath)