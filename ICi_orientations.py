import numpy as np
from utils import rotation_matrix
from copy import deepcopy

class Mixin:
    
    def generate_orientations(self):

        if self.npatch == 2:
            out = self.generate_orientations_2P()
        else:
            print("NOT ALLOWED"); exit(1)

        return out

    def generate_orientations_2P(self):

        if self.default_topo:
            if self.samepatch:
                #### 1 patch-patch 2 equator-equator 3 equator-patch
                out = [[ [[1,0,0],[-1,0,0]], [[1,0,0],[-1,0,0]] ],
                       [ [[0,0,1],[0,0,-1]], [[0,0,1],[0,0,-1]] ],
                       [ [[0,0,1],[0,0,-1]], [[1,0,0],[-1,0,0]] ]]
            else:
                ### 0 pp (-+ -+) 1 pp (-+ +-) 2 pp (+- -+) 3 ee (+_- +_-) 4 ee (+_- -_+) 5 ep (+_- -+) 6 ep (+_- +-)  
                out = [[ [[1,0,0],[-1,0,0]], [[1,0,0],[-1,0,0]] ],
                       [ [[1,0,0],[-1,0,0]], [[-1,0,0],[1,0,0]] ],
                       [ [[-1,0,0],[1,0,0]], [[1,0,0],[-1,0,0]] ],
                       [ [[0,0,1],[0,0,-1]], [[0,0,1],[0,0,-1]] ],
                       [ [[0,0,1],[0,0,-1]], [[0,0,-1],[0,0,1]] ],
                       [ [[0,0,1],[0,0,-1]], [[1,0,0],[-1,0,0]] ],
                       [ [[0,0,1],[0,0,-1]], [[-1,0,0],[1,0,0]] ] ]
        else:
            
            print('non-polar topologies are not allowed'); exit(1)
            

        return out

