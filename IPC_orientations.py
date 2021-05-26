import numpy as np
from utils import rotation_matrix
from copy import deepcopy

class Mixin:
    
    def generate_orientations(self):

        if self.npatch == 1:
            out = self.generate_orientations_1P()
        elif self.npatch == 2:
            out = self.generate_orientations_2P()
        elif self.npatch == 3:
            out = self.generate_orientations_3P()
        else:
            print("NOT ALLOWED"); exit(1)

        return out

    def generate_orientations_1P(self):
        
        out = [[ [[1,0,0]], [[-1,0,0]] ], [ [[0,0,1]], [[-1,0,0]] ], 
               [ [[-1,0,0]], [[1,0,0]] ], [ [[0,0,1]], [[0,0,1]] ],
               [ [[1,0,0]], [[1,0,0]] ],  [ [[0,0,1]], [[1,0,0]] ] ]

        return out

    def generate_orientations_2P(self):

        if self.default_topo:
            if self.samepatch:
                #### 1 patch-patch 2 equator-equator 3 equator-patch
                out = [[ [[1,0,0],[-1,0,0]], [[1,0,0],[-1,0,0]] ],
                       [ [[0,0,1],[0,0,-1]], [[0,0,1],[0,0,-1]] ],
                       [ [[0,0,1],[0,0,-1]], [[1,0,0],[-1,0,0]] ]]
            else:
                out = [[ [[1,0,0],[-1,0,0]], [[1,0,0],[-1,0,0]] ],
                       [ [[1,0,0],[-1,0,0]], [[-1,0,0],[1,0,0]] ],
                       [ [[-1,0,0],[1,0,0]], [[1,0,0],[-1,0,0]] ],
                       [ [[0,0,1],[0,0,-1]], [[0,0,1],[0,0,-1]] ],
                       [ [[0,0,1],[0,0,-1]], [[1,0,0],[-1,0,0]] ],
                       [ [[0,0,1],[0,0,-1]], [[-1,0,0],[1,0,0]] ] ]
        else:
            
            out = []
            ### calculate the sum and the differece of the patches vectors
            ps = self.topo[1]+self.topo[0]

            ##patch-patch rotations (p1 vs p1)
            v1, v2, _mat1 = rotation_matrix(self.topo[0], [1,0,0])
            v1, v2, _mat2 = rotation_matrix(self.topo[0], [-1,0,0])

            out.append([[np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])],
                        [np.dot(_mat2,self.topo[0]), np.dot(_mat2,self.topo[1])]]) 

            ##equator-equator 
            v1, v2, _mat1 = rotation_matrix(ps, [1,0,0])
            v1, v2, _mat2 = rotation_matrix(ps, [-1,0,0])
            
            temp1 = [np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])]
            temp2 = [np.dot(_mat2,self.topo[0]), np.dot(_mat2,self.topo[1])]
            
            pd1 = temp1[1]-temp1[0]; pd2 = temp2[1]-temp2[0]
            v1, v2, _mat31 = rotation_matrix(pd1, [0,0,1])
            v1, v2, _mat32 = rotation_matrix(pd2, [0,0,-1])
 
            out.append([[np.dot(_mat31,temp1[0]), np.dot(_mat31,temp1[1])],
                        [np.dot(_mat32,temp2[0]), np.dot(_mat32,temp2[1])]])

            ##equator-patch1
            ## _mat 1 is the same calculated above
            v1, v2, _mat3 = rotation_matrix(self.topo[0], [1,0,0])
            
            temp1 = [np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])]
            pd1 = temp1[1]-temp1[0];
            v1, v2, _mat2 = rotation_matrix(pd1, [0,0,-1])

            out.append([[np.dot(_mat3,self.topo[0]), np.dot(_mat3,self.topo[1])],
                        [np.dot(_mat2,temp1[0]), np.dot(_mat2,temp1[1])]])
            
            if not self.samepatch:
                ## patch-patch p1 vs p2
                v1, v2, _mat1 = rotation_matrix(self.topo[0], [1,0,0])
                v1, v2, _mat2 = rotation_matrix(self.topo[1], [-1,0,0])

                out.append([[np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])],
                            [np.dot(_mat2,self.topo[0]), np.dot(_mat2,self.topo[1])]])
                
                ## patch-patch p2 vs p2
                v1, v2, _mat1 = rotation_matrix(self.topo[1], [1,0,0])
                v1, v2, _mat2 = rotation_matrix(self.topo[1], [-1,0,0])

                out.append([[np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])],
                            [np.dot(_mat2,self.topo[0]), np.dot(_mat2,self.topo[1])]])

                ##Equator-patch2
                v1, v2, _mat1 = rotation_matrix(ps, [1,0,0])
                v1, v2, _mat3 = rotation_matrix(self.topo[1], [1,0,0])
            
                temp1 = [np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])]
                pd1 = temp1[1]-temp1[0];
                v1, v2, _mat2 = rotation_matrix(pd1, [0,0,-1])

                out.append([[np.dot(_mat3,self.topo[0]), np.dot(_mat3,self.topo[1])], 
                            [np.dot(_mat2,temp1[0]), np.dot(_mat2,temp1[1])]])
            
            if self.doubleint:
                ## _mat 1 is the same calculated above
                v1, v2, _mat2 = rotation_matrix(ps, [-1,0,0])
            
                temp1 = [np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])]
                temp2 = [np.dot(_mat2,self.topo[0]), np.dot(_mat2,self.topo[1])]
            
                pd1 = temp1[1]-temp1[0]; pd2 = temp2[1]-temp2[0]
                v1, v2, _mat31 = rotation_matrix(pd1, [0,0,1])
                v1, v2, _mat32 = rotation_matrix(pd2, [0,0,1])
 
                out.append([[np.dot(_mat31,temp1[0]), np.dot(_mat31,temp1[1])],
                            [np.dot(_mat32,temp2[0]), np.dot(_mat32,temp2[1])]])
                
                ##double patch-patch rotation matrices
                ## _mat 1 is the same calculated above
                v1, v2, _mat2 = rotation_matrix(ps, [-1,0,0])
                
                temp1 = [np.dot(_mat1,self.topo[0]), np.dot(_mat1,self.topo[1])]
                temp2 = [np.dot(_mat2,self.topo[0]), np.dot(_mat2,self.topo[1])]
            
                pd1 = temp1[1]-temp1[0]; pd2 = temp2[1]-temp2[0]
                v1, v2, _mat31 = rotation_matrix(pd1, [0,0,-1])
                v1, v2, _mat32 = rotation_matrix(pd2, [0,0,1])
                
                temp21 = [np.dot(_mat31,temp1[0]), np.dot(_mat31,temp1[1])]
                temp22 = [np.dot(_mat32,temp2[0]), np.dot(_mat32,temp2[1])]

                out.append([temp21,temp22])

                if not self.samepatch:
                    temp23 = deepcopy(temp22)
                    temp23[0][2] *= -1; temp23[1][2] *= -1

                    out.append([temp21, temp23])


        return out

    def generate_orientations_3P(self): 
          
        out = []
        ##patch-patch 1 1 
        v1, v2, _mat1 = rotation_matrix(self.topo[0], [1,0,0])
        _mat2 = np.reshape(np.asarray([-1,0,0,0,1,0,0,0,1]),(3,3))

        out.append([[np.dot(_mat1,self.topo[i]) for i in range(self.npatch)],
                    [np.dot(_mat2,self.topo[i]) for i in range(self.npatch)]])
            
        ## equator-equator
        v1, v2, _mat1 = rotation_matrix([0,0,1], [1,0,0])

        out.append([[np.dot(_mat1,self.topo[i]) for i in range(self.npatch)],
                    [np.dot(_mat1,self.topo[i]) for i in range(self.npatch)]])
            
        ##equator-patch 1 
        v1, v2, _mat1 = rotation_matrix(self.topo[0], [1,0,0])
        v1, v2, _mat2 = rotation_matrix([0,0,1], [1,0,0])
            
        out.append([[np.dot(_mat1,self.topo[i]) for i in range(self.npatch)],
                    [np.dot(_mat2,self.topo[i]) for i in range(self.npatch)]])

        if not self.samepatch:
           
            ##patch-patch i j  
            for k in range(self.npatch):
                for j in range(k,self.npatch):
                    if j == 0:
                        continue
                    
                    v1, v2, _mat1 = rotation_matrix(self.topo[k], [1,0,0])
                    v1, v2, _mat2= rotation_matrix(self.topo[j], [-1,0,0])

                    out.append([[np.dot(_mat1,self.topo[i]) for i in range(self.npatch)],
                                [np.dot(_mat2,self.topo[i]) for i in range(self.npatch)]])
         
            ## patch-equator j
            for k in range(1,self.npatch):
                v1, v2, _mat1 = rotation_matrix(self.topo[k], [1,0,0])
                v1, v2, _mat2 = rotation_matrix([0,0,1], [1,0,0])
            
                out.append([[np.dot(_mat1,self.topo[i]) for i in range(self.npatch)],
                            [np.dot(_mat2,self.topo[i]) for i in range(self.npatch)]])



        return out

