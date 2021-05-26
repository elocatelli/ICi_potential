import numpy as np

def to_spherical(_in):

    _in = np.asarray(_in)
    r = np.linalg.norm(_in)
    
    th = _in[2]/r
    #if np.fabs(th) > 1.:
    #    temp = np.fabs(th)/th; th=temp
    if _in[2] >= 0:
        th = np.arccos(th)
    else:
        th = -np.arccos(th)
    
    ph = np.arctan2(_in[1], _in[0])

    return [r, th, ph]


def safecopy(_copyfrom, _copyto):

    _copyfrom = np.atleast_2d(_copyfrom)
    _copyfrom.reshape(_copyfrom.size,1)

    if _copyfrom.size == 1:
         _copyto.fill(_copyfrom[0,0])
    else:
        if _copyfrom.size == _copyto.size:
            _copyfrom = _copyfrom.reshape(_copyto.shape)
            _copyto[:] = _copyfrom[:]
        else:
            print("dimensions are not compatible, check your input file!")
            print("dimensions inputed", _copyfrom.size, "dimensions expected", _copyto.size)
            exit(1)

def myconvert(_dict_name, content):

    if type(_dict_name) is int:
        try:
            _dict_name = int(content)
        except Exception as e:
            print(e)
            exit(1)

    if type(_dict_name) is float:
        try:
            _dict_name = float(content)
        except Exception as e:
            print(e)
            exit(1)

    if type(_dict_name) is np.ndarray:
        try:
            temp = np.asarray(content.split(","), dtype = float); print(temp)
        except:
            temp = np.asarray(content, dtype = float)
        safecopy(temp, _dict_name)

    if type(_dict_name) is bool:
        if content == 'False':
            _dict_name = False
        elif content == 'True':
            _dict_name = True
        else:
            raise ValueError


def mybroadcast_to(_input, _shape):

    out_flatshape = np.prod(_shape)
    out = np.empty(out_flatshape,dtype = _input.dtype)
    inshape = _input.size

    for i in range(int(out_flatshape/inshape)):
        out[i*inshape:(i+1)*inshape] = _input

    return np.reshape(out, _shape, order='C')

def axis_angle_rotation(in_axis, in_angle):

    #wants normalized vectors!
    tempmat = np.empty((3,3),dtype = float)
    in_angle2 = in_angle/2.

    a = np.cos(in_angle2); b = in_axis[0]*np.sin(in_angle2);  c = in_axis[1]*np.sin(in_angle2); d = in_axis[2]*np.sin(in_angle2);
    tempmat[0,0] = a*a + b*b - c*c -d*d; tempmat[0,1] = 2*(b*c-a*d);  tempmat[0,2] = 2*(d*b + a*c);
    tempmat[1,0] = 2.*(b*c+a*d); tempmat[1,1] = a*a + c*c -b*b -d*d; tempmat[1,2] = 2*(c*d - a*b)
    tempmat[2,0] = 2.*(b*d - a*c); tempmat[2,1] = 2*(c*d + a*b); tempmat[2,2] = a*a +d*d - c*c - b*b

    return tempmat

def rotation_matrix(input_vec, input_ref):

    out = np.empty((3, 3), dtype=float)

    n1 = np.linalg.norm(input_vec)
    if n1 != 1.0:
        v1 = input_vec / n1
    else:
        v1 = input_vec

    n2 = np.linalg.norm(input_ref)
    if n2 != 1.0:
        v2 = input_ref / n2
    else:
        v2 = input_ref

    c = np.dot(v1, v2)
    if c == -1:
        #print("vectors ", v1, "and ", v2," are opposite: rotation not possible!")
        #exit(1)
        out[0,0] = 1-2*v1[0]**2; out[0,1] = -2*v1[0]*v1[1]; out[0,2] = -2*v1[0]*v1[2];
        out[1,0] = out[0,1]; out[1,1] = 1-2*v1[1]**2; out[1,2] = -2*v1[1]*v1[2];
        out[2,0] = out[0,2]; out[2,1] = out[1,2]; out[2,2] = 1-2*v1[2]**2
    else:

        if np.array_equal(v1, v2) == True:
            out = np.identity(3)
        else:
            v = np.cross(v1, v2)
            vmat = np.reshape( np.asarray([0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0]), (3, 3) )
            out = np.identity(3) + vmat + np.dot(vmat, vmat) * (1.0 / (1.0 + c))

    return v1, v2, out

def check_patches(_input, npatches, ecc):

    for i in range(1,npatches+1):
        if np.linalg.norm(_input[i]-_input[0]) - ecc[i-1] > 1E-5:
            print("patch is not at the right distance", np.linalg.norm(_input[i]-_input[0]),  ecc[i-1])
            exit(1)


def move_part(_part, _mov):
    _mov = np.asarray(_mov)
    return _part[:] + _mov

def rotate_part(_part, _axis, angle):
    _mat = axis_angle_rotation(_axis, angle)
    _out = np.empty_like(_part);

    for i in range(_part.shape[0]):
        _out[i,:3] = np.dot(_mat,_part[i,:3])

    return _out
