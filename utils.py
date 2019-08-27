import numpy as np

def safecopy(_copyfrom, _copyto):

    _copyfrom = np.asarray(_copyfrom)
    _copyfrom.reshape((_copyfrom.size,1))

    try: 
        _copyto[:] = _copyfrom[:]
    except:
        if _copyfrom.size == 1:
            _copyto[:] = _copyfrom[0]
        else:
            print "dimensions are not compatible, check your input file!" 
            exit(1)

def mybroadcast_to(_input, _shape):

        out_flatshape = np.prod(_shape)
        out = np.empty(out_flatshape,dtype = _input.dtype)
        inshape = _input.size

        for i in range(out_flatshape/inshape):
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


