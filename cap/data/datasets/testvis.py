import os
import torch
import mmcv
import numpy as np
from PIL import Image
from cap.registry import OBJECT_REGISTRY
import torch.utils.data as data
# from pyquaternion import Quaternion
from math import sqrt, pi, sin, cos, asin, acos, atan2, exp, log
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from matplotlib.axes import Axes
import warnings
import functools
from inspect import getfullargspec
from mmcv.ops import box_iou_rotated, points_in_boxes_all, points_in_boxes_part
import cv2

__all__ = [
    "NuscDetDataset",
]

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


class Quaternion:
    """Class to represent a 4-dimensional complex number or quaternion.

    Quaternion objects can be used generically as 4D numbers,
    or as unit quaternions to represent rotations in 3D space.

    Attributes:
        q: Quaternion 4-vector represented as a Numpy array

    """

    def __init__(self, *args, **kwargs):
        """Initialise a new Quaternion object.

        See Object Initialisation docs for complete behaviour:

        https://kieranwynn.github.io/pyquaternion/#object-initialisation

        """
        s = len(args)
        if s == 0:
            # No positional arguments supplied
            if kwargs:
                # Keyword arguments provided
                if ("scalar" in kwargs) or ("vector" in kwargs):
                    scalar = kwargs.get("scalar", 0.0)
                    if scalar is None:
                        scalar = 0.0
                    else:
                        scalar = float(scalar)

                    vector = kwargs.get("vector", [])
                    vector = self._validate_number_sequence(vector, 3)

                    self.q = np.hstack((scalar, vector))
                elif ("real" in kwargs) or ("imaginary" in kwargs):
                    real = kwargs.get("real", 0.0)
                    if real is None:
                        real = 0.0
                    else:
                        real = float(real)

                    imaginary = kwargs.get("imaginary", [])
                    imaginary = self._validate_number_sequence(imaginary, 3)

                    self.q = np.hstack((real, imaginary))
                elif ("axis" in kwargs) or ("radians" in kwargs) or (
                        "degrees" in kwargs) or ("angle" in kwargs):
                    try:
                        axis = self._validate_number_sequence(
                            kwargs["axis"], 3)
                    except KeyError:
                        raise ValueError(
                            "A valid rotation 'axis' parameter must be provided to describe a meaningful rotation."
                        )
                    angle = kwargs.get('radians') or self.to_radians(
                        kwargs.get('degrees')) or kwargs.get('angle') or 0.0
                    self.q = Quaternion._from_axis_angle(axis, angle).q
                elif "array" in kwargs:
                    self.q = self._validate_number_sequence(kwargs["array"], 4)
                elif "matrix" in kwargs:
                    optional_args = {
                        key: kwargs[key]
                        for key in kwargs if key in ['rtol', 'atol']
                    }
                    self.q = Quaternion._from_matrix(kwargs["matrix"],
                                                     **optional_args).q
                else:
                    keys = sorted(kwargs.keys())
                    elements = [kwargs[kw] for kw in keys]
                    if len(elements) == 1:
                        r = float(elements[0])
                        self.q = np.array([r, 0.0, 0.0, 0.0])
                    else:
                        self.q = self._validate_number_sequence(elements, 4)

            else:
                # Default initialisation
                self.q = np.array([1.0, 0.0, 0.0, 0.0])
        elif s == 1:
            # Single positional argument supplied
            if isinstance(args[0], Quaternion):
                self.q = args[0].q
                return
            if args[0] is None:
                raise TypeError("Object cannot be initialised from {}".format(
                    type(args[0])))
            try:
                r = float(args[0])
                self.q = np.array([r, 0.0, 0.0, 0.0])
                return
            except TypeError:
                pass  # If the single argument is not scalar, it should be a sequence

            self.q = self._validate_number_sequence(args[0], 4)
            return

        else:
            # More than one positional argument supplied
            self.q = self._validate_number_sequence(args, 4)

    def __hash__(self):
        return hash(tuple(self.q))

    def _validate_number_sequence(self, seq, n):
        """Validate a sequence to be of a certain length and ensure it's a numpy array of floats.

        Raises:
            ValueError: Invalid length or non-numeric value
        """
        if seq is None:
            return np.zeros(n)
        if len(seq) == n:
            try:
                l = [float(e) for e in seq]
            except ValueError:
                raise ValueError(
                    "One or more elements in sequence <{!r}> cannot be interpreted as a real number"
                    .format(seq))
            else:
                return np.asarray(l)
        elif len(seq) == 0:
            return np.zeros(n)
        else:
            raise ValueError(
                "Unexpected number of elements in sequence. Got: {}, Expected: {}."
                .format(len(seq), n))

    # Initialise from matrix
    @classmethod
    def _from_matrix(cls, matrix, rtol=1e-05, atol=1e-08):
        """Initialise from matrix representation

        Create a Quaternion by specifying the 3x3 rotation or 4x4 transformation matrix
        (as a numpy array) from which the quaternion's rotation should be created.

        """
        try:
            shape = matrix.shape
        except AttributeError:
            raise TypeError(
                "Invalid matrix type: Input must be a 3x3 or 4x4 numpy array or matrix"
            )

        if shape == (3, 3):
            R = matrix
        elif shape == (4, 4):
            R = matrix[:-1][:, :-1]  # Upper left 3x3 sub-matrix
        else:
            raise ValueError(
                "Invalid matrix shape: Input must be a 3x3 or 4x4 numpy array or matrix"
            )

        # Check matrix properties
        if not np.allclose(
                np.dot(R,
                       R.conj().transpose()), np.eye(3), rtol=rtol, atol=atol):
            raise ValueError(
                "Matrix must be orthogonal, i.e. its transpose should be its inverse"
            )
        if not np.isclose(np.linalg.det(R), 1.0, rtol=rtol, atol=atol):
            raise ValueError(
                "Matrix must be special orthogonal i.e. its determinant must be +1.0"
            )

        def decomposition_method(matrix):
            """ Method supposedly able to deal with non-orthogonal matrices - NON-FUNCTIONAL!
            Based on this method: http://arc.aiaa.org/doi/abs/10.2514/2.4654
            """
            x, y, z = 0, 1, 2  # indices
            K = np.array([[
                R[x, x] - R[y, y] - R[z, z], R[y, x] + R[x, y],
                R[z, x] + R[x, z], R[y, z] - R[z, y]
            ],
                          [
                              R[y, x] + R[x, y], R[y, y] - R[x, x] - R[z, z],
                              R[z, y] + R[y, z], R[z, x] - R[x, z]
                          ],
                          [
                              R[z, x] + R[x, z], R[z, y] + R[y, z],
                              R[z, z] - R[x, x] - R[y, y], R[x, y] - R[y, x]
                          ],
                          [
                              R[y, z] - R[z, y], R[z, x] - R[x, z],
                              R[x, y] - R[y, x], R[x, x] + R[y, y] + R[z, z]
                          ]])
            K = K / 3.0

            e_vals, e_vecs = np.linalg.eig(K)
            print('Eigenvalues:', e_vals)
            print('Eigenvectors:', e_vecs)
            max_index = np.argmax(e_vals)
            principal_component = e_vecs[max_index]
            return principal_component

        def trace_method(matrix):
            """
            This code uses a modification of the algorithm described in:
            https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
            which is itself based on the method described here:
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

            Altered to work with the column vector convention instead of row vectors
            """
            m = matrix.conj().transpose(
            )  # This method assumes row-vector and postmultiplication of that vector
            if m[2, 2] < 0:
                if m[0, 0] > m[1, 1]:
                    t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                    q = [
                        m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0],
                        m[2, 0] + m[0, 2]
                    ]
                else:
                    t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                    q = [
                        m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t,
                        m[1, 2] + m[2, 1]
                    ]
            else:
                if m[0, 0] < -m[1, 1]:
                    t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                    q = [
                        m[0, 1] - m[1, 0], m[2, 0] + m[0, 2],
                        m[1, 2] + m[2, 1], t
                    ]
                else:
                    t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                    q = [
                        t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2],
                        m[0, 1] - m[1, 0]
                    ]

            q = np.array(q).astype('float64')
            q *= 0.5 / sqrt(t)
            return q

        return cls(array=trace_method(R))

    # Initialise from axis-angle
    @classmethod
    def _from_axis_angle(cls, axis, angle):
        """Initialise from axis and angle representation

        Create a Quaternion by specifying the 3-vector rotation axis and rotation
        angle (in radians) from which the quaternion's rotation should be created.

        Params:
            axis: a valid numpy 3-vector
            angle: a real valued angle in radians
        """
        mag_sq = np.dot(axis, axis)
        if mag_sq == 0.0:
            raise ZeroDivisionError("Provided rotation axis has no length")
        # Ensure axis is in unit vector form
        if (abs(1.0 - mag_sq) > 1e-12):
            axis = axis / sqrt(mag_sq)
        theta = angle / 2.0
        r = cos(theta)
        i = axis * sin(theta)

        return cls(r, i[0], i[1], i[2])

    @classmethod
    def random(cls):
        """Generate a random unit quaternion.

        Uniformly distributed across the rotation space
        As per: http://planning.cs.uiuc.edu/node198.html
        """
        r1, r2, r3 = np.random.random(3)

        q1 = sqrt(1.0 - r1) * (sin(2 * pi * r2))
        q2 = sqrt(1.0 - r1) * (cos(2 * pi * r2))
        q3 = sqrt(r1) * (sin(2 * pi * r3))
        q4 = sqrt(r1) * (cos(2 * pi * r3))

        return cls(q1, q2, q3, q4)

    # Representation
    def __str__(self):
        """An informal, nicely printable string representation of the Quaternion object.
        """
        return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(
            self.q[0], self.q[1], self.q[2], self.q[3])

    def __repr__(self):
        """The 'official' string representation of the Quaternion object.

        This is a string representation of a valid Python expression that could be used
        to recreate an object with the same value (given an appropriate environment)
        """
        return "Quaternion({!r}, {!r}, {!r}, {!r})".format(
            self.q[0], self.q[1], self.q[2], self.q[3])

    def __format__(self, formatstr):
        """Inserts a customisable, nicely printable string representation of the Quaternion object

        The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types.
        Check out the official Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for details.
        """
        if formatstr.strip() == '':  # Defualt behaviour mirrors self.__str__()
            formatstr = '+.3f'

        string = \
            "{:" + formatstr +"} "  + \
            "{:" + formatstr +"}i " + \
            "{:" + formatstr +"}j " + \
            "{:" + formatstr +"}k"
        return string.format(self.q[0], self.q[1], self.q[2], self.q[3])

    # Type Conversion
    def __int__(self):
        """Implements type conversion to int.

        Truncates the Quaternion object by only considering the real
        component and rounding to the next integer value towards zero.
        Note: to round to the closest integer, use int(round(float(q)))
        """
        return int(self.q[0])

    def __float__(self):
        """Implements type conversion to float.

        Truncates the Quaternion object by only considering the real
        component.
        """
        return float(self.q[0])

    def __complex__(self):
        """Implements type conversion to complex.

        Truncates the Quaternion object by only considering the real
        component and the first imaginary component.
        This is equivalent to a projection from the 4-dimensional hypersphere
        to the 2-dimensional complex plane.
        """
        return complex(self.q[0], self.q[1])

    def __bool__(self):
        return not (self == Quaternion(0.0))

    def __nonzero__(self):
        return not (self == Quaternion(0.0))

    def __invert__(self):
        return (self == Quaternion(0.0))

    # Comparison
    def __eq__(self, other):
        """Returns true if the following is true for each element:
        `absolute(a - b) <= (atol + rtol * absolute(b))`
        """
        if isinstance(other, Quaternion):
            r_tol = 1.0e-13
            a_tol = 1.0e-14
            try:
                isEqual = np.allclose(self.q, other.q, rtol=r_tol, atol=a_tol)
            except AttributeError:
                raise AttributeError(
                    "Error in internal quaternion representation means it cannot be compared like a numpy array."
                )
            return isEqual
        return self.__eq__(self.__class__(other))

    # Negation
    def __neg__(self):
        return self.__class__(array=-self.q)

    # Absolute value
    def __abs__(self):
        return self.norm

    # Addition
    def __add__(self, other):
        if isinstance(other, Quaternion):
            return self.__class__(array=self.q + other.q)
        return self + self.__class__(other)

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    # Subtraction
    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.__class__(array=np.dot(self._q_matrix(), other.q))
        return self * self.__class__(other)

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self.__class__(other) * self

    def __matmul__(self, other):
        if isinstance(other, Quaternion):
            return self.q.__matmul__(other.q)
        return self.__matmul__(self.__class__(other))

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __rmatmul__(self, other):
        return self.__class__(other).__matmul__(self)

    # Division
    def __div__(self, other):
        if isinstance(other, Quaternion):
            if other == self.__class__(0.0):
                raise ZeroDivisionError("Quaternion divisor must be non-zero")
            return self * other.inverse
        return self.__div__(self.__class__(other))

    def __idiv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        return self.__class__(other) * self.inverse

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    # Exponentiation
    def __pow__(self, exponent):
        # source: https://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power
        exponent = float(exponent)  # Explicitly reject non-real exponents
        norm = self.norm
        if norm > 0.0:
            try:
                n, theta = self.polar_decomposition
            except ZeroDivisionError:
                # quaternion is a real number (no vector or imaginary part)
                return Quaternion(scalar=self.scalar**exponent)
            return (self.norm**exponent) * Quaternion(
                scalar=cos(exponent * theta),
                vector=(n * sin(exponent * theta)))
        return Quaternion(self)

    def __ipow__(self, other):
        return self**other

    def __rpow__(self, other):
        return other**float(self)

    # Quaternion Features
    def _vector_conjugate(self):
        return np.hstack((self.q[0], -self.q[1:4]))

    def _sum_of_squares(self):
        return np.dot(self.q, self.q)

    @property
    def conjugate(self):
        """Quaternion conjugate, encapsulated in a new instance.

        For a unit quaternion, this is the same as the inverse.

        Returns:
            A new Quaternion object clone with its vector part negated
        """
        return self.__class__(scalar=self.scalar, vector=-self.vector)

    @property
    def inverse(self):
        """Inverse of the quaternion object, encapsulated in a new instance.

        For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation.

        Returns:
            A new Quaternion object representing the inverse of this object
        """
        ss = self._sum_of_squares()
        if ss > 0:
            return self.__class__(array=(self._vector_conjugate() / ss))
        else:
            raise ZeroDivisionError(
                "a zero quaternion (0 + 0i + 0j + 0k) cannot be inverted")

    @property
    def norm(self):
        """L2 norm of the quaternion 4-vector.

        This should be 1.0 for a unit quaternion (versor)
        Slow but accurate. If speed is a concern, consider using _fast_normalise() instead

        Returns:
            A scalar real number representing the square root of the sum of the squares of the elements of the quaternion.
        """
        mag_squared = self._sum_of_squares()
        return sqrt(mag_squared)

    @property
    def magnitude(self):
        return self.norm

    def _normalise(self):
        """Object is guaranteed to be a unit quaternion after calling this
        operation UNLESS the object is equivalent to Quaternion(0)
        """
        if not self.is_unit():
            n = self.norm
            if n > 0:
                self.q = self.q / n

    def _fast_normalise(self):
        """Normalise the object to a unit quaternion using a fast approximation method if appropriate.

        Object is guaranteed to be a quaternion of approximately unit length
        after calling this operation UNLESS the object is equivalent to Quaternion(0)
        """
        if not self.is_unit():
            mag_squared = np.dot(self.q, self.q)
            if (mag_squared == 0):
                return
            if (abs(1.0 - mag_squared) < 2.107342e-08):
                mag = (
                    (1.0 + mag_squared) / 2.0
                )  # More efficient. Pade approximation valid if error is small
            else:
                mag = sqrt(
                    mag_squared
                )  # Error is too big, take the performance hit to calculate the square root properly

            self.q = self.q / mag

    @property
    def normalised(self):
        """Get a unit quaternion (versor) copy of this Quaternion object.

        A unit quaternion has a `norm` of 1.0

        Returns:
            A new Quaternion object clone that is guaranteed to be a unit quaternion
        """
        q = Quaternion(self)
        q._normalise()
        return q

    @property
    def polar_unit_vector(self):
        vector_length = np.linalg.norm(self.vector)
        if vector_length <= 0.0:
            raise ZeroDivisionError(
                'Quaternion is pure real and does not have a unique unit vector'
            )
        return self.vector / vector_length

    @property
    def polar_angle(self):
        return acos(self.scalar / self.norm)

    @property
    def polar_decomposition(self):
        """
        Returns the unit vector and angle of a non-scalar quaternion according to the following decomposition

        q =  q.norm() * (e ** (q.polar_unit_vector * q.polar_angle))

        source: https://en.wikipedia.org/wiki/Polar_decomposition#Quaternion_polar_decomposition
        """
        return self.polar_unit_vector, self.polar_angle

    @property
    def unit(self):
        return self.normalised

    def is_unit(self, tolerance=1e-14):
        """Determine whether the quaternion is of unit length to within a specified tolerance value.

        Params:
            tolerance: [optional] maximum absolute value by which the norm can differ from 1.0 for the object to be considered a unit quaternion. Defaults to `1e-14`.

        Returns:
            `True` if the Quaternion object is of unit length to within the specified tolerance value. `False` otherwise.
        """
        return abs(
            1.0 - self._sum_of_squares()
        ) < tolerance  # if _sum_of_squares is 1, norm is 1. This saves a call to sqrt()

    def _q_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return np.array([[self.q[0], -self.q[1], -self.q[2], -self.q[3]],
                         [self.q[1], self.q[0], -self.q[3], self.q[2]],
                         [self.q[2], self.q[3], self.q[0], -self.q[1]],
                         [self.q[3], -self.q[2], self.q[1], self.q[0]]])

    def _q_bar_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return np.array([[self.q[0], -self.q[1], -self.q[2], -self.q[3]],
                         [self.q[1], self.q[0], self.q[3], -self.q[2]],
                         [self.q[2], -self.q[3], self.q[0], self.q[1]],
                         [self.q[3], self.q[2], -self.q[1], self.q[0]]])

    def _rotate_quaternion(self, q):
        """Rotate a quaternion vector using the stored rotation.

        Params:
            q: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

        Returns:
            A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
        """
        self._normalise()
        return self * q * self.conjugate

    def rotate(self, vector):
        """Rotate a 3D vector by the rotation stored in the Quaternion object.

        Params:
            vector: A 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values.
                Some types that are recognised are: numpy arrays, lists and tuples.
                A 3-vector can also be represented by a Quaternion object who's scalar part is 0 and vector part is the required 3-vector.
                Thus it is possible to call `Quaternion.rotate(q)` with another quaternion object as an input.

        Returns:
            The rotated vector returned as the same type it was specified at input.

        Raises:
            TypeError: if any of the vector elements cannot be converted to a real number.
            ValueError: if `vector` cannot be interpreted as a 3-vector or a Quaternion object.

        """
        if isinstance(vector, Quaternion):
            return self._rotate_quaternion(vector)
        q = Quaternion(vector=vector)
        a = self._rotate_quaternion(q).vector
        if isinstance(vector, list):
            l = [x for x in a]
            return l
        elif isinstance(vector, tuple):
            l = [x for x in a]
            return tuple(l)
        else:
            return a

    @classmethod
    def exp(cls, q):
        """Quaternion Exponential.

        Find the exponential of a quaternion amount.

        Params:
             q: the input quaternion/argument as a Quaternion object.

        Returns:
             A quaternion amount representing the exp(q). See [Source](https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation for more information and mathematical background).

        Note:
             The method can compute the exponential of any quaternion.
        """
        tolerance = 1e-17
        v_norm = np.linalg.norm(q.vector)
        vec = q.vector
        if v_norm > tolerance:
            vec = vec / v_norm
        magnitude = exp(q.scalar)
        return Quaternion(scalar=magnitude * cos(v_norm),
                          vector=magnitude * sin(v_norm) * vec)

    @classmethod
    def log(cls, q):
        """Quaternion Logarithm.

        Find the logarithm of a quaternion amount.

        Params:
             q: the input quaternion/argument as a Quaternion object.

        Returns:
             A quaternion amount representing log(q) := (log(|q|), v/|v|acos(w/|q|)).

        Note:
            The method computes the logarithm of general quaternions. See [Source](https://math.stackexchange.com/questions/2552/the-logarithm-of-quaternion/2554#2554) for more details.
        """
        v_norm = np.linalg.norm(q.vector)
        q_norm = q.norm
        tolerance = 1e-17
        if q_norm < tolerance:
            # 0 quaternion - undefined
            return Quaternion(scalar=-float('inf'),
                              vector=float('nan') * q.vector)
        if v_norm < tolerance:
            # real quaternions - no imaginary part
            return Quaternion(scalar=log(q_norm), vector=[0, 0, 0])
        vec = q.vector / v_norm
        return Quaternion(scalar=log(q_norm),
                          vector=acos(q.scalar / q_norm) * vec)

    @classmethod
    def exp_map(cls, q, eta):
        """Quaternion exponential map.

        Find the exponential map on the Riemannian manifold described
        by the quaternion space.

        Params:
             q: the base point of the exponential map, i.e. a Quaternion object
           eta: the argument of the exponential map, a tangent vector, i.e. a Quaternion object

        Returns:
            A quaternion p such that p is the endpoint of the geodesic starting at q
            in the direction of eta, having the length equal to the magnitude of eta.

        Note:
            The exponential map plays an important role in integrating orientation
            variations (e.g. angular velocities). This is done by projecting
            quaternion tangent vectors onto the quaternion manifold.
        """
        return q * Quaternion.exp(eta)

    @classmethod
    def sym_exp_map(cls, q, eta):
        """Quaternion symmetrized exponential map.

        Find the symmetrized exponential map on the quaternion Riemannian
        manifold.

        Params:
             q: the base point as a Quaternion object
           eta: the tangent vector argument of the exponential map
                as a Quaternion object

        Returns:
            A quaternion p.

        Note:
            The symmetrized exponential formulation is akin to the exponential
            formulation for symmetric positive definite tensors [Source](http://www.academia.edu/7656761/On_the_Averaging_of_Symmetric_Positive-Definite_Tensors)
        """
        sqrt_q = q**0.5
        return sqrt_q * Quaternion.exp(eta) * sqrt_q

    @classmethod
    def log_map(cls, q, p):
        """Quaternion logarithm map.

        Find the logarithm map on the quaternion Riemannian manifold.

        Params:
             q: the base point at which the logarithm is computed, i.e.
                a Quaternion object
             p: the argument of the quaternion map, a Quaternion object

        Returns:
            A tangent vector having the length and direction given by the
            geodesic joining q and p.
        """
        return Quaternion.log(q.inverse * p)

    @classmethod
    def sym_log_map(cls, q, p):
        """Quaternion symmetrized logarithm map.

        Find the symmetrized logarithm map on the quaternion Riemannian manifold.

        Params:
             q: the base point at which the logarithm is computed, i.e.
                a Quaternion object
             p: the argument of the quaternion map, a Quaternion object

        Returns:
            A tangent vector corresponding to the symmetrized geodesic curve formulation.

        Note:
            Information on the symmetrized formulations given in [Source](https://www.researchgate.net/publication/267191489_Riemannian_L_p_Averaging_on_Lie_Group_of_Nonzero_Quaternions).
        """
        inv_sqrt_q = (q**(-0.5))
        return Quaternion.log(inv_sqrt_q * p * inv_sqrt_q)

    @classmethod
    def absolute_distance(cls, q0, q1):
        """Quaternion absolute distance.

        Find the distance between two quaternions accounting for the sign ambiguity.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive scalar corresponding to the chord of the shortest path/arc that
           connects q0 to q1.

        Note:
           This function does not measure the distance on the hypersphere, but
           it takes into account the fact that q and -q encode the same rotation.
           It is thus a good indicator for rotation similarities.
        """
        q0_minus_q1 = q0 - q1
        q0_plus_q1 = q0 + q1
        d_minus = q0_minus_q1.norm
        d_plus = q0_plus_q1.norm
        if d_minus < d_plus:
            return d_minus
        else:
            return d_plus

    @classmethod
    def distance(cls, q0, q1):
        """Quaternion intrinsic distance.

        Find the intrinsic geodesic distance between q0 and q1.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive amount corresponding to the length of the geodesic arc
           connecting q0 to q1.

        Note:
           Although the q0^(-1)*q1 != q1^(-1)*q0, the length of the path joining
           them is given by the logarithm of those product quaternions, the norm
           of which is the same.
        """
        q = Quaternion.log_map(q0, q1)
        return q.norm

    @classmethod
    def sym_distance(cls, q0, q1):
        """Quaternion symmetrized distance.

        Find the intrinsic symmetrized geodesic distance between q0 and q1.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive amount corresponding to the length of the symmetrized
           geodesic curve connecting q0 to q1.

        Note:
           This formulation is more numerically stable when performing
           iterative gradient descent on the Riemannian quaternion manifold.
           However, the distance between q and -q is equal to pi, rendering this
           formulation not useful for measuring rotation similarities when the
           samples are spread over a "solid" angle of more than pi/2 radians
           (the spread refers to quaternions as point samples on the unit hypersphere).
        """
        q = Quaternion.sym_log_map(q0, q1)
        return q.norm

    @classmethod
    def slerp(cls, q0, q1, amount=0.5):
        """Spherical Linear Interpolation between quaternions.
        Implemented as described in https://en.wikipedia.org/wiki/Slerp

        Find a valid quaternion rotation at a specified distance along the
        minor arc of a great circle passing through any two existing quaternion
        endpoints lying on the unit radius hypersphere.

        This is a class method and is called as a method of the class itself rather than on a particular instance.

        Params:
            q0: first endpoint rotation as a Quaternion object
            q1: second endpoint rotation as a Quaternion object
            amount: interpolation parameter between 0 and 1. This describes the linear placement position of
                the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`.
                Defaults to the midpoint (0.5).

        Returns:
            A new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion.

        Note:
            This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere).
                Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.
        """
        # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
        q0._fast_normalise()
        q1._fast_normalise()
        amount = np.clip(amount, 0, 1)

        dot = np.dot(q0.q, q1.q)

        # If the dot product is negative, slerp won't take the shorter path.
        # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
        # Fix by reversing one quaternion
        if dot < 0.0:
            q0.q = -q0.q
            dot = -dot

        # sin_theta_0 can not be zero
        if dot > 0.9995:
            qr = Quaternion(q0.q + amount * (q1.q - q0.q))
            qr._fast_normalise()
            return qr

        theta_0 = np.arccos(
            dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * amount
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        qr = Quaternion((s0 * q0.q) + (s1 * q1.q))
        qr._fast_normalise()
        return qr

    @classmethod
    def intermediates(cls, q0, q1, n, include_endpoints=False):
        """Generator method to get an iterable sequence of `n` evenly spaced quaternion
        rotations between any two existing quaternion endpoints lying on the unit
        radius hypersphere.

        This is a convenience function that is based on `Quaternion.slerp()` as defined above.

        This is a class method and is called as a method of the class itself rather than on a particular instance.

        Params:
            q_start: initial endpoint rotation as a Quaternion object
            q_end:   final endpoint rotation as a Quaternion object
            n:       number of intermediate quaternion objects to include within the interval
            include_endpoints: [optional] if set to `True`, the sequence of intermediates
                will be 'bookended' by `q_start` and `q_end`, resulting in a sequence length of `n + 2`.
                If set to `False`, endpoints are not included. Defaults to `False`.

        Yields:
            A generator object iterating over a sequence of intermediate quaternion objects.

        Note:
            This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere).
            Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.
        """
        step_size = 1.0 / (n + 1)
        if include_endpoints:
            steps = [i * step_size for i in range(0, n + 2)]
        else:
            steps = [i * step_size for i in range(1, n + 1)]
        for step in steps:
            yield cls.slerp(q0, q1, step)

    def derivative(self, rate):
        """Get the instantaneous quaternion derivative representing a quaternion rotating at a 3D rate vector `rate`

        Params:
            rate: numpy 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively.

        Returns:
            A unit quaternion describing the rotation rate
        """
        rate = self._validate_number_sequence(rate, 3)
        return 0.5 * self * Quaternion(vector=rate)

    def integrate(self, rate, timestep):
        """Advance a time varying quaternion to its value at a time `timestep` in the future.

        The Quaternion object will be modified to its future value.
        It is guaranteed to remain a unit quaternion.

        Params:

        rate: numpy 3-array (or array-like) describing rotation rates about the
            global x, y and z axes respectively.
        timestep: interval over which to integrate into the future.
            Assuming *now* is `T=0`, the integration occurs over the interval
            `T=0` to `T=timestep`. Smaller intervals are more accurate when
            `rate` changes over time.

        Note:
            The solution is closed form given the assumption that `rate` is constant
            over the interval of length `timestep`.
        """
        self._fast_normalise()
        rate = self._validate_number_sequence(rate, 3)

        rotation_vector = rate * timestep
        rotation_norm = np.linalg.norm(rotation_vector)
        if rotation_norm > 0:
            axis = rotation_vector / rotation_norm
            angle = rotation_norm
            q2 = Quaternion(axis=axis, angle=angle)
            self.q = (self * q2).q
            self._fast_normalise()

    @property
    def rotation_matrix(self):
        """Get the 3x3 rotation matrix equivalent of the quaternion rotation.

        Returns:
            A 3x3 orthogonal rotation matrix as a 3x3 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

        """
        self._normalise()
        product_matrix = np.dot(self._q_matrix(),
                                self._q_bar_matrix().conj().transpose())
        return product_matrix[1:][:, 1:]

    @property
    def transformation_matrix(self):
        """Get the 4x4 homogeneous transformation matrix equivalent of the quaternion rotation.

        Returns:
            A 4x4 homogeneous transformation matrix as a 4x4 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        t = np.array([[0.0], [0.0], [0.0]])
        Rt = np.hstack([self.rotation_matrix, t])
        return np.vstack([Rt, np.array([0.0, 0.0, 0.0, 1.0])])

    @property
    def yaw_pitch_roll(self):
        """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention

        Returns:
            yaw:    rotation angle around the z-axis in radians, in the range `[-pi, pi]`
            pitch:  rotation angle around the y'-axis in radians, in the range `[-pi/2, -pi/2]`
            roll:   rotation angle around the x''-axis in radians, in the range `[-pi, pi]`

        The resulting rotation_matrix would be R = R_x(roll) R_y(pitch) R_z(yaw)

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """

        self._normalise()
        yaw = np.arctan2(2 * (self.q[0] * self.q[3] - self.q[1] * self.q[2]),
                         1 - 2 * (self.q[2]**2 + self.q[3]**2))
        pitch = np.arcsin(2 * (self.q[0] * self.q[2] + self.q[3] * self.q[1]))
        roll = np.arctan2(2 * (self.q[0] * self.q[1] - self.q[2] * self.q[3]),
                          1 - 2 * (self.q[1]**2 + self.q[2]**2))

        return yaw, pitch, roll

    def _wrap_angle(self, theta):
        """Helper method: Wrap any angle to lie between -pi and pi

        Odd multiples of pi are wrapped to +pi (as opposed to -pi)
        """
        result = ((theta + pi) % (2 * pi)) - pi
        if result == -pi:
            result = pi
        return result

    def get_axis(self, undefined=np.zeros(3)):
        """Get the axis or vector about which the quaternion rotation occurs

        For a null rotation (a purely real quaternion), the rotation angle will
        always be `0`, but the rotation axis is undefined.
        It is by default assumed to be `[0, 0, 0]`.

        Params:
            undefined: [optional] specify the axis vector that should define a null rotation.
                This is geometrically meaningless, and could be any of an infinite set of vectors,
                but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.

        Returns:
            A Numpy unit 3-vector describing the Quaternion object's axis of rotation.

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        tolerance = 1e-17
        self._normalise()
        norm = np.linalg.norm(self.vector)
        if norm < tolerance:
            # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
            return undefined
        else:
            return self.vector / norm

    @property
    def axis(self):
        return self.get_axis()

    @property
    def angle(self):
        """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis.

        This is guaranteed to be within the range (-pi:pi) with the direction of
        rotation indicated by the sign.

        When a particular rotation describes a 180 degree rotation about an arbitrary
        axis vector `v`, the conversion to axis / angle representation may jump
        discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`,
        each being geometrically equivalent (see Note in documentation).

        Returns:
            A real number in the range (-pi:pi) describing the angle of rotation
                in radians about a Quaternion object's axis of rotation.

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        self._normalise()
        norm = np.linalg.norm(self.vector)
        return self._wrap_angle(2.0 * atan2(norm, self.scalar))

    @property
    def degrees(self):
        return self.to_degrees(self.angle)

    @property
    def radians(self):
        return self.angle

    @property
    def scalar(self):
        """ Return the real or scalar component of the quaternion object.

        Returns:
            A real number i.e. float
        """
        return self.q[0]

    @property
    def vector(self):
        """ Return the imaginary or vector component of the quaternion object.

        Returns:
            A numpy 3-array of floats. NOT guaranteed to be a unit vector
        """
        return self.q[1:4]

    @property
    def real(self):
        return self.scalar

    @property
    def imaginary(self):
        return self.vector

    @property
    def w(self):
        return self.q[0]

    @property
    def x(self):
        return self.q[1]

    @property
    def y(self):
        return self.q[2]

    @property
    def z(self):
        return self.q[3]

    @property
    def elements(self):
        """ Return all the elements of the quaternion object.

        Returns:
            A numpy 4-array of floats. NOT guaranteed to be a unit vector
        """
        return self.q

    def __getitem__(self, index):
        index = int(index)
        return self.q[index]

    def __setitem__(self, index, value):
        index = int(index)
        self.q[index] = float(value)

    def __copy__(self):
        result = self.__class__(self.q)
        return result

    def __deepcopy__(self, memo):
        result = self.__class__(deepcopy(self.q, memo))
        memo[id(self)] = result
        return result

    @staticmethod
    def to_degrees(angle_rad):
        if angle_rad is not None:
            return float(angle_rad) / pi * 180.0

    @staticmethod
    def to_radians(angle_deg):
        if angle_deg is not None:
            return float(angle_deg) / 180.0 * pi


class BasePoints(object):
    """Base class for Points.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int, optional): Number of the dimension of a point.
            Each row is (x, y, z). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the
            meaning of extra dimension. Defaults to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, points_dim)).to(dtype=torch.float32,
                                                        device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == \
            points_dim, tensor.size()

        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self):
        """torch.Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        """Set the coordinates of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self):
        """torch.Tensor:
            A vector with height of each point in shape (N, 1), or None."""
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor):
        """Set the height of each point."""
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self):
        """torch.Tensor:
            A vector with color of each point in shape (N, 3), or None."""
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor):
        """Set the color of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self):
        """torch.Shape: Shape of points."""
        return self.tensor.shape

    def shuffle(self):
        """Shuffle the points.

        Returns:
            torch.Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float | np.ndarray | torch.Tensor): Rotation matrix
                or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or \
            rotation.numel() == 1, f'invalid rotation shape {rotation.shape}'

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rotated_points, rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, :3][None], rotation, axis=axis, return_mat=True)
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            # rotation.numel() == 9
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
        """
        pass

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector (np.ndarray, torch.Tensor): Translation
                vector of size 3 or nx3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and \
                trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(
                f'Unsupported translation vector of shape {trans_vector.shape}'
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each point is
                inside the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 2] > point_range[2])
                          & (self.tensor[:, 0] < point_range[3])
                          & (self.tensor[:, 1] < point_range[4])
                          & (self.tensor[:, 2] < point_range[5]))
        return in_range_flags

    @property
    def bev(self):
        """torch.Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside
                the reference range.
        """
        in_range_flags = ((self.bev[:, 0] > point_range[0])
                          & (self.bev[:, 1] > point_range[1])
                          & (self.bev[:, 0] < point_range[2])
                          & (self.bev[:, 1] < point_range[3]))
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted box of the same type
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_points = points[3]`:
                return a `Points` that contains only one point.
            2. `new_points = points[2:10]`:
                return a slice of points.
            3. `new_points = points[vector]`:
                where vector is a torch.BoolTensor with `length = len(points)`.
                Nonzero elements in the vector will be selected.
            4. `new_points = points[3:11, vector]`:
                return a slice of points and attribute dims.
            5. `new_points = points[4:12, 2]`:
                return a slice of points with single attribute.
            Note that the returned Points might share storage with this Points,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of
                :class:`BasePoints` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1),
                                 points_dim=self.points_dim,
                                 attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if \
                    item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(
                set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')

        assert p.dim() == 2, \
            f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(p,
                             points_dim=p.shape[1],
                             attribute_dims=attribute_dims)

    def __len__(self):
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list):
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (list[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated Points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(torch.cat([p.tensor for p in points_list], dim=0),
                         points_dim=points_list[0].tensor.shape[1],
                         attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def to(self, device):
        """Convert current points to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new boxes object on the
                specific device.
        """
        original_type = type(self)
        return original_type(self.tensor.to(device),
                             points_dim=self.points_dim,
                             attribute_dims=self.attribute_dims)

    def clone(self):
        """Clone the Points.

        Returns:
            :obj:`BasePoints`: Box object with the same properties
                as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(),
                             points_dim=self.points_dim,
                             attribute_dims=self.attribute_dims)

    @property
    def device(self):
        """str: The device of the points are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a point as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A point of shape (4,).
        """
        yield from self.tensor

    def new_point(self, data):
        """Create a new point object with data.

        The new point and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor,
                             points_dim=self.points_dim,
                             attribute_dims=self.attribute_dims)


def array_converter(to_torch=True,
                    apply_to=tuple(),
                    template_arg_name_=None,
                    recover=True):
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy ndarrays
    for middle calculation, then convert output to original data-type if
    `recover=True`.

    Args:
        to_torch (Bool, optional): Whether convert to PyTorch tensors
            for middle calculation. Defaults to True.
        apply_to (tuple[str], optional): The arguments to which we apply
            data-type conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template (
            return arrays should have the same dtype and device
            as the template). Defaults to None. If None, we will use the
            first argument in `apply_to` as the template argument.
        recover (Bool, optional): Whether or not recover the wrapped function
            outputs to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or
            when apply_to contains an arg which is not among all args,
            a ValueError will be raised. When the template argument or
            an argument to convert is a list or tuple, and cannot be
            converted to a NumPy array, a ValueError will be raised.
        TypeError: When the type of the template argument or
                an argument to convert does not belong to the above range,
                or the contents of such an list-or-tuple-type argument
                do not share the same data type, a TypeError is raised.

    Returns:
        (function): wrapped function.

    Example:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add()
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the '
                                 f'argument list of function {func_name}')

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f'{arg_to_apply} is not '
                                     f'an argument of {func_name}')

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(
                        converter.convert(input_array=arg_value,
                                          target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=kwargs[arg_name],
                            target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data,
                                                         tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper


@array_converter(apply_to=('val', ))
def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor | np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        (torch.Tensor | np.ndarray): Value in the range of
            [-offset * period, (1-offset) * period]
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


@array_converter(apply_to=('points', 'angles'))
def rotation_3d_in_axis(points,
                        angles,
                        axis=0,
                        return_mat=False,
                        clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple | float):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(f'axis should in range '
                             f'[-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(
        ), 'Error: Pointcloud points must have format: %d x n' % self.nbr_dims(
        )
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'PointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep(
            cls,
            nusc: 'NuScenes',
            sample_rec: Dict,
            chan: str,
            ref_chan: str,
            nsweeps: int = 5,
            min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros(
            (cls.nbr_dims(), 0),
            dtype=np.float32 if cls == LidarPointCloud else np.float64)
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor',
                              ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                        Quaternion(ref_cs_rec['rotation']),
                                        inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'],
                                           Quaternion(
                                               ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(
                osp.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose',
                                        current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(
                current_pose_rec['translation'],
                Quaternion(current_pose_rec['rotation']),
                inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get(
                'calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(
                current_cs_rec['translation'],
                Quaternion(current_cs_rec['rotation']),
                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [
                ref_from_car, car_from_global, global_from_car,
                car_from_current
            ])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec[
                'timestamp']  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data',
                                          current_sd_rec['prev'])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()),
                                        size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(
            np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(self,
                      ax: Axes,
                      view: np.ndarray = np.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax: Axes,
                         view: np.ndarray = np.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self, color_channel: int, ax: Axes, view: np.ndarray,
                       x_lim: Tuple[float, float], y_lim: Tuple[float, float],
                       marker_size: float) -> None:
        """
        Helper function for rendering.
        :param color_channel: Point channel to use as color.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :],
                   points[1, :],
                   c=self.points[color_channel, :],
                   s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])


class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(
            file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)


def view_points(points: np.ndarray, view: np.ndarray,
                normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Defaults to True.
        origin (tuple[float], optional): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32,
                                                     device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """torch.Tensor:
            A vector with the top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor:
            A vector with bottom's height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for clearer usage.

        Returns:
            torch.Tensor: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        pass

    @property
    def corners(self):
        """torch.Tensor:
            a tensor with 8 corners of each box in shape (N, 8, 3)."""
        pass

    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation
            in XYWHR format, in shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):
        """torch.Tensor: A tensor of 2D BEV box of each box
            without rotation."""
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:,
                                                                [0, 1, 3, 2]],
                                  bev_rotated_boxes[:, :4])

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)

        Note:
            The original implementation of SECOND checks whether boxes in
            a range by checking whether the points are in a convex
            polygon, we reduce the burden for simpler cases.

        Returns:
            torch.Tensor: Whether each box is inside the reference range.
        """
        in_range_flags = ((self.bev[:, 0] > box_range[0])
                          & (self.bev[:, 1] > box_range[1])
                          & (self.bev[:, 0] < box_range[2])
                          & (self.bev[:, 1] < box_range[3]))
        return in_range_flags

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | numpy.ndarray |
                :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str, optional): Direction by which to flip.
                Can be chosen from 'horizontal' and 'vertical'.
                Defaults to 'horizontal'.
        """
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (torch.Tensor): Translation vector of size (1, 3).
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each box is
                inside the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 2] > box_range[2])
                          & (self.tensor[:, 0] < box_range[3])
                          & (self.tensor[:, 1] < box_range[4])
                          & (self.tensor[:, 2] < box_range[5]))
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor  # velocity

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float, optional): The offset of the yaw. Defaults to 0.5.
            period (float, optional): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold=0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float, optional): The threshold of minimal sizes.
                Defaults to 0.0.

        Returns:
            torch.Tensor: A binary vector which represents whether each
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
                :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1),
                                 box_dim=self.box_dim,
                                 with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0),
                        box_dim=boxes_list[0].tensor.shape[1],
                        with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the
                specific device.
        """
        original_type = type(self)
        return original_type(self.tensor.to(device),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties
                as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of IoU calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated 3D overlaps of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou', 'iof']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap
        overlaps_h = cls.height_overlaps(boxes1, boxes2)

        # bev overlap
        iou2d = box_iou_rotated(boxes1.bev, boxes2.bev)
        areas1 = (boxes1.bev[:, 2] * boxes1.bev[:, 3]).unsqueeze(1).expand(
            rows, cols)
        areas2 = (boxes2.bev[:, 2] * boxes2.bev[:, 3]).unsqueeze(0).expand(
            rows, cols)
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)

        # 3d overlaps
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d,
                                              min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d

    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor,
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    def points_in_boxes_part(self, points, boxes_override=None):
        """Find the box in which each point is.

        Args:
            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (torch.Tensor, optional): Boxes to override
                `self.tensor`. Defaults to None.

        Returns:
            torch.Tensor: The index of the first box that each point
                is in, in shape (M, ). Default value is -1
                (if the point is not enclosed by any box).

        Note:
            If a point is enclosed by multiple boxes, the index of the
            first box will be returned.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        if points.dim() == 2:
            points = points.unsqueeze(0)
        box_idx = points_in_boxes_part(points,
                                       boxes.unsqueeze(0).to(
                                           points.device)).squeeze(0)
        return box_idx

    def points_in_boxes_all(self, points, boxes_override=None):
        """Find all boxes in which each point is.

        Args:
            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (torch.Tensor, optional): Boxes to override
                `self.tensor`. Defaults to None.

        Returns:
            torch.Tensor: A tensor indicating whether a point is in a box,
                in shape (M, T). T is the number of boxes. Denote this
                tensor as A, if the m^th point is in the t^th box, then
                `A[m, t] == 1`, elsewise `A[m, t] == 0`.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1

        boxes = boxes.to(points_clone.device).unsqueeze(0)
        box_idxs_of_pts = points_in_boxes_all(points_clone, boxes)

        return box_idxs_of_pts.squeeze(0)

    def points_in_boxes(self, points, boxes_override=None):
        warnings.warn('DeprecationWarning: points_in_boxes is a '
                      'deprecated method, please consider using '
                      'points_in_boxes_part.')
        return self.points_in_boxes_part(points, boxes_override)

    def points_in_boxes_batch(self, points, boxes_override=None):
        warnings.warn('DeprecationWarning: points_in_boxes_batch is a '
                      'deprecated method, please consider using '
                      'points_in_boxes_all.')
        return self.points_in_boxes_all(points, boxes_override)


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of y.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 2

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3),
                     axis=1)).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners,
                                      self.tensor[:, 6],
                                      axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angles (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns
                None, otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction == 'vertical':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 1] = -points[:, 1]
                elif bev_direction == 'vertical':
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): the target Box mode
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`:
                The converted box of the same type in the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(box=self,
                                 src=Box3DMode.LIDAR,
                                 dst=dst,
                                 rt_mat=rt_mat)

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements,
                                  other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label)
                                                and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score)
                                                and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity)
               or (np.all(np.isnan(self.velocity))
                   and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(
            self.label, self.score, self.center[0], self.center[1],
            self.center[2], self.wlh[0], self.wlh[1], self.wlh[2],
            self.orientation.axis[0], self.orientation.axis[1],
            self.orientation.axis[2], self.orientation.degrees,
            self.orientation.radians, self.velocity[0], self.velocity[1],
            self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]],
                          color=color,
                          linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2],
                      linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0],
                  linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im, (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])), color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im, (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(
            im, (int(center_bottom[0]), int(center_bottom[1])),
            (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
            colors[0][::-1], linewidth)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


@OBJECT_REGISTRY.register
class testvisdataset(data.Dataset):

    def __init__(self,
                 ida_aug_conf,
                 bda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 is_train,
                 use_cbgs=False,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 return_depth=False,
                 sweep_idxes=list(),
                 key_idxes=list(),
                 use_fusion=False,
                 visual_imgs=False,
                 visual_save_path='nuscenes_visual_path'):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
            use_fusion (bool): Whether to use lidar data.
                default: False.
        """
        super().__init__()
        if isinstance(info_paths, list):
            self.infos = list()
            for info_path in info_paths:
                self.infos.extend(mmcv.load(info_path))
        else:
            self.infos = mmcv.load(info_paths)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) == len(
            sweep_idxes), 'All `sweep_idxes` must greater than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes
                    ]) == len(key_idxes), 'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes
        self.use_fusion = use_fusion

        self.visual_imgs = visual_imgs  # 对处理后的图像进行可视化
        if self.visual_imgs:
            self.visual_save_path = visual_save_path  # 处理后的图像可视化保存的路径
            self.ori_img = []
            self.post_img = []
            os.makedirs(self.visual_save_path, exist_ok=True)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW /
                         W)  # (256 / 900, 704 / 1600) = (0.28444444, 0.44)
            # resize = (704. / 1600., 256. / 900.)
            resize_dims = (int(W * resize), int(H * resize))
            # resize_dims = (704, 256)
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # crop = (0, 0, fW, fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth = map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)

    def get_image(self, cam_infos, cams, lidar_infos=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos:
                lidar_path = lidar_info['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)

        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            for sweep_idx, cam_info in enumerate(
                    cam_infos
            ):  # 之所以会有这个循环，主要因为存在2key的情况，如果是2key，这里会循环两次，否则，这就一次循环

                img = Image.open(
                    os.path.join(self.data_root,
                                 cam_info[cam]['filename']))  # 拿到camera的图像文件位置
                # img = cv2.imread(
                #     os.path.join(self.data_root, cam_info[cam]['filename'])) # 拿到camera的图像文件位置 改用opencv进行读取，和PIL不同的是通道顺序不一样
                if self.visual_imgs:  ########## 如果要进行可视化，这里会把图像装进来
                    self.ori_img.append(
                        np.array(img)[:, :, ::-1].astype(np.uint8))

                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor'][
                    'rotation']  # camera到车身的旋转
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)  # camera到车身的旋转
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']
                    ['translation'])  # camera到车身的平移
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3,
                                     -1] = sweepsensor2sweepego_tran  # camera到车身的旋转和平移的齐次矩阵
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']  # 车身到全局
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)  # 车身到全局的旋转矩阵
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros(
                    (4, 4))  # 车身到全局的平移矩阵
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3,
                                -1] = sweepego2global_tran  # 车身到全局的旋转和平移的齐次矩阵

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose'][
                    'rotation']  # 这个是key的信息 第0号信息 车身到全局的旋转
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)  # key的车身到全局的旋转
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])  # key的车身到全局的平移
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3,
                              -1] = keyego2global_tran  # key的信息 车身到全局的旋转和平移的齐次矩阵
                global2keyego = keyego2global.inverse()  # 全局到key的旋转和平移的齐次矩阵

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor'][
                    'rotation']  # 这个是key的信息 第0号信息 camera到车身的旋转
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y,
                               z).rotation_matrix)  # key的camera到key的车身的旋转
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']
                    ['translation'])  # key的camera到key的车身的平移
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3,
                                 -1] = keysensor2keyego_tran  # key的camera到key的车身的旋转和平移的齐次矩阵
                keyego2keysensor = keysensor2keyego.inverse(
                )  # key的车身到key的camera的旋转和平移的齐次矩阵
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego
                ).inverse()  # key的camera到非key的camera的矩阵
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego # 非key的camera到key的ego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                #img = mmcv.imnormalize(np.array(img), self.img_mean, self.img_std, self.to_rgb) # 此时变为BGR图像
                # img = mmcv.imnormalize(img, self.img_mean, self.img_std, self.to_rgb) # 这里的逻辑是先进性通道转换（如果需要的话），然后减均值，除以标准差 这里通道本就是opencv BGR，不用转了
                if self.visual_imgs:  ########## 如果要进行可视化，这里会把图像装进来
                    self.post_img.append(
                        mmcv.imdenormalize(img,
                                           self.img_mean,
                                           self.img_std,
                                           to_bgr=False).astype(np.uint8))
                """Currently we use old model for inference! Can be modified later."""
                if self.is_train:
                    img = np.array(img)
                    img = torch.from_numpy(img).permute(2, 0,
                                                        1).to(torch.uint8)
                else:
                    img = np.array(img)
                    img = torch.from_numpy(img).permute(2, 0,
                                                        1).to(torch.uint8)
                    # img = mmcv.imnormalize(np.array(img), self.img_mean, self.img_std, self.to_rgb) # 此时变为BGR图像
                    # img = torch.from_numpy(img).permute(2, 0, 1)
                #img = torch.from_numpy(img).permute(2, 0, 1)
                # img = img.permute(2,0,1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(
                torch.stack(sensor2ego_mats))  # sweep的camera到key的车身
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(
                torch.stack(sensor2sensor_mats))  # sweep的camera到key的camera
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
        # Get mean pose of all cams.
        if self.is_train:
            ego2global_rotation = np.mean(
                [key_info[cam]['ego_pose']['rotation'] for cam in cams],
                0)  # 以key为准求均值
            ego2global_translation = np.mean(
                [key_info[cam]['ego_pose']['translation'] for cam in cams],
                0)  # 以key为准求均值
            img_metas = dict(
                box_type_3d=LiDARInstance3DBoxes,
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
            )
        else:
            ego2global_rotation = np.mean(
                [key_info[cam]['ego_pose']['rotation'] for cam in cams],
                0)  # 以key为准求均值
            ego2global_translation = np.mean(
                [key_info[cam]['ego_pose']['translation'] for cam in cams],
                0)  # 以key为准求均值
            ego2global_rotation_corn = [
                key_info[cam]['ego_pose']['rotation'] for cam in cams
            ]
            ego2global_translation_corn = [
                key_info[cam]['ego_pose']['translation'] for cam in cams
            ]
            camera_intrinsic = [
                key_info[cam]['calibrated_sensor']['camera_intrinsic']
                for cam in cams
            ]
            file_name = [key_info[cam]['filename'] for cam in cams]
            cam_infos = key_info
            img_metas = dict(
                box_type_3d=LiDARInstance3DBoxes,
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                ego2global_rotation_corn=ego2global_rotation_corn,
                ego2global_translation_corn=ego2global_translation_corn,
                camera_intrinsic=camera_intrinsic,
                file_name=file_name,
                cam_infos=cam_infos)

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))

        if self.visual_imgs:  ################ 如果需要可视化，则在这里进行可视化
            width_interval = 10
            height_interval = 20
            width = self.ida_aug_conf['final_dim'][1]
            height = self.ida_aug_conf['final_dim'][0]
            ######### 写预处理后的图像
            post_big_img = np.zeros(
                (height + height_interval + height, width + width_interval +
                 width + width_interval + width, 3)) + 128

            post_big_img[0:height, 0:width, :] = self.post_img[0]
            post_big_img[0:height, width + width_interval:2 * width +
                         width_interval, :] = self.post_img[1]
            post_big_img[0:height, 2 * width + 2 * width_interval:3 * width +
                         2 * width_interval, :] = self.post_img[2]
            post_big_img[height + height_interval:,
                         0:width, :] = self.post_img[3]
            post_big_img[height + height_interval:,
                         width + width_interval:2 * width +
                         width_interval, :] = self.post_img[4]
            post_big_img[height + height_interval:,
                         2 * width + 2 * width_interval:3 * width +
                         2 * width_interval, :] = self.post_img[5]

            save_path = self.visual_save_path + os.sep + os.path.basename(
                cam_infos[0]["CAM_FRONT"]['filename'])  # 以前视camera去命名
            save_path = save_path[0:save_path.rfind(
                ".")] + "_post" + save_path[save_path.rfind("."):]
            cv2.imwrite(save_path, post_big_img.astype(np.uint8))

            ######### 写预处理后的图像
            H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
            fH, fW = self.ida_aug_conf['final_dim']
            resize = max(fH / H, fW /
                         W)  # (256 / 900, 704 / 1600) = (0.28444444, 0.44)
            width = int(self.ida_aug_conf['W'] * resize)
            height = int(self.ida_aug_conf['H'] * resize)
            post_big_img = np.zeros(
                (height + height_interval + height, width + width_interval +
                 width + width_interval + width, 3)) + 128

            post_big_img[0:height,
                         0:width, :] = cv2.resize(self.ori_img[0],
                                                  (width, height))
            post_big_img[0:height, width + width_interval:2 * width +
                         width_interval, :] = cv2.resize(
                             self.ori_img[1], (width, height))
            post_big_img[0:height, 2 * width + 2 * width_interval:3 * width +
                         2 * width_interval, :] = cv2.resize(
                             self.ori_img[2], (width, height))
            post_big_img[height + height_interval:,
                         0:width, :] = cv2.resize(self.ori_img[3],
                                                  (width, height))
            post_big_img[height + height_interval:,
                         width + width_interval:2 * width +
                         width_interval, :] = cv2.resize(
                             self.ori_img[4], (width, height))
            post_big_img[height + height_interval:,
                         2 * width + 2 * width_interval:3 * width +
                         2 * width_interval, :] = cv2.resize(
                             self.ori_img[5], (width, height))

            save_path = self.visual_save_path + os.sep + os.path.basename(
                cam_infos[0]["CAM_FRONT"]['filename'])  # 以前视camera去命名
            save_path = save_path[0:save_path.rfind(".")] + "_ori" + save_path[
                save_path.rfind("."):]
            cv2.imwrite(save_path, post_big_img.astype(np.uint8))

        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)  # box是车身下的坐标

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()

        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])
            lidar_sweep_timestamps = [
                lidar_sweep['LIDAR_TOP']['timestamp']
                for lidar_sweep in info['lidar_sweeps']
            ]
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                    lidar_infos.append(info['lidar_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx),
                                   -1, -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            cam_timestamp = np.mean([
                                val['timestamp']
                                for val in info['cam_sweeps'][i].values()
                            ])
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(lidar_sweep_timestamps -
                                               cam_timestamp).argmin()
                            lidar_infos.append(info['lidar_sweeps'][lidar_idx])
                            break
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)

        else:
            image_data_list = self.get_image(cam_infos, cams)
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = sweep_imgs.new_zeros(4, 4).float()
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ]
        if self.return_depth:
            ret_list.append(image_data_list[7])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)