import math
import numpy
"""
Created a simple multi-layer feed-forward neural net program with backpropigation, nnet, based off of what I know from differential geometry.
This can generalize neural nets to arbitrary manifolds.  
Also built a neural net that uses sin(x) instead of a sigmoid function in parts of the layers, circleNet, 
which might work well in producing functions that have some periodic properties.  
Could also use it as a sort of fourier transform.   

This program does not monitor the convergence or have great convergence techniques, but that wasn't the intention.
I mostly wanted to make sure that the differenctial geometry checks out.  The idea is simple:  A neural net
is just a bunch of functions composed.  You can tack an energy function into \mathbb{R} on the end, and then
you have a one parameter family of differentials \lambda*dx which you can pull back to each of the parameter
spaces to do a flow which will move the energy function towards a minima.  
"""

class nnet():
 def __init__(self, layerSizes):
  """ layerSizes is a list of the sizes of the layers, starting with the domain, 
  going through all of the hidden layers, and then ending in the codomain.
  Randomly initialize all matrices to have entries between -1 and 1.""" 
  self.ms = [ numpy.matrix(2.0*numpy.random.random([target, source])-1.0) for (source, target) in zip(layerSizes[:-1], layerSizes[1:]) ]
  self.layerSizes = layerSizes
 def sigmoid(self, y):
  """ y is a column matrix
  returns a column matrix  whose entries have all been evaluated at a sigmoid function """
  return numpy.matrix(map(math.atan, y)).T
 def dsigmoid(self, y):
  """ y should be a column matrix, returns an column matrix whose entries have all been evaluated at a the derivative of a sigmoid function """
  return 1.0/(1.0 + numpy.multiply(y,y))
 def error(self, y, t):
  return numpy.linalg.norm(y-t)**2
 def derror(self, y, t):
  return 2.0*(y - t)
 def evaluate(self,v):
  """ v is a column matrix of size self.layerSizes[0]
  Note that this neural net only outputs values between -pi/2 and pi/2, if the sigmoid is arctan. """ 
  return reduce(lambda x, m: self.sigmoid(m*x), self.ms, v)
 def train(self, trainIn, trainOut, delta):
  """ Trains only one sample.
   trainIn is the input training data as a column matrix
   trainOut is the output training data as a column matrix
   delta is the learning rate, i.e. the choice of a differential: delta * dy""" 
  # propigate data forward
  ys = self.scanl(lambda z, m: self.sigmoid(m*z), self.ms, trainIn)
  # generate error differential
  de = delta*self.derror(ys[-1], trainOut)
  # propigate error differential backwards
  dys = self.scanl(lambda y, k: self.pointCotangentMap(k[0], k[1], y), zip(self.ms, ys[:-1])[::-1] , de)
  # propigate error differential to matrix differentials
  dms = [ self.parameterCotangentMap(m, x, dy) for (m, x, dy) in zip(self.ms, ys[:-1], dys[-2::-1]) ] 
  # flow on matrices using matrix differentials
  self.ms = [ m-dm for (m, dm) in zip(self.ms, dms) ]
 def pointCotangentMap(self, m, x, dy):
  """ m: M -> N a matrix, x in M a column matrix, dy in T^*(N) a column matrix
  let f_m(x) = sigmoid(m*x)
  returns (f_m)^* at x evaluated at dy
  computed as the product of the cotangent maps from sigmoid and multiplication by m on the left. """
  return (m.T) * numpy.multiply( self.dsigmoid(m*x), dy )
 def parameterCotangentMap(self, m, x, dy):
  """ m: M -> N a matrix, x in M a column matrix, dy in T^*(N) a column matrix
   let f_x(m) = sigmoid(m*x)
   returns (f_x)^* at m evaluated at dy
   computed as the product of the cotangent maps from sigmoid and multiplication by x on the right.""" 
  return numpy.multiply(self.dsigmoid(m*x), dy) * (x.T)
 def scanl(self, f, xs, x0):
  """Why doesn't python have this?  It is like haskell's scanl.""" 
  tmp = x0
  result = [tmp]
  for x in xs:
   tmp = f(tmp, x)
   result.append(tmp)
  return result
 
class circleNet():
 def __init__(self, layerSizes, numSins):
  """ layerSizes is a list of the sizes of each layer.  numSins is a list of the number of sin parts to each sigmoid function. """
  #TODO fix instantiation and you should be done.  
  self.layerSizes = layerSizes 
  self.numSins = numSins
  ##print self.layerSizes, self.numSins
  self.ms = [ numpy.matrix(2.0*numpy.random.random([target, source])-1.0) for (source, target) in zip(self.layerSizes[:-1], self.layerSizes[1:]) ]
 def sigmoid(self, y, numSin):
  """ y is a column matrix, numSin is the number of components that have sin applied to them.  
  Note that the sin componets are on top. 
  returns a column matrix  whose entries have all been evaluated at a sigmoid function """
  sinVec, atanVec = numpy.vsplit(y, [numSin])
  return numpy.matrix(map(math.sin, sinVec) + map(math.atan, atanVec)).T
 def dsigmoid(self, y, numSin):
  """ y should be a column matrix, returns an column matrix whose entries have all been evaluated at a the derivative of a sigmoid function """
  if numSin == len(y):
   return numpy.matrix(map(math.cos, y)).T
  else:
   sinVec, atanVec = numpy.vsplit(y, [numSin])
   return numpy.vstack(( numpy.matrix(map(math.cos, sinVec)).T, 1.0 / (1.0 + numpy.multiply(atanVec, atanVec))))
 def error(self, y, t):
  return numpy.linalg.norm(y-t)**2
 def derror(self, y, t):
  return 2.0*(y - t)
 def evaluate(self, v):
  """ v is a column matrix of size self.layerSizes[0]
  Note that this neural net only outputs values between -pi/2 and pi/2, if the sigmoid is arctan. """
  return reduce(lambda x, k: self.sigmoid(k[0]*x, k[1]), zip(self.ms, self.numSins), v)
 def train(self, trainIn, trainOut, delta):
  """ Trains only one sample.
   trainIn is the input training data as a column matrix
   trainOut is the output training data as a column matrix
   delta is the learning rate, i.e. the choice of a differential: delta * dy"""
  # propigate data forward
  ys = self.scanl(lambda z, k: self.sigmoid(k[0]*z, k[1]), zip(self.ms, self.numSins), trainIn)
  # generate error differential
  de = delta*self.derror(ys[-1], trainOut)
  # propigate error differential backwards
  dys = self.scanl(lambda y, k: self.pointCotangentMap(k[0], k[1], k[2], y), zip(self.ms, self.numSins, ys[:-1])[::-1] , de)
  # propigate error differential to matrix differentials
  dms = [ self.parameterCotangentMap(m, numSin, x, dy) for (m, numSin, x, dy) in zip(self.ms, self.numSins, ys[:-1], dys[-2::-1]) ]
  # flow on matrices using matrix differentials
  self.ms = [ m-dm for (m, dm) in zip(self.ms, dms) ]
 def pointCotangentMap(self, m, numSin, x, dy):
  """ m: M -> N a matrix, x in M a column matrix, dy in T^*(N) a column matrix
  let f_m(x) = sigmoid(m*x)
  returns (f_m)^* at x evaluated at dy
  computed as the product of the cotangent maps from sigmoid and multiplication by m on the left. """
  return (m.T) * numpy.multiply( self.dsigmoid(m*x, numSin), dy )
 def parameterCotangentMap(self, m, numSin, x, dy):
  """ m: M -> N a matrix, x in M a column matrix, dy in T^*(N) a column matrix
   let f_x(m) = sigmoid(m*x)
   returns (f_x)^* at m evaluated at dy
   computed as the product of the cotangent maps from sigmoid and multiplication by x on the right."""
  return numpy.multiply(self.dsigmoid(m*x, numSin), dy) * (x.T)
 def scanl(self, f, xs, x0):
  """Why doesn't python have this?  It is like haskell's scanl."""
  tmp = x0
  result = [tmp]
  for x in xs:
   tmp = f(tmp, x)
   result.append(tmp)
  return result

if __name__=='__main__':
 import random
 n = circleNet( [ 2, 3, 3 ], [3,3])
 nsamples = 3 
 vs = [ numpy.matrix([2.0 * random.random() - 1.0 for i in range(n.layerSizes[0])]).T for j in range(nsamples) ]
 ts = [ numpy.matrix([2.0 * random.random() - 1.0 for i in range(n.layerSizes[-1])]).T for j in range(nsamples) ]
 for i in range(2000 * nsamples):
  n.train(vs[i % nsamples], ts[i % nsamples], 0.1)
 for i in range(nsamples):
  print i, n.error(n.evaluate(vs[i]), ts[i]) 
#TODO: make graph of learning rate vs number of training trials, n'stuff.  
# TODO: Make more generalizable to other manifolds.  Do circles first, as you know how to map in and out of them and they are groups.   
# Just replace arctan with sin some times and it should be equivalent.  
# TODO: A nnet that takes end values in a circle, but with energy function that tries to make it like the uniform distribution.   probably maximum entropy on the circle.  
# could make it complex too using e^(i * theta) and then use to process pictures and the like.  see if they can learn bare pots from marked ones, or handwriting.  
# Circle to circle maps could be rotations as the wrapping maps have a discrete parameter space.  
# TODO: make a neural net with a single end node that guesses the next sample in a time series by pulling back delta*dx.  
# It could be trained by pushing a difference tangent vector forward and then into the parameters. Try to model a markov process or finite state automata.   
# TODO:maybe:definitely: make energy function separate so we can edit with different energy functions, make pair: error(y,t), derror(y,t)
# if i can i should do some sort  of dynamic thing and have the values of the energy function feed into another layer. 
