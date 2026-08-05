import numpy as np
import pandas as pd
from eta_digital.contexts.membership import TriangularMembership,TrapezoidalMembership
from eta_digital.contexts.context_model import ContextDefinition,ContextModel

def test_memberships_are_bounded():
    x=np.linspace(-2,12,100); assert np.all((TriangularMembership(0,5,10)(x)>=0)&(TriangularMembership(0,5,10)(x)<=1)); assert TrapezoidalMembership(0,2,8,10)(np.array([5.]))[0]==1

def test_context_weights_sum_to_one():
    model=ContextModel([ContextDefinition("a",{"x":TriangularMembership(0,5,10)}),ContextDefinition("b",{"x":TriangularMembership(5,10,15)})])
    weights=model.weights(pd.DataFrame({"x":[2,7,20]})); assert np.allclose(weights.sum(axis=1),1)
