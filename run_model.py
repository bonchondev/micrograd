#!/usr/bin/env python
# coding: utf-8

# ###  MicroGrad demo

# In[1]:
from sqlalchemy import select
from sqlalchemy.orm import Session
from micrograd.engine import Value
from micrograd.nn import MLP
from db import BigX, SmallY, engine
# initialize a model 
model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))

with Session(engine) as session:
    Xb = session.scalars(select(BigX)).all()
    y = session.scalars(select(SmallY)).all()
    yb = [item.value for item in y]


# loss function
def loss():
    
    # inline DataLoader :)
    inputs = [list(map(Value, [xrow.first_value,xrow.second_value])) for xrow in list(Xb)]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(scorei*-yi + 1).relu() for yi, scorei in zip(yb, scores)]
    data_loss: Value = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss: Value = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

total_loss, acc = loss()
print(total_loss, acc)

# In[7]:


# optimization
for k in range(100):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

