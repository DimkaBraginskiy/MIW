import numpy as np
import time
import matplotlib.pyplot as plt

def updateMatrix(val1, val2):
    if(val1 == 'P'):
        if (val2 == 'P'):
            wyst[0,0]+=1
        if (val2 == 'St'):
            wyst[0, 1] += 1
        if (val2 == 'Sc'):
            wyst[0, 2] += 1
    if (val1 == 'St'):
        if (val2 == 'P'):
            wyst[1, 0] += 1
        if (val2 == 'St'):
            wyst[1, 1] += 1
        if (val2 == 'Sc'):
            wyst[1, 2] += 1
    if (val1 == 'Sc'):
        if (val2 == 'P'):
            wyst[2, 0] += 1
        if (val2 == 'St'):
            wyst[2, 1] += 1
        if (val2 == 'Sc'):
            wyst[2, 2] += 1

def winOrLoseState(our, op):
    return (
        (our == 'P' and op == 'Sc') or
        (our == 'Sc' and op == 'St') or
        (our == 'St' and op == 'P')
    )


# state matrix
t1 = ['P','St','Sc']
#matrix of repetition
wyst=np.array([[1,2,1], [0,1,0], [0,0,1]])
#oponent probability distribution - could be any
prob_op = np.array([1, 0, 0])
print(wyst)
print(wyst[0])
print(wyst[0]/sum(wyst[0]))
print(np.random.choice(t1, p=wyst[0]/sum(wyst[0])))

score = 0
scoreHistory = []

n = 30
state = 'P'

for i in range(n):
    if state == 'P':
        print('P state : --------------------------------')
        #prediction of oponent's movement
        pred = np.random.choice(t1, p=wyst[0]/sum(wyst[0]))
        print('predicted: ' + pred)
        print(wyst[0]/sum(wyst[0]))
        #map our prediction to our action
        if pred == 'P':
            our_move = 'Sc'
        elif pred == 'St':
            our_move = 'P'
        elif pred == 'Sc':
            our_move = 'St'
        #real movement of oponent 
        op_akc = np.random.choice(t1, p=prob_op)
        #update score acording to op_akc and our_move
        if winOrLoseState(op_akc, our_move):
            score += 1
        #update the matrix wyst based on real oponent movement - this is what we learn
        updateMatrix(state, op_akc)
        print(wyst)
        #go to state op_akc
        print('Opponent move: ' + op_akc)
        state = op_akc

    
    if state == 'St':
        print('St state : --------------------------------')
        # prediction of oponent's movement
        pred = np.random.choice(t1, p=wyst[1] / sum(wyst[1]))
        print('predicted: ' + pred)
        print(wyst[1] / sum(wyst[1]))
        # map our prediction to our action
        if pred == 'P':
            our_move = 'Sc'
        elif pred == 'St':
            our_move = 'P'
        elif pred == 'Sc':
            our_move = 'St'
        # real movement of oponent
        op_akc = np.random.choice(t1, p=prob_op)
        # update score acording to op_akc and our_move
        if winOrLoseState(op_akc, our_move):
            score += 1
        # update the matrix wyst based on real oponent movement - this is what we learn
        updateMatrix(state, op_akc)
        print(wyst)
        # go to state op_akc
        print('Opponent move: ' + op_akc)
        state = op_akc
        
  
    if state == 'Sc':
        print('Sc state : --------------------------------')
        pred = np.random.choice(t1, p=wyst[2] / sum(wyst[2]))
        print('predicted: ' + pred)
        print(wyst[2] / sum(wyst[2]))
        # map our prediction to our action
        if pred == 'P':
            our_move = 'Sc'
        elif pred == 'St':
            our_move = 'P'
        elif pred == 'Sc':
            our_move = 'St'
        # real movement of oponent
        op_akc = np.random.choice(t1, p=prob_op)
        # update score acording to op_akc and our_move
        if winOrLoseState(op_akc, our_move):
            score += 1
        # update the matrix wyst based on real oponent movement - this is what we learn
        updateMatrix(state, op_akc)
        print(wyst)
        # go to state op_akc
        print('Opponent move: ' + op_akc)
        state = op_akc

    scoreHistory.append(score)

    print("\n")
    time.sleep(0.1)

plt.plot(scoreHistory)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Score over time')
plt.show()


