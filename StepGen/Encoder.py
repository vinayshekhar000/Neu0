import random
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, merge,Bidirectional
from keras.preprocessing import sequence
from keras.models import load_model

def test():
    while (True):
        try:
            inst = raw_input("Enter instruction: ")
            if inst == "QUIT":
                break
            elif inst == "SAVE":
                name=raw_input("Name: ")
                model.save(name+".h5")
                break
            inputs = np.array([encode_inst(inst)])
            inputs = sequence.pad_sequences(inputs, maxlen=20)
            predict = model.predict([inputs])
            for prediction in predict:
                print np.argmax(prediction),prediction
        except(Exception) as e:
            print e

train=raw_input("Train?")


num_regs = 15
symbols = list(map(chr, range(65, 91))) + map(str, range(0, 10)) + [",", " ", "#", "[", "]"] + ["*"]
print symbols
random.seed(12)


def encode_char(c):
    l = [0] * len(symbols)
    l[symbols.index(c)] = 1
    return l


def encode_inst(inst):
    encoded_inst = []
    num = False
    for c in inst:
        encoded_char = encode_char(c)
        encoded_inst.append(encoded_char)
        if c == "#":
            break
    return encoded_inst


op = "CMP"
ops = ["SUB", "ADD", "MUL", "MOV", "CMP", "B"]
conds = ["", "NE", "EQ", "GT", "LT"]

# samples = []
# outputs = []
before = []
for op in ops[:3]:
    for cond in conds:  # 5
        for r0 in range(0, num_regs - 1):  # 14
            for bracket in [True, False]:  # 2
                open = close = ""
                i_bracket = [1, 0]
                if bracket:
                    open = "["
                    close = "]"
                    i_bracket = [0, 1]
                for r1 in range(0, num_regs - 1):  # 14
                    for r2 in range(0, num_regs - 1):  # 14 : 27,440
                        if (r0 == r1 or r0 == r2):
                            continue
                        inst = op + cond + " R" + str(r0) + "," + open + "R" + str(r1) + close + ",R" + str(r2)
                        i_r2 = [0] * num_regs
                        i_r2[r2] = 1
                        # samples.append(inst)
                        i = [0] * len(ops)
                        i[ops.index(op)] = 1
                        i_cond = [0] * len(conds)
                        i_cond[conds.index(cond)] = 1
                        i_r0 = [0] * num_regs
                        i_r0[r0] = 1
                        i_r1 = [0] * num_regs
                        i_r1[r1] = 1
                        # outputs.append([i,i_r0,i_r1,i_r2])
                        branch_addr = 0
                        before.append((inst, [i, i_r0, i_r1, i_r2, i_cond, i_bracket]))

# For branch -Dataset creation
for j in range(215):
    for cond in conds: #5
        for r in range(num_regs-1): #14
            inst = "B" + cond + " R" +str(r)
            r_out = [0] * num_regs
            r_out[-1] = 1
            r2_out = [0] * num_regs
            r2_out[r] = 1
            i_inst = [0] * len(ops)
            i_inst[ops.index("B")] = 1
            i_cond = [0] * len(conds)
            i_cond[conds.index(cond)] = 1
            before.append((inst, [i_inst, r_out, r_out, r2_out, i_cond, [1, 0]]))

# For MOV -Dataset creation
op = "MOV"
for r0 in range(num_regs - 1):#14
    for cond in conds: #5
        for j in range(15): #15
            for r2 in range(0, num_regs - 1): #14
                inst = op + cond + " R" + str(r0) + "," + "R" + str(r2)
                i_r2 = [0] * num_regs
                i_r2[r2] = 1
                i_r1 = [0] * num_regs
                i_r1[-1] = 1
                i_r0 = [0] * num_regs
                i_r0[r0] = 1
                i = [0] * len(ops)
                i[ops.index(op)] = 1
                i_cond = [0] * len(conds)
                i_cond[conds.index(cond)] = 1
                before.append((inst, [i, i_r0, i_r1, i_r2, i_cond, [1, 0]]))

# TODO- Create conditions for mov and adjust the dataset size
# For CMP - Dataset Creation
op = "CMP"
for j in range(80):
    for r0 in range(num_regs - 1):
        for r2 in range(0, num_regs - 1):
            inst = op + " R" + str(r0) + "," + "R" + str(r2)
            i_r2 = [0] * num_regs
            i_r2[r2] = 1
            i_r1 = [0] * num_regs
            i_r1[-1] = 1
            i_r0 = [0] * num_regs
            i_r0[r0] = 1
            i = [0] * len(ops)
            i[ops.index(op)] = 1
            i_cond = [0] * len(conds)
            i_cond[conds.index("")] = 1
            before.append((inst, [i, i_r0, i_r1, i_r2, i_cond,[1, 0]]))

print "Samples generated"
random.shuffle(before)
samples = [x[0] for x in before]
outputs = [x[1] for x in before]
print len(samples), len(outputs)
samples = samples
outputs = outputs


def split(l, ratio):
    train_data = l[:int(ratio * len(l))]
    test_Data = l[int(ratio * len(l)):]
    return train_data, test_Data


inputs = []
inst_outputs = []
cond_outputs = []
r0_outputs = []
r1_outputs = []
r2_outputs = []
const_outputs = []
branch_outputs = []
bracket_outputs = []
for sample in samples:
    inputs.append(encode_inst(sample))
for out in outputs:
    inst_outputs.append(out[0])
    r0_outputs.append(out[1])
    r1_outputs.append(out[2])
    r2_outputs.append(out[3])
    cond_outputs.append(out[4])
    bracket_outputs.append(out[5])
print len(inputs)
print len(inst_outputs)

ratio = 0.7
inputs, e_inputs = split(inputs, ratio)
inst_outputs, e_inst = split(inst_outputs, ratio)
r0_outputs, e_r0 = split(r0_outputs, ratio)
r1_outputs, e_r1 = split(r1_outputs, ratio)
r2_outputs, e_r2 = split(r2_outputs, ratio)
cond_outputs, e_outputs3 = split(cond_outputs, ratio)
bracket_outputs, e_bracket = split(bracket_outputs, ratio)

inputs = np.array(inputs)
inputs = sequence.pad_sequences(inputs, maxlen=20)
inst_outputs = np.array(inst_outputs)
r0_outputs = np.array(r0_outputs)
r1_outputs = np.array(r1_outputs)
r2_outputs = np.array(r2_outputs)
cond_outputs = np.array(cond_outputs)
bracket_outputs = np.array(bracket_outputs)

e_inputs = np.array(e_inputs)
e_inputs = sequence.pad_sequences(e_inputs, maxlen=20)
e_inst = np.array(e_inst)
e_r0 = np.array(e_r0)
e_r1 = np.array(e_r1)
e_r2 = np.array(e_r2)
e_conds = np.array(e_outputs3)
e_bracket = np.array(e_bracket)

if train=="Y":
    inputl = Input(shape=(20, len(symbols)))
    lstm1 = LSTM(512, activation='relu', name='lstm1', return_sequences=True)(inputl)
    lstm = LSTM(256, activation='relu', name='lstm', return_sequences=False)(lstm1)
    instr = Dense(len(ops), activation='softmax', name='instr')(lstm)
    cond = Dense(len(conds), activation='softmax', name='cond')(lstm)
    r0 = Dense(num_regs, activation="softmax", name='r0')(lstm)
    r1 = Dense(num_regs, activation="softmax", name='r1')(lstm)
    r2 = Dense(num_regs, activation="softmax", name='r2')(lstm)
    bracket = Dense(2, activation='softmax', name='bracket')(lstm)
    import winsound

    Freq = 2500  # Set Frequency To 2500 Hertz
    Dur = 1000
    model = Model(input=[inputl], output=[instr, cond, r0, r1, r2, bracket])
    model.compile(optimizer='rmsprop',
                  loss=['categorical_crossentropy'] * 6,
                  metrics=['accuracy'])
    print model.summary()
    while (True):
        model.fit([inputs],
                  [inst_outputs, cond_outputs, r0_outputs, r1_outputs, r2_outputs, bracket_outputs],
                  batch_size=128,
                  nb_epoch=1)
        winsound.Beep(Freq, Dur)
        inp=raw_input("Test?")
        if inp=="Y":
            print model.evaluate([e_inputs], [e_inst, e_conds, e_r0, e_r1, e_r2, e_bracket])
        test()
        inp = raw_input("More epoch? Y?")
        if (inp != "Y"):
            break
else:
    model=load_model("encoder.h5")
    test()
