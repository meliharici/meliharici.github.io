---
title: 'Text Generation using RNNs'
date: 2019-07-06
permalink: /posts/2019/07/blog-post-1/
tags:
  - Deep Learning
  - Artificial Intellegence
  - Recurrent Neural Networks
---

# Object Oriented Text Generation Using RNN
 In this post, I will be talking about an object oriented Python implementation of Recurrent Neural Networks (RNNs) on text generation field. RNNs are one of the popular neural network implementations in Deep Learning field. Specifically on time series data, RNNs are very powerful because of their memory-like structure that enables them to use past information for future decisions. Text generation is one of the popular implementations of RNNs and has extensively been studied in many tutorials so far. I tried to implement RNNs in an object oriented fashion in this post. Let's start!


```python
import numpy as np
```

**The Adventures of Sherlock Holmes by Arthur Conan Doyle** book used to train my network. Website: <br>
http://www.gutenberg.org/ebooks/1661  

#### Configurations 


```python
configurations = {}
configurations['NUM_HIDDEN_LAYERS'] = 150
configurations['STEP_SIZE'] = 500000
configurations['SEQUENTIAL_LENGTH'] = 25
configurations['LEARNING RATE'] = 1e-1
# this helps to compute gradients and the optimization is easier to solve:
configurations['LOSS_CONSTANT'] = 0.001
configurations['INPUT_NAME'] = 'holmes.txt'

```

**Preprocessor** is responsible for reading data, exctract unique characters and enumerate them. Based on sequence length, it gives a sequence of input to RNN. 


```python
class Preprocessor:

    def __init__(self, path):
        self.path = path
        self.dictionary = {}

    def read(self):
        with open(self.path, 'r', encoding='utf8') as file:
            self.data = file.read()
        print('{0} characters in total.'.format(len(self.data)))
        unique_chars = list(set(self.data))
        print('number of unique chars: {0}'.format(len(unique_chars)))
        configurations['NUM_UNIQUE_CHARS'] = len(unique_chars)
        return self.data

    def print(self):
        print(self.data)

    def convert(self, id):
        target_index = -1
        target_char = ''
        for i in range(len(id)):
            if id[i][0] == 1:
                target_index = i
        for char, id in self.dictionary.items():
            if id == target_index:
                target_char = char
        return target_char

    def enumerated_data(self):
        unique_chars = list(set(self.data))
        # id, char
        enumerated = enumerate(unique_chars)
        # dictionary to store the unique id of each char
        dictionary = {}
        for id, unique_char in enumerated:
            dictionary[unique_char] = id
        self.dictionary = dictionary
        enumerated_data = []
        for char in list(self.data):
            id = dictionary[char]
            enumerated_data.append(id)
        return enumerated_data

    def get_x_and_y(self, index):
        sequnce_length = configurations['SEQUENTIAL_LENGTH']
        enum_data = self.enumerated_data()
        x, y = [], []
        end_index = index + sequnce_length
        '''
        if end_index + 1 > len(enum_data):
            index = 0
            end_index = index + sequnce_length
        '''
        for i in range(index, end_index):
            x.append(enum_data[i])
        for i in range(index + 1, end_index + 1):
            y.append(enum_data[i])
        return x, y

    def data_size(self):
        return len(self.data)
```

**NetworkHelpers** method contains basic operations of neural network. These operations are one-hot encoding, calculating cross entropy loss and softmax function. 


```python
class NetworkHelpers:

    # softmax
    def logits(self, probabilities):
        return np.exp(probabilities) / np.sum(np.exp(probabilities))

    # loss for single 't'
    def xentropy_loss(self, logits, index):
        xentropy = -np.log(logits[index, 0])
        return xentropy

    def square(self, x):
        return x * x

    def one_hot(self, index, num_class):
        x = np.zeros((num_class, 1))
        x[index] = 1
        return x
```

**HiddenLayer** class represents one hidden layer of the network **at time "t"**. So, it is one of the vertical components of the rnn. For instance, if we have sequences of length 20, then we will have 20 hidden layers for t = 1,2,....,20. My definition of hidden layer can be understood by the following figure:

![title](deeprnn.png)


```python
class HiddenLayer:

    # Stack of Hidden Units that builds a Hidden Layer.

    def __init__(self, t):
        self.networkHelpers = NetworkHelpers()
        self.t = t
        self.x = np.zeros((configurations['NUM_UNIQUE_CHARS'], 1))
        self.h = np.zeros((configurations['NUM_HIDDEN_LAYERS'], 1))
        self.y = np.zeros((0, 0))
        self.p = np.zeros((0, 0))
        self.hidden_loss = np.zeros((0, 0))

    def set_input(self, hidden_input):
        self.x = hidden_input

    def get_input(self):
        return self.x

    def get_t(self):
        return self.t

    def get_h(self):
        return self.h

    def get_probs(self):
        return self.p

    def hidden_layer_out(self,  w_input_hidden, w_hidden_hidden, hiddenLayerOutPrev, hidden_bias):
        h = np.dot(w_input_hidden.get_matrix(), self.get_input()) + np.dot(w_hidden_hidden.get_matrix(), hiddenLayerOutPrev) + hidden_bias
        self.h = np.tanh(h)
        return self.h

    def hiddenY(self, w_hidden_output, hiddenLayerOut, output_bias):
        self.y = np.dot(w_hidden_output.get_matrix(), hiddenLayerOut) + output_bias
        return self.y

    def hiddenProbs(self, hiddenY):
        self.p = self.networkHelpers.logits(hiddenY)
        return self.p

    def loss(self, logits, input_index):
        self.hidden_loss = self.networkHelpers.xentropy_loss(logits, input_index)
        return self.hidden_loss
```

**Weight** class represent weights of recurrent nn. It contains basic matrix operations.


```python
class Weight:

    def __init__(self, shape):
        self.matrix = np.random.randn(shape[0], shape[1])
        self.scale(0.01)

    def scale(self, factor):
        self.matrix *= factor

    def add_gradient(self, gradient):
        self.matrix += gradient

    def shape(self):
        return self.matrix.shape

    def get_matrix(self):
        return self.matrix

    def transpose(self):
        return self.matrix.T
```

**RNN** is the main object of this code. Forward pass, backward pass, sampling and updating weights based on gradiens are the main operations of this object. "step" function is being called in each iteration and applies one forward and one backward pass. For the optimizer of RNN, Adagrad is the most used option as far as I searched. That's why, I also used it.


```python
class RNN:

    def __init__(self, hidden_layers):
        self.hp = NetworkHelpers()
        self.pr = Preprocessor(configurations['INPUT_NAME'])
        self.hidden_layers = hidden_layers
        self.total_loss = 0.0
        self.iteration_loss = -np.log(1.0 / configurations['NUM_UNIQUE_CHARS']) * configurations['SEQUENTIAL_LENGTH']
        self.init()
        self.reset()

    def init(self):
        # init weights
        self.w_hidden_hidden = Weight((configurations['NUM_HIDDEN_LAYERS'], configurations['NUM_HIDDEN_LAYERS']))
        self.w_hidden_output = Weight((configurations['NUM_UNIQUE_CHARS'], configurations['NUM_HIDDEN_LAYERS']))
        self.w_input_hidden = Weight((configurations['NUM_HIDDEN_LAYERS'], configurations['NUM_UNIQUE_CHARS']))
        # init biases
        self.bias_output = np.zeros((configurations['NUM_UNIQUE_CHARS'], 1))  # output bias
        self.bias_hidden = np.zeros((configurations['NUM_HIDDEN_LAYERS'], 1))  # hidden bias
        # init history of weights and biases will be used by the optimizer later.
        self.hist_input_hidden, self.history_hidden_hidden, self.history_hidden_output = np.zeros(self.w_input_hidden.shape()), np.zeros(self.w_hidden_hidden.shape()), np.zeros(self.w_hidden_output.shape())
        self.history_bias_hidden, self.history_bias_output = np.zeros_like(self.bias_hidden), np.zeros_like(self.bias_output)

    def add_to_weights(self, add_wxh, add_whh, add_why, add_bh, add_by):
        self.w_input_hidden.add_gradient(add_wxh)
        self.w_hidden_hidden.add_gradient(add_whh)
        self.w_hidden_output.add_gradient(add_why)
        self.bias_hidden += add_bh
        self.bias_output += add_by

    def add_to_memories(self, add_mwxh, add_mwhh, add_mwhy, add_mbh, add_mby):
        self.hist_input_hidden += add_mwxh
        self.history_hidden_hidden += add_mwhh
        self.history_hidden_output += add_mwhy
        self.history_bias_hidden += add_mbh
        self.history_bias_output += add_mby

    def set_pr(self, pr):
        self.pr = pr

    def reset(self):
        self.gradient_input_hidden = np.zeros((configurations['NUM_HIDDEN_LAYERS'],
                                               configurations['NUM_UNIQUE_CHARS']))
        self.gradient_hidden_hidden = np.zeros((configurations['NUM_HIDDEN_LAYERS'],
                                                configurations['NUM_HIDDEN_LAYERS']))
        self.gradient_hidden_output = np.zeros((configurations['NUM_UNIQUE_CHARS'],
                                                configurations['NUM_HIDDEN_LAYERS']))
        self.gradient_bias_hidden = np.zeros((configurations['NUM_HIDDEN_LAYERS'], 1))
        self.gradient_bias_output = np.zeros((configurations['NUM_UNIQUE_CHARS'], 1))
        self.gradient_next_hidden = np.zeros_like(self.hidden_layers[0].get_h())
        self.total_loss = 0.0

    def sample_sequence(self, hl, first_char, length):
        chars = []
        first_char_input = np.zeros((configurations['NUM_UNIQUE_CHARS'], 1))
        first_char_input[first_char] = 1
        chars.append(first_char_input)
        char = first_char_input
        for _ in range(length):
            next_char = self.predict_next(hl, char)
            chars.append(next_char)
            char = next_char
        result_text = ""
        # print('type char: {0}'.format(type(char)))
        for char in chars:
            result_text += self.pr.convert(char)
        print(result_text)

    def predict_next(self, hl, char):
        hl.set_input(char)
        hidden_layer_out = hl.hidden_layer_out(self.w_input_hidden, self.w_hidden_hidden, hl.get_h(), self.bias_hidden)
        hidden_layer_output = hl.hiddenY(self.w_hidden_output, hidden_layer_out, self.bias_output)
        probs = hl.hiddenProbs(hidden_layer_output)
        ix = np.random.choice(range(configurations['NUM_UNIQUE_CHARS']), p=probs.ravel())
        next_char = np.zeros((configurations['NUM_UNIQUE_CHARS'], 1))
        next_char[ix] = 1
        return next_char

    '''
    hl: hidden layer
    X: input sequence
    Y: target sequence
    '''
    def step_loss(self, hl, X, Y):
        x = np.zeros((configurations['NUM_UNIQUE_CHARS'], 1))
        x[X[hl.get_t()]] = 1
        hl.set_input(x)
        hidden_layer_out = hl.hidden_layer_out(self.w_input_hidden, self.w_hidden_hidden, self.hidden_layers[hl.get_t() - 1].get_h(), self.bias_hidden)
        hidden_layer_output = hl.hiddenY(self.w_hidden_output, hidden_layer_out, self.bias_output)
        probs = hl.hiddenProbs(hidden_layer_output)
        loss = hl.loss(probs, Y[hl.get_t()])
        return loss


    '''
    hl: hidden layer
    X: input sequence
    Y: target sequence
    '''
    def back_propagate(self, hl, X, Y):
        t = hl.get_t()
        # print('t: {0}'.format(t))
        dprobabilities = hl.get_probs()
        dprobabilities[Y[t]] -= 1
        self.gradient_hidden_output += np.dot(dprobabilities, hl.get_h().T)
        self.gradient_bias_output += dprobabilities
        self.gradient_hidden = np.dot(self.w_hidden_output.transpose(), dprobabilities) + self.gradient_next_hidden
        gradient_activation = (1 - hl.get_h() * hl.get_h()) * self.gradient_hidden
        self.gradient_bias_hidden += gradient_activation
        self.gradient_input_hidden += np.dot(gradient_activation, hl.get_input().T)
        self.gradient_hidden_hidden += np.dot(gradient_activation, self.hidden_layers[t - 1].get_h().T)
        self.gradient_next_hidden = np.dot(self.w_hidden_hidden.transpose(), gradient_activation)
        # to avoid vanishing gradients
        self.clip()

    def clip(self):
        for gradient in [self.gradient_input_hidden, self.gradient_hidden_hidden, self.gradient_hidden_output, self.gradient_bias_hidden, self.gradient_bias_output]:
            np.clip(gradient, -5, 5, out=gradient)

    def gradients(self):
        return [self.gradient_input_hidden, self.gradient_hidden_hidden, self.gradient_hidden_output, self.gradient_bias_hidden, self.gradient_bias_output]

    def histories(self):
        return [self.hist_input_hidden, self.history_hidden_hidden, self.history_hidden_output, self.history_bias_hidden, self.history_bias_output]

    def update_weights(self, gradients):
        hp = self.hp
        self.add_to_memories(hp.square(gradients[0]), hp.square(gradients[1]), hp.square(gradients[2]), hp.square(gradients[3]), hp.square(gradients[4]))
        gradient_adds = []
        for i in range(len(gradients)):
            gradient_adds.append(- configurations['LEARNING RATE'] * gradients[i] / np.sqrt(self.histories()[i] + 1e-8))
        self.add_to_weights(gradient_adds[0], gradient_adds[1], gradient_adds[2], gradient_adds[3], gradient_adds[4])

    # X = inputs,  Y = targets
    def step(self, X, Y, iteration):

        # sample
        if iteration % 1000 == 0:
            self.sample_sequence(self.hidden_layers[-1], X[0], 200)

        # t time sequences
        time_periods = len(X)

        loss = 0.0

        for t in range(time_periods):
            loss += self.step_loss(self.hidden_layers[t], X, Y)

        for t in range(1, time_periods + 1):
            self.back_propagate(self.hidden_layers[time_periods - t], X, Y)


        loss_constant_multiplier = configurations['LOSS_CONSTANT']
        self.iteration_loss = self.iteration_loss * (1 - loss_constant_multiplier) + loss * loss_constant_multiplier


        if iteration % 100 == 0:
            print('Iteration ({0}) - Loss: {1}'.format(iteration, self.iteration_loss))


        # update weights
        self.update_weights(self.gradients())

        self.reset()
```


```python
# read data
preprocessor = Preprocessor(configurations['INPUT_NAME'])
data = preprocessor.read()

# generate hidden time layers of rnn
hidden_layers = []
for t in range(configurations['SEQUENTIAL_LENGTH']):
    hidden_layers.append(HiddenLayer(t))

print('number of layers ( t ): {0}'.format(len(hidden_layers)))

# generate rnn
rnn = RNN(hidden_layers)
rnn.set_pr(preprocessor)
rnn.reset()

sequential_length = configurations['SEQUENTIAL_LENGTH']

data_index = 0
data_size = preprocessor.data_size()
num_iteration_at_epoch = data_size // sequential_length - 1
print('Number of iterations at each epoch: {0}'.format(num_iteration_at_epoch))

iteration_counter = 0
for i in range(configurations['STEP_SIZE']):

    if iteration_counter > num_iteration_at_epoch:
        iteration_counter = 0
        data_index = 0

    x, y = preprocessor.get_x_and_y(data_index)
    rnn.step(x, y, i)
    # update counters
    iteration_counter += 1
    data_index += sequential_length
    # print(x)
```

    561833 characters in total.
    number of unique chars: 77
    number of layers ( t ): 25
    Number of iterations at each epoch: 22472
    Av6iqT4UdTlowJaH/KS2hG3BcddkGgyMo3?yy&ADTd3MI/Clbac('i?CcSZ.V0cj:3I4cmgV"wcx"a(o nv,ky;f. CK,.l)W0ZBRS-UP4EPtnNqeX"Qi-e;AiY4IK9qAKjWjw,ceOy!OQFpgoi3CP.FD7e .gh.OMI-pKqxcdbH8 !s m6zE.Mbn6ihVd?o3rrUx/w:V
    Iteration (0) - Loss: 108.59513499761356
    Iteration (100) - Loss: 110.06245325211738
    Iteration (200) - Loss: 108.10469407414759
    Iteration (300) - Loss: 105.90874798623935
    Iteration (400) - Loss: 104.13724521691843
    Iteration (500) - Loss: 101.98095871403835
    Iteration (600) - Loss: 100.16210473013778
    Iteration (700) - Loss: 98.72254869495625
    Iteration (800) - Loss: 96.89326801608637
    Iteration (900) - Loss: 95.04767010613082
    hArs rdt  i tn,aetetl ho,uennro fllnki f uw mns wehidebwwt,anton ws nofyi dwhiseg ad-wIOhomn-  kdg dtt,vr'y"ciit as in  t wiendmtineuHfuaw wd at id d tf n n nnesIwinw mhkind snraBrdisag.'e  wule t td
    w
    Iteration (1000) - Loss: 93.37634400045272
    Iteration (1100) - Loss: 91.63443612541688
    Iteration (1200) - Loss: 90.18678511318709
    Iteration (1300) - Loss: 88.64368296092185
    Iteration (1400) - Loss: 87.0142240181531
    Iteration (1500) - Loss: 85.53605960040306
    Iteration (1600) - Loss: 84.04261356353992
    Iteration (1700) - Loss: 83.0196484894267
    Iteration (1800) - Loss: 82.04587106091253
    Iteration (1900) - Loss: 81.05648332764305
    m uue v, od pefer hhe atanilo seGharadownoaomve d the moole  f lfonntet aocke
    ko
     sqf
    dendt, afon dyintothe thden ponhee tafu c aapov aen I ngpoin e , am amao"aped thiod h 
    et npo w me Ged asgot haaf t
    Iteration (2000) - Loss: 79.79411753952787
    Iteration (2100) - Loss: 78.84033531239669
    Iteration (2200) - Loss: 78.28437228181772
    Iteration (2300) - Loss: 77.24039856432466
    Iteration (2400) - Loss: 76.13243451070247
    Iteration (2500) - Loss: 75.19515327304907
    Iteration (2600) - Loss: 74.42412058656272
    Iteration (2700) - Loss: 73.72086854534766
    Iteration (2800) - Loss: 73.12577988282655
    Iteration (2900) - Loss: 72.46237135848087
    e s
    '"g or aod nou ran 
    unt is, onuye inder the iok ay masgth yomyakgide ato"te" 
    oy mip fand yeemay an he Long lN sirtd" tfou sir ysua, vhins hoI iye aa. Solmas Hurle go bous , had ad er tly, nour ake
    Iteration (3000) - Loss: 71.90758065263248
    Iteration (3100) - Loss: 71.24913880619388
    Iteration (3200) - Loss: 70.58604368687496
    Iteration (3300) - Loss: 70.10540231091727
    Iteration (3400) - Loss: 69.5620558514523
    Iteration (3500) - Loss: 69.04486221259786
    Iteration (3600) - Loss: 68.46575923207031
    Iteration (3700) - Loss: 68.08693969990676
    Iteration (3800) - Loss: 67.32297617904301
    Iteration (3900) - Loss: 67.11862266345007
    torel t hy ereintary to thear. cf gf whe
    csamed of hacof uh f erimwreverd
    thay redishens cobeiene
    cicy
    a
    sr is rnithpenitth
    remigle fistmath rics ir mase theronetphecremof ithalaren
    in. atireicrst, the
    Iteration (4000) - Loss: 66.56007546391174
    Iteration (4100) - Loss: 66.19355236750775
    Iteration (4200) - Loss: 65.85346070175584
    Iteration (4300) - Loss: 65.32257079964094
    Iteration (4400) - Loss: 64.79757297424665
    Iteration (4500) - Loss: 64.33715180543959
    Iteration (4600) - Loss: 63.919697988214935
    Iteration (4700) - Loss: 63.78363410842953
    Iteration (4800) - Loss: 63.64170275892246
    Iteration (4900) - Loss: 63.35763785476804
    cereres. Mal hereresd" anqe sopor. Whivl,or thideer qnet iu ofoo sint ratle hapyrameeverire at 2erlerkeand rewcor,
     rore daid hoto-re fuand,
    as ble tout therr herin ane
    
    
    "d co theul?"
    
    "harw indlsoute
    Iteration (5000) - Loss: 63.14249218820869
    Iteration (5100) - Loss: 62.86094098048672
    Iteration (5200) - Loss: 62.52851707763194
    Iteration (5300) - Loss: 62.240499831416194
    Iteration (5400) - Loss: 62.3683408441632
    Iteration (5500) - Loss: 62.2068455511785
    Iteration (5600) - Loss: 61.93748829968843
    Iteration (5700) - Loss: 61.78198910683419
    Iteration (5800) - Loss: 61.51060676248748
    Iteration (5900) - Loss: 61.27484233198334
    e ireowe
    dolpe son, whe uning er mad. Iblomag be ofel:,"I hyre warly serouse ?aDonet ef an'dof boy the par sape beryoanw'''
    
    ""Whe soide'"
    
    "Ther. Ton, Ielt ory'y'o-. sipad om toy ofed'
    "A'd los beb yo
    Iteration (6000) - Loss: 61.06665525616778
    Iteration (6100) - Loss: 60.98376716935855
    Iteration (6200) - Loss: 60.849373890320194
    Iteration (6300) - Loss: 60.64537179282313
    Iteration (6400) - Loss: 60.47538961017566
    Iteration (6500) - Loss: 60.10601064738209
    Iteration (6600) - Loss: 60.090912231819075
    Iteration (6700) - Loss: 59.82443332616919
    Iteration (6800) - Loss: 59.52032758506661
    Iteration (6900) - Loss: 59.508264523870956
    wor the Le heiglely im
    the yDof al che."
     Broutt imerorf irs oust, in
    us aur int bog fouls' hal orteme she robif o
    cucosmisrisivasomoghe
    Hs mereroulne mered. I -ore shet anpis. tBowidkin?, on Somy."
    "B
    Iteration (7000) - Loss: 59.439626243455436
    Iteration (7100) - Loss: 59.33949676056015
    Iteration (7200) - Loss: 59.21101629559594
    Iteration (7300) - Loss: 59.03166400803833
    Iteration (7400) - Loss: 58.861365135172015
    Iteration (7500) - Loss: 58.97499525697226
    Iteration (7600) - Loss: 58.78364304322691
    Iteration (7700) - Loss: 58.73009346081587
    Iteration (7800) - Loss: 58.71507168149454
    Iteration (7900) - Loss: 58.539163125619766
    nd seaint aclime Ther b
    thed hat the
    palken.
    "'Ssatlen the
    teratar wat and ther, wa reas, gortuch to
    ghas for, in. theet andeve
    lave waf in withres Iprinen
    
    of serepous as
    eOd mere 8erd an.
    
    
    Thill ind
    Iteration (8000) - Loss: 58.46556438776454
    Iteration (8100) - Loss: 58.422705818810925
    Iteration (8200) - Loss: 58.273490840403284
    Iteration (8300) - Loss: 58.18663508930594
    Iteration (8400) - Loss: 58.01944085591771
    Iteration (8500) - Loss: 58.158035718904486
    Iteration (8600) - Loss: 58.007193578538875
    Iteration (8700) - Loss: 58.04386820138211
    Iteration (8800) - Loss: 58.01240133347546
    Iteration (8900) - Loss: 57.97308324058208
    at angand ebase, I I mby var tharinVy, on shat waucd whre pcerob W thotnelthao hed fol wrech
    i
    ing wis pill.
    
    
    "Whis we werere That of ThitDes
    gal u as. Gesmid th hill, was thine fas her, to Sbe rotas 
    Iteration (9000) - Loss: 58.01084683835042
    Iteration (9100) - Loss: 58.023911436751554
    Iteration (9200) - Loss: 57.91018999786178
    Iteration (9300) - Loss: 57.92583260092568
    Iteration (9400) - Loss: 57.807660596174046
    Iteration (9500) - Loss: 57.83042656412816
    Iteration (9600) - Loss: 57.95120560440679
    Iteration (9700) - Loss: 57.69050724675623
    Iteration (9800) - Loss: 57.47367957981443
    Iteration (9900) - Loss: 57.258917506338975
     and elteres thenklintalthuseeve a hingatven as whans iy aon herenurnowe beal faf me
    then indthoen benis bothat wton oby bu. Theinglemlhanases
    if upakeben itthep."
    
    hird wo beeed soigren ant thease ald
    Iteration (10000) - Loss: 56.99256183222765
    Iteration (10100) - Loss: 56.9199367235505
    Iteration (10200) - Loss: 56.79894084826174
    Iteration (10300) - Loss: 56.72949206662789
    Iteration (10400) - Loss: 56.601422120253346
    Iteration (10500) - Loss: 56.48551326650133
    Iteration (10600) - Loss: 56.446489634562894
    Iteration (10700) - Loss: 56.45508279926159
    Iteration (10800) - Loss: 56.376266975595776
    Iteration (10900) - Loss: 56.344271106552384
    ,
    I arue fiin,
    int is leifersrest, hith acses th
    is be
    lcichver sadesthirykinclitat tiin I Tithes whathen dolt his our awo fort."
    
    "Now merycen riin on venis thanithalk pott th in annistid an
    hirk.
    
    "Y
    Iteration (11000) - Loss: 56.475379030313285
    Iteration (11100) - Loss: 56.48186115467722
    Iteration (11200) - Loss: 56.37169356501144
    Iteration (11300) - Loss: 56.25958852301443
    Iteration (11400) - Loss: 56.07887637826713
    Iteration (11500) - Loss: 56.23170195859434
    Iteration (11600) - Loss: 56.21696604854362
    Iteration (11700) - Loss: 56.10192327998588
    Iteration (11800) - Loss: 56.153195317087125
    Iteration (11900) - Loss: 56.316992882334254
    wricheo by Mts puou he'ky nour ore, rocs upterensektty momme win, I I your! Wellas thoh a diss so mas th't a sint."
    
    "Sas-lat and a wat hu hem the liid gase efen?' holebeded,'gheul the by. Wfare doo fo
    Iteration (12000) - Loss: 56.2875291396421
    Iteration (12100) - Loss: 56.37725359105117
    Iteration (12200) - Loss: 56.21010905773266
    Iteration (12300) - Loss: 56.13998874253491
    Iteration (12400) - Loss: 56.15884987456366
    Iteration (12500) - Loss: 55.850650364945714
    Iteration (12600) - Loss: 55.67669412002651
    Iteration (12700) - Loss: 55.91167424174099
    Iteration (12800) - Loss: 55.83433865571892
    Iteration (12900) - Loss: 55.72754770903091
     carlge mthsare Sor meordedtin grall-to fat foabe Ro-no nour thest-ot heocrle
    loa-vin-of at  of a quebonsid ouy to sted my agl yHrame mowabe ant
    fronamw as
    bter ucomy goltommave -horted tto loone torgo
    Iteration (13000) - Loss: 55.914586072008376
    Iteration (13100) - Loss: 55.89422572700567
    Iteration (13200) - Loss: 55.80632575387392
    Iteration (13300) - Loss: 55.63712438258629
    Iteration (13400) - Loss: 55.48880606633484
    Iteration (13500) - Loss: 55.270017006899494
    Iteration (13600) - Loss: 55.082254865594585
    Iteration (13700) - Loss: 55.22430176179597
    Iteration (13800) - Loss: 55.15805684864708
    Iteration (13900) - Loss: 54.99156441662983
     wo dilrthich wid ables. Thoud bowe owen inosoin andedile yow hend nack hor threg me but by the thenthy thite is he hith
    the. Tele wos the rothe thend waone, sfardilled pand in hienadt speelokes
    u pron
    Iteration (14000) - Loss: 54.85368565435314
    Iteration (14100) - Loss: 54.808024592820786
    Iteration (14200) - Loss: 54.57850762486522
    Iteration (14300) - Loss: 54.44445668367123
    Iteration (14400) - Loss: 54.44559669121064
    Iteration (14500) - Loss: 54.325641535450444
    Iteration (14600) - Loss: 54.314957744753215
    Iteration (14700) - Loss: 54.30771596747091
    Iteration (14800) - Loss: 54.532598263534574
    Iteration (14900) - Loss: 54.63390689304652
    havt, Agrimaicesyy a8s a
    tith tha mapbaond. sheke'd and supverdamy."
    
    "Which th os boucom walk wand to tiend inn do ouping the theen ronging, oxs
    eve fis it thon iintess whrel," hin overer mowp busk th
    Iteration (15000) - Loss: 54.676754395270116
    Iteration (15100) - Loss: 54.64553138102609
    Iteration (15200) - Loss: 54.774832206659894
    Iteration (15300) - Loss: 54.784829734058604
    Iteration (15400) - Loss: 54.73493048371352
    Iteration (15500) - Loss: 54.65109064178653
    Iteration (15600) - Loss: 54.41322346060721
    Iteration (15700) - Loss: 54.388764834191015
    Iteration (15800) - Loss: 54.34700367747244
    Iteration (15900) - Loss: 54.223297416086545
    tann pol yowthit upile, teaydd ther inted ming mthe to nin Ik thate lait oy I my demand the cank.'
    
    "The hpulked coble the prew hound the tsit ous ard yof pord of
    we it in vand, shild the the he fryint
    Iteration (16000) - Loss: 54.04669729171812
    Iteration (16100) - Loss: 53.913586004868904
    Iteration (16200) - Loss: 53.78768444836985
    Iteration (16300) - Loss: 53.98564892744103
    Iteration (16400) - Loss: 53.94014242209811
    Iteration (16500) - Loss: 53.952460386546434
    Iteration (16600) - Loss: 54.30249687790801
    Iteration (16700) - Loss: 54.382627494447604
    Iteration (16800) - Loss: 54.666610328779434
    Iteration (16900) - Loss: 54.78794899809115
    g be theclo norl i of tef lasoden Pirisce mudear ont on beernessece intiand
    you tse jrigasene ofrouppo hordt
    u whighend came if and mesesthowert alles ascaln?"
    
    OV and the orang sheotture for cond the 
    Iteration (17000) - Loss: 54.73009111254472
    Iteration (17100) - Loss: 54.70712307543883
    Iteration (17200) - Loss: 54.61422944270721
    Iteration (17300) - Loss: 54.375074597502085
    Iteration (17400) - Loss: 54.29562225653532
    Iteration (17500) - Loss: 54.22314804350849
    Iteration (17600) - Loss: 54.14283180962308
    Iteration (17700) - Loss: 54.1933088152519
    Iteration (17800) - Loss: 54.17309960914689
    Iteration (17900) - Loss: 54.208724549189384
    rk Sookipaave wer it iy by hivand ourd wads," seasiakt tfe the his one the Lowd hithime;
    pliskr, wholn herce tardomike of him hat wisin. I jeat at
    ut.
    I him has the firens. Ine the wed Ind hering doid 
    Iteration (18000) - Loss: 54.031860536049344
    Iteration (18100) - Loss: 53.83136827470842
    Iteration (18200) - Loss: 53.77506928558524
    Iteration (18300) - Loss: 53.8249098099709
    Iteration (18400) - Loss: 54.059729862812475
    Iteration (18500) - Loss: 54.04630853225515
    Iteration (18600) - Loss: 54.058905741410754
    Iteration (18700) - Loss: 53.91661241823939
    Iteration (18800) - Loss: 53.90695320136703
    Iteration (18900) - Loss: 53.8540208555319
     a have my as , rastind, buwgep, the
    thend, your and frou sufw depf, to hocmy.' AAgticped I my inuching, bs thin
    day-," sor!com-hict. He sing of hist sabtelces invave gercompind alosdith parsstilk le. 
    Iteration (19000) - Loss: 53.65927894340873
    Iteration (19100) - Loss: 53.33734056653707
    Iteration (19200) - Loss: 53.23675946873569
    Iteration (19300) - Loss: 53.245563492738704
    Iteration (19400) - Loss: 53.16482393960648
    Iteration (19500) - Loss: 52.98571344640662
    Iteration (19600) - Loss: 52.93945785733341
    Iteration (19700) - Loss: 52.9245745861811
    Iteration (19800) - Loss: 52.79891108941057
    Iteration (19900) - Loss: 52.884222022410675
    d heon end frot andaw syed his thing bed m bele and wor wuteed thallney, st!" whigh po to thove is
    ihss. You weredurk? yeesed,
    prinkid sto plesery he so lat hesid on honfel soss of sest kothat whe.
    
    "A
    Iteration (20000) - Loss: 52.79517503780511
    Iteration (20100) - Loss: 52.66653059346988
    Iteration (20200) - Loss: 52.42180118051421
    Iteration (20300) - Loss: 52.4214985786372
    Iteration (20400) - Loss: 52.881950628218426
    Iteration (20500) - Loss: 52.91125166193463
    Iteration (20600) - Loss: 53.37185655141102
    Iteration (20700) - Loss: 53.4562615988137
    Iteration (20800) - Loss: 53.554119246482124
    Iteration (20900) - Loss: 53.71048143190373
    talss esourded
    dthe ar-himpaes
    "Thout huy wat
    ald, in kooke
    be caidp houke a meres my
    I be rony fma he ttould shever."
    
    "Ithe hern.
    
    "This
    ittiode you ly
    ay alk
    os
    Patcaus you I mrossinby wyou my, I ca
    Iteration (21000) - Loss: 53.6529897054109
    Iteration (21100) - Loss: 53.708605622711836
    Iteration (21200) - Loss: 53.77825658314363
    Iteration (21300) - Loss: 53.64776951887084
    Iteration (21400) - Loss: 53.52592818884828
    Iteration (21500) - Loss: 53.38739117336511
    Iteration (21600) - Loss: 53.339301141493216
    Iteration (21700) - Loss: 53.2926003482122
    Iteration (21800) - Loss: 53.018810698892246
    Iteration (21900) - Loss: 52.78121897698093
    her to ser m, haed.
    
    I dithery wave I mo. I as aln't I evever highinged feal,'re-fins was' is the wam om cour rhatiadt misks wnapetuk the perter, rook
    At tale trere fa kut and I to adered a the
    toled w
    Iteration (22000) - Loss: 52.61546468021404
    Iteration (22100) - Loss: 52.532843309181224
    Iteration (22200) - Loss: 52.51338501550218
    Iteration (22300) - Loss: 52.5533840775661
    Iteration (22400) - Loss: 52.44709120182766
    Iteration (22500) - Loss: 52.7398667118083
    Iteration (22600) - Loss: 52.982125339036344
    Iteration (22700) - Loss: 53.027812597045255
    Iteration (22800) - Loss: 53.00429930429926
    Iteration (22900) - Loss: 53.202881972384006
    , greie beor. Hule than
    amaneprle bedant-sust pomeiso-case wtaikre baoly of mes the berqumeywonnt potire's tale ibtome or?"
    joud, Heasearozey thes on ougwind seare, to hade frame cimenang."
    
    I seans, i
    Iteration (23000) - Loss: 53.15646537903676
    Iteration (23100) - Loss: 53.54275705148411
    Iteration (23200) - Loss: 53.425723060015066
    Iteration (23300) - Loss: 53.349551717926175
    Iteration (23400) - Loss: 53.329388294841216
    Iteration (23500) - Loss: 53.33559551962336
    Iteration (23600) - Loss: 53.29480388636828
    Iteration (23700) - Loss: 53.01484373805625
    Iteration (23800) - Loss: 53.12716854275169
    Iteration (23900) - Loss: 53.00303454569842
    tome ax. Buttolme wis had orded bmady fomm aige, a mould dolmence. Tharn coo."
    
    I whousd Is of hindy iced the mmear.
    PThe
    he wimed a po-my."
    
    "I mitten
    wane do
    neep-are Soldenofows, a dangcacmed, I mis
    Iteration (24000) - Loss: 52.74160837233282
    Iteration (24100) - Loss: 52.665622284063545
    Iteration (24200) - Loss: 52.82523250370155
    Iteration (24300) - Loss: 53.20198891658341
    Iteration (24400) - Loss: 53.350588310566195
    Iteration (24500) - Loss: 53.33231955155621
    Iteration (24600) - Loss: 53.489392273768686
    Iteration (24700) - Loss: 53.67122887416047
    Iteration (24800) - Loss: 53.516841513852135
    Iteration (24900) - Loss: 53.307163904190816
    nes. reensid be thay prioudt. Cor apend tat in that lamqferatfill, bo calling to Eithy, a. "'I grien "hastely vad 'nfidre.
    
    O''thteclat food, bolad don es, of tol mtilo,' itecess, Jolm, whe had I peacu
    Iteration (25000) - Loss: 53.23207184351436
    Iteration (25100) - Loss: 53.12832170929614
    Iteration (25200) - Loss: 53.40542740686829
    Iteration (25300) - Loss: 53.383465604151304
    Iteration (25400) - Loss: 53.2562042757415
    Iteration (25500) - Loss: 53.28955656403411
    Iteration (25600) - Loss: 53.362380829662484
    Iteration (25700) - Loss: 53.34808748743969
    Iteration (25800) - Loss: 53.29483101281913
    Iteration (25900) - Loss: 53.20859863261514
    d chood thin wure cod kncore, then the wolrer in gatimesed, fris to tesl. Thay of ghertato
    g that, mave duangive whur wime tar show onte beatury exif at net! syight id. Fent chels mad the wursent. You 
    Iteration (26000) - Loss: 53.08270190259872
    Iteration (26100) - Loss: 52.94644465233609
    Iteration (26200) - Loss: 52.825380771914794
    Iteration (26300) - Loss: 52.7011487158309
    Iteration (26400) - Loss: 52.968454899347144
    Iteration (26500) - Loss: 52.84450926381657
    Iteration (26600) - Loss: 52.80032351336464
    Iteration (26700) - Loss: 52.724732545351934
    Iteration (26800) - Loss: 52.439307251055276
    Iteration (26900) - Loss: 52.19024173303743
    ald mise thie hery in all ming? Hury, wive my reat wave reester chms unhed the was dot was my.
    "You
    hemathe to of tteird, what it perees havy intbeame in shevery yooedmerjadimenn hill in wabl
    the d oni
    Iteration (27000) - Loss: 51.90759326145922
    Iteration (27100) - Loss: 51.855157628963156
    Iteration (27200) - Loss: 51.95022455234472
    Iteration (27300) - Loss: 52.14130120985913
    Iteration (27400) - Loss: 52.101281821743946
    Iteration (27500) - Loss: 52.12379863350411
    Iteration (27600) - Loss: 52.01108420269622
    Iteration (27700) - Loss: 51.903950865951764
    Iteration (27800) - Loss: 51.84901851783128
    Iteration (27900) - Loss: 52.245952417247345
    e who mise. If mive. shas may oses he. Motor he
    st base ous more for. T" he Antl?"
    
    Mo
    mather cqaton the the, rored tr. Hous. He the soon and "Thablichia toller-exerittind dour at and
    knam, foo site ge
    Iteration (28000) - Loss: 52.356232925745786
    Iteration (28100) - Loss: 52.26857687443346
    Iteration (28200) - Loss: 52.30680172575464
    Iteration (28300) - Loss: 52.184669268076235
    Iteration (28400) - Loss: 52.206221580594004
    Iteration (28500) - Loss: 51.96849203216348
    Iteration (28600) - Loss: 52.04101288453506
    Iteration (28700) - Loss: 51.92605418563335
    Iteration (28800) - Loss: 51.83644573194356
    Iteration (28900) - Loss: 51.73158323666828
    de
    low yound lene
    fith
    Stheng the whercaing, aodeln quving your plevllougher of androuke tist siok. Cat, mxsur fore faalle reatong. He ncer
    the stornedd, whith teen. Onad a leit.
    
    "therdesteast of a bi
    Iteration (29000) - Loss: 51.64730008381188
    Iteration (29100) - Loss: 51.66544041406171
    Iteration (29200) - Loss: 51.607917998068565
    Iteration (29300) - Loss: 51.39728816269829
    Iteration (29400) - Loss: 51.52805230167254
    Iteration (29500) - Loss: 51.5922300863619
    Iteration (29600) - Loss: 51.52920024402129
    Iteration (29700) - Loss: 51.51201809651703
    Iteration (29800) - Loss: 51.35242601271424
    Iteration (29900) - Loss: 51.524458170567485
    e with hat ratire wexarncat, whid sfoudsire whes mistire notl of to falk lene trece forlres the dinupableld helder hele ours mach iit Vas the, oen thither. I wiline wath sofulpe Choine
    thing wevapret a
    Iteration (30000) - Loss: 51.59438818883261
    Iteration (30100) - Loss: 51.42617561281717
    Iteration (30200) - Loss: 51.51915802182362
    Iteration (30300) - Loss: 51.62218394800561
    Iteration (30400) - Loss: 51.516433414986
    Iteration (30500) - Loss: 51.60255325302106
    Iteration (30600) - Loss: 51.55238141389903
    Iteration (30700) - Loss: 51.48258679576616
    Iteration (30800) - Loss: 51.497836639842276
    Iteration (30900) - Loss: 51.52480255317022
     drerely to serteres elo-nonly of lery of alyery your armanny reack the foris sainet and the cother the clase mas thit that thestultely
    to resennce wiy the ayel evas this. Wrom aDe to qucebee of same. 
    Iteration (31000) - Loss: 51.7401160613568
    Iteration (31100) - Loss: 51.87218964403046
    Iteration (31200) - Loss: 51.86018976647348
    Iteration (31300) - Loss: 51.858145840157306
    Iteration (31400) - Loss: 51.850593452996875
    Iteration (31500) - Loss: 52.113597831814815
    Iteration (31600) - Loss: 52.07592157890001
    Iteration (31700) - Loss: 52.17717337497132
    Iteration (31800) - Loss: 52.16813873382917
    Iteration (31900) - Loss: 52.16601200383595
     to wies of walwis I sheres shere my."
    
    "Golling he age his!," therat I, tisk you dore my lack you, whe oo tiske aalllibe youre mong tithoustiove tomat wnigack of leabwere, the
    loca, a poles, she the s
    Iteration (32000) - Loss: 52.17162856141818
    Iteration (32100) - Loss: 52.2818454709924
    Iteration (32200) - Loss: 52.0535602023822
    Iteration (32300) - Loss: 51.956802528374084
    Iteration (32400) - Loss: 51.75234651684854
    Iteration (32500) - Loss: 51.46465549450941
    Iteration (32600) - Loss: 51.4540843217106
    Iteration (32700) - Loss: 51.29895422257442
    Iteration (32800) - Loss: 51.18965345049846
    Iteration (32900) - Loss: 51.20895797927693
    ?"
    
    "I tinked and yout and she
    terquinkdeaterind. Ay at gelr."
    
    "I."
    
    "Bun. He have pro ditrowh hour to
    shan wers iver Lee Lesion Smmackingas. I eask tho vear she feven, beet-for as. It hin my lals the
    Iteration (33000) - Loss: 51.19430531261058
    Iteration (33100) - Loss: 51.25856945823939
    Iteration (33200) - Loss: 51.260850041427716
    Iteration (33300) - Loss: 51.342503246522206
    Iteration (33400) - Loss: 51.2805902860513
    Iteration (33500) - Loss: 51.36600025434846
    Iteration (33600) - Loss: 51.4353284252791
    Iteration (33700) - Loss: 51.36900828177911
    Iteration (33800) - Loss: 51.19881969441012
    Iteration (33900) - Loss: 51.20864615385022
     the catupton, is asd.
    "Whad whise promossing ase oprevery whighint pryeruor and
    Bn condy ant nount Ifsed anmy onseousts."
    
    "Snedyswast and
    nisten nast ts is to of the rectlpalreystares, papess the Cor
    Iteration (34000) - Loss: 51.338264918054364
    Iteration (34100) - Loss: 51.35764066685255
    Iteration (34200) - Loss: 51.34853486024919
    Iteration (34300) - Loss: 51.386563231823835
    Iteration (34400) - Loss: 51.652711876525004
    Iteration (34500) - Loss: 51.57116400993098
    Iteration (34600) - Loss: 51.68335343536918
    Iteration (34700) - Loss: 51.5008322451973
    Iteration (34800) - Loss: 51.52423215073285
    Iteration (34900) - Loss: 51.44836109193933
    '
    
    "Ot pesterey,
    waw the rest noe I and suigrou wad tto you gr
    whigl.
    
    "'R't it
    ', the weske, opr; I ky'sle why copmex and that the scrakes owarkhy of ant for Sthavoede wher ih
    the foo fact, 'hat in ow
    Iteration (35000) - Loss: 51.21348815066355
    Iteration (35100) - Loss: 51.07266012333748
    Iteration (35200) - Loss: 51.356359787957565
    Iteration (35300) - Loss: 51.2725784609351
    Iteration (35400) - Loss: 51.24739728626246
    Iteration (35500) - Loss: 51.57229725963554
    Iteration (35600) - Loss: 51.576432067735446
    Iteration (35700) - Loss: 51.5173073835747
    Iteration (35800) - Loss: 51.34651635464582
    Iteration (35900) - Loss: 51.23265037339154
    reas."
    
    "
    the the with comeste and Whithe my ase to ralt-tore sithtichincb,
    mhave eh with
    hvirse himnAakern-not waster so
    be licoods is gad to her aromy as, wimy fownes wabl ut as the doghte-soy,"
    Yene
    Iteration (36000) - Loss: 50.992564701334246
    Iteration (36100) - Loss: 51.000634810367295
    Iteration (36200) - Loss: 51.13714499860396
    Iteration (36300) - Loss: 50.998952014366
    Iteration (36400) - Loss: 50.927041661708785
    Iteration (36500) - Loss: 50.80977387408794
    Iteration (36600) - Loss: 50.796628534628695
    Iteration (36700) - Loss: 50.633425862091464
    Iteration (36800) - Loss: 50.462882884913554
    Iteration (36900) - Loss: 50.372469154708455
     lail sid, bustlyle the."
    
    "Im, deeching, wers and that ding you dthert whanow of thenalmd,
    a subing lo thiealded I bech shistiep then as it a sely fas he, sith at emabss firsim in a that gthe perthent
    Iteration (37000) - Loss: 50.32728311804418
    Iteration (37100) - Loss: 50.48949299607886
    Iteration (37200) - Loss: 50.36081653768325
    Iteration (37300) - Loss: 50.72282091382568
    Iteration (37400) - Loss: 50.821352255133434
    Iteration (37500) - Loss: 50.731863724771166
    Iteration (37600) - Loss: 50.78363232661019
    Iteration (37700) - Loss: 50.872909218526715
    Iteration (37800) - Loss: 50.82060787786644
    Iteration (37900) - Loss: 50.76962604903388
    es," Crame.'
    
    "'Thoued at prove; chescely, buangent, It howed He thach it.
    
    "'Heur, wnouco ment."
    
    "'Thes, mn to seighaf' was is tindsteed-derin iver nowness out he sery po itwerlen weacte. I thongs th
    Iteration (38000) - Loss: 50.70343597615312
    Iteration (38100) - Loss: 50.49866127283658
    Iteration (38200) - Loss: 50.57409463814959
    Iteration (38300) - Loss: 50.55604120426858
    Iteration (38400) - Loss: 50.43693381610727
    Iteration (38500) - Loss: 50.34468347932976
    Iteration (38600) - Loss: 50.28051447549611
    Iteration (38700) - Loss: 50.16380054546846
    Iteration (38800) - Loss: 50.33269631355611
    Iteration (38900) - Loss: 50.329430341536636
     oo
    sad ough and the glick and tors, my the my. Wherines of
    and
    a sut mo
    tone the mance waghind of vane frige vach verllin, momenps opsins I has a mure anthary afaods of had mich enowmewwer the sill, y
    Iteration (39000) - Loss: 50.36559547979955
    Iteration (39100) - Loss: 50.71771732427915
    Iteration (39200) - Loss: 50.94537388575984
    Iteration (39300) - Loss: 51.23199099672592
    Iteration (39400) - Loss: 51.36337704649981
    Iteration (39500) - Loss: 51.15463725556048
    Iteration (39600) - Loss: 51.18912255842495
    Iteration (39700) - Loss: 51.04184881833608
    Iteration (39800) - Loss: 50.87578255219682
    Iteration (39900) - Loss: 50.81323062932975
    nce, ofer, Ir ableet, I, the prew gad, lase ace dipm, a gabrusams of the a hould laim
    sowalaanve to mave, of that a bet Protgeis."
    
    "Wive a be son dive a chotlime your Stice."
    
    "Youl? Ay in and it I a 
    Iteration (40000) - Loss: 50.808004152002354
    Iteration (40100) - Loss: 50.71533073819218
    Iteration (40200) - Loss: 50.87406669988159
    Iteration (40300) - Loss: 50.78567424610703
    Iteration (40400) - Loss: 50.81210159209594
    Iteration (40500) - Loss: 50.63707759220693
    Iteration (40600) - Loss: 50.47392833129973
    Iteration (40700) - Loss: 50.489505302857445
    Iteration (40800) - Loss: 50.669299193863594
    Iteration (40900) - Loss: 50.63090663391429
     exsuly im ve you reemy mtery-me sittildt with
    reer spabl hored are he son Stod the
    ufing ficticing Hoak beppey ack to he lamed the pasinven fare
    an
    han an frich to y
    your Dremured m. And me. Serted fr
    Iteration (41000) - Loss: 50.689703993117476
    Iteration (41100) - Loss: 50.64253223552052
    Iteration (41200) - Loss: 50.60847339527681
    Iteration (41300) - Loss: 50.57366960388829
    Iteration (41400) - Loss: 50.42472572705323
    Iteration (41500) - Loss: 50.22994262368175
    Iteration (41600) - Loss: 49.92415927781082
    Iteration (41700) - Loss: 49.883795133561044
    Iteration (41800) - Loss: 49.97199575218179
    Iteration (41900) - Loss: 49.89105017150947
    obys
    ut uponet, hime thared
    whiwsed in to dake? Herice fesed mea, to souttat, as who to yeel fare om
    a could byot you sist my of aferyedsendssiog konittMy ew
    the shaw se ifo fistelics intomrishy
    of the
    Iteration (42000) - Loss: 49.7245592000836
    Iteration (42100) - Loss: 49.610485184061076
    Iteration (42200) - Loss: 49.57234539952354
    Iteration (42300) - Loss: 49.5266232865469
    Iteration (42400) - Loss: 49.49882446326772
    Iteration (42500) - Loss: 49.406961858613165
    Iteration (42600) - Loss: 49.345555507017366
    Iteration (42700) - Loss: 49.121706271823896
    Iteration (42800) - Loss: 49.23921615657094
    Iteration (42900) - Loss: 49.623821145520694
    . Burrs ang ostat and
    sef and oh stors foour so smave
    bleenct for
    hea for a beed aokoar
    blon by."
    
    he poled to my im and be twe hersaclerofouvappinigalmave a as
    litustentle ther, whicht he hard," oly a
    Iteration (43000) - Loss: 49.89551144338542
    Iteration (43100) - Loss: 50.180799573852305
    Iteration (43200) - Loss: 50.33624688472691
    Iteration (43300) - Loss: 50.38147960244163
    Iteration (43400) - Loss: 50.608167877183696
    Iteration (43500) - Loss: 50.53936392324192
    Iteration (43600) - Loss: 50.65408477163618
    Iteration (43700) - Loss: 50.66343604636907
    Iteration (43800) - Loss: 50.49987477471547
    Iteration (43900) - Loss: 50.45388132584069
    m
    for."
    
    "The care decher and
    herselty to ableand to gele wel ale coun on lady fo ent dow dengemence, and. that courlded onasdigher, at en. Whice ann guncly
    me fulues
    youad ablen, oves, you dalre conce
    Iteration (44000) - Loss: 50.403882063213274
    Iteration (44100) - Loss: 50.28751045577519
    Iteration (44200) - Loss: 50.23019377360325
    Iteration (44300) - Loss: 50.00950923806504
    Iteration (44400) - Loss: 49.8316633066582
    Iteration (44500) - Loss: 49.659429656309385
    Iteration (44600) - Loss: 49.61056588627633
    Iteration (44700) - Loss: 49.665589475293025
    Iteration (44800) - Loss: 49.65023803298565
    Iteration (44900) - Loss: 49.49723782352959
     ter. The aocdsallons in then mas ser non, ceied
    enen boquins. Fraret comlele rupilpee vir veant indont were it who smating, chan
    enprow my. Hould whicred oney in
    to face fare whi hisV fare cosen lins
    
    Iteration (45000) - Loss: 49.88981976176248
    Iteration (45100) - Loss: 50.111612432559234
    Iteration (45200) - Loss: 50.166818540375125
    Iteration (45300) - Loss: 50.361875207585584
    Iteration (45400) - Loss: 50.34893475785469
    Iteration (45500) - Loss: 50.40699981578714
    Iteration (45600) - Loss: 50.78564718693475
    Iteration (45700) - Loss: 50.74979731788285
    Iteration (45800) - Loss: 50.654263383259156
    Iteration (45900) - Loss: 50.63899329207185
    ng wfos woughird, duap.r and was I Treattice not. Waw fur
    out-uremeh it the deisestle juch a woulnshich, 0ertir
    there thiu
    on. Tht when, herder. If and fored ther of namacht mfar
    that ferbant."
    
    "It er
    Iteration (46000) - Loss: 50.64086588288461
    Iteration (46100) - Loss: 50.53552472935556
    Iteration (46200) - Loss: 50.4953056998234
    Iteration (46300) - Loss: 50.47454908816654
    Iteration (46400) - Loss: 50.29418928357265
    Iteration (46500) - Loss: 50.11071331520311
    Iteration (46600) - Loss: 49.99958426685238
    Iteration (46700) - Loss: 50.21429421387747
    Iteration (46800) - Loss: 50.577215599295975
    Iteration (46900) - Loss: 50.67221127856746
     nound folk uppolvet, en
    hitttaprave Yis the starmen in, and made tu neal a heam?"
    nancass betiomvirill in that Erlow ot me us there or
    neno-ed, mut in Mr and ramate cecans
    abe, id lade, ate, and con
    c
    Iteration (47000) - Loss: 50.74776241074684
    Iteration (47100) - Loss: 51.04634502671359
    Iteration (47200) - Loss: 51.056546067676024
    Iteration (47300) - Loss: 50.858761143478425
    Iteration (47400) - Loss: 50.73120095533407
    Iteration (47500) - Loss: 50.521035483218355
    Iteration (47600) - Loss: 50.447184735047294
    Iteration (47700) - Loss: 50.761399382770854
    Iteration (47800) - Loss: 50.66158699870781
    Iteration (47900) - Loss: 50.801986957216776
    rt by at saod whew thenelabed the korg the gosters.
    '"
    Weile
    mantt tecr noke be"thiring a
    cors melt fugensed was I conencrean for from helsomphimed
    seish bro thound reraRt ter. Shat dodn.
    "We?'
    
    "A
    tha
    Iteration (48000) - Loss: 50.698722248655606
    Iteration (48100) - Loss: 50.73699223828946
    Iteration (48200) - Loss: 50.75640276020176
    Iteration (48300) - Loss: 50.72922279713683
    Iteration (48400) - Loss: 50.66714685058542
    Iteration (48500) - Loss: 50.50397454318632
    Iteration (48600) - Loss: 50.42121107179345
    Iteration (48700) - Loss: 50.20338563305448
    Iteration (48800) - Loss: 50.40390808383845
    Iteration (48900) - Loss: 50.438656388658316
    d a rast lighime
    very upoted folr mem in pufed that thear ave, at trarting cole tur
    shoush maar, ne has wand hat the reming tapilled was and val beraesid ofropke were ame but to come, quiting to of all
    Iteration (49000) - Loss: 50.3651159884344
    Iteration (49100) - Loss: 50.209544931522665
    Iteration (49200) - Loss: 50.20718457794563
    Iteration (49300) - Loss: 49.910458984254134
    Iteration (49400) - Loss: 49.67740721917596
    Iteration (49500) - Loss: 49.32989131082971
    Iteration (49600) - Loss: 49.37417822503153
    Iteration (49700) - Loss: 49.486820385579286
    Iteration (49800) - Loss: 49.718657354715404
    Iteration (49900) - Loss: 49.68555156781515
     a an muss has, spanthike this. Whered of ghark, suenatto.
    "'D. Hombliognlatuant to seake heariy to dow," orsin anjily to dano, Make in posmatcrice in to has myogk, a lare mitimim, whittice hard he fel
    Iteration (50000) - Loss: 49.654385140125285
    Iteration (50100) - Loss: 49.63164699778145
    Iteration (50200) - Loss: 49.52927497311267
    Iteration (50300) - Loss: 49.801342207171686
    Iteration (50400) - Loss: 49.78732577206835
    Iteration (50500) - Loss: 50.0204988189154
    Iteration (50600) - Loss: 49.95974973538918
    Iteration (50700) - Loss: 49.88632210443436
    Iteration (50800) - Loss: 49.794464669210925
    Iteration (50900) - Loss: 49.707347801084595
    nd sura nollamut you samm
    buted. I
    heasiouge. I on his?" geped you wat the twawly of a ca shoung his plong guth uppos is ef
    ous of he the Cuwan and in'tlen a nast in our sint.
    
    I Aray, the for
    -hallt n
    Iteration (51000) - Loss: 49.63306508539978
    Iteration (51100) - Loss: 49.66467254834791
    Iteration (51200) - Loss: 49.54676022879063
    Iteration (51300) - Loss: 49.39046743408085
    Iteration (51400) - Loss: 49.393232036560875
    Iteration (51500) - Loss: 49.29402616648115
    Iteration (51600) - Loss: 49.3358771935965
    Iteration (51700) - Loss: 49.1828906770929
    Iteration (51800) - Loss: 49.13356322811921
    Iteration (51900) - Loss: 49.24082732493824
    ntenon stwen the
    cangesteccance vounneswros copron the kmved of cor bod a cheit hor At fal. It seen?
    But anf gither'
    
    "I tas soar hmise. The clesthas in nequytiof sution the con. 
    "You "Burd?"
    
    "Whaid 
    Iteration (52000) - Loss: 49.25399854290805
    Iteration (52100) - Loss: 49.195651594503595
    Iteration (52200) - Loss: 49.167096853073176
    Iteration (52300) - Loss: 48.99985671668226
    Iteration (52400) - Loss: 49.183629445932446
    Iteration (52500) - Loss: 49.25801876960518
    Iteration (52600) - Loss: 49.09846598602505
    Iteration (52700) - Loss: 49.180313142748204
    Iteration (52800) - Loss: 49.3588739353878
    Iteration (52900) - Loss: 49.23288009843304
    h-now he bad he the mad.'
    
    "'0' to bychy parn, and obeaft."
    
    oy ovind. He't be pwopriount man'.'
    
    "Ked.'
    
    Hiss, inked te sme,
    how. he to. I Foint frove to that on anding ammytall. Ban my bush gmins, yo
    Iteration (53000) - Loss: 49.30304709389319
    Iteration (53100) - Loss: 49.31790012659409
    Iteration (53200) - Loss: 49.4426320668252
    Iteration (53300) - Loss: 49.22879132646775
    Iteration (53400) - Loss: 49.48587699382146
    Iteration (53500) - Loss: 49.505599420327634
    Iteration (53600) - Loss: 49.731635427916494
    Iteration (53700) - Loss: 49.762861272928795
    Iteration (53800) - Loss: 49.688997537433394
    Iteration (53900) - Loss: 49.76072548772453
    I and he to mus, unfion, acise be.
    
    "It te or iivl elee bon have Holmouar--"I luch I to holl we to a deruped Snomed gellice for its is of the retur usme's
    he
    beagh
    therisingh, Lerppiled in noy soud mor
    Iteration (54000) - Loss: 50.2149038817802
    Iteration (54100) - Loss: 50.13637746090953
    Iteration (54200) - Loss: 50.284684115698866
    Iteration (54300) - Loss: 50.20024001548645
    Iteration (54400) - Loss: 50.214427044439454
    Iteration (54500) - Loss: 50.278407571985966
    Iteration (54600) - Loss: 50.27504721405085
    Iteration (54700) - Loss: 50.013668222426425
    Iteration (54800) - Loss: 49.979570586604275
    Iteration (54900) - Loss: 49.68883473476743
    iending well howid with hilk as arocher alsser, altalt and comventideshigaed opo the. Ig poure that
    This cearnadode-siot, I herure at for the
    hackecestent of Mr the dowriant thot. H wwin I shige ccened
    Iteration (55000) - Loss: 49.53448677523627
    Iteration (55100) - Loss: 49.45831756419176
    Iteration (55200) - Loss: 49.42591276452698
    Iteration (55300) - Loss: 49.17640944618555
    Iteration (55400) - Loss: 49.11676694950877
    Iteration (55500) - Loss: 49.0834408629562
    Iteration (55600) - Loss: 49.25572981877956
    Iteration (55700) - Loss: 49.22888754347925
    Iteration (55800) - Loss: 49.2795479904639
    Iteration (55900) - Loss: 49.16032948955418
    out arders will my thighed hingioy my mir,
    which wairet wolreliblt be thuther-on't yound bome, nown allelthy," slingonved to
    lopred that, mad heall."
    
    "On,
    whe with whitt is I havseds glotuln usore asa
    Iteration (56000) - Loss: 49.329393364040925
    Iteration (56100) - Loss: 49.4576108998352
    Iteration (56200) - Loss: 49.353082081333156
    Iteration (56300) - Loss: 49.15693655278042
    Iteration (56400) - Loss: 49.26523015108362
    Iteration (56500) - Loss: 49.3852481973135
    Iteration (56600) - Loss: 49.43646676093834
    Iteration (56700) - Loss: 49.44206869108307
    Iteration (56800) - Loss: 49.58594563566737
    Iteration (56900) - Loss: 49.771361943044035
    lkeld sice that the supokonf?"
    
    
    Orent mankesfin-y misigeytion all aicurg it squateld blon notstarf stliride have ray rould lidl roon shore the lenily, for hourk, you limeyicl,"
    
    It whom ally holked th
    Iteration (57000) - Loss: 49.62388578098728
    Iteration (57100) - Loss: 49.73078568104353
    Iteration (57200) - Loss: 49.655891342972986
    Iteration (57300) - Loss: 49.6139234751015
    Iteration (57400) - Loss: 49.488546965877624
    Iteration (57500) - Loss: 49.16745440946265
    Iteration (57600) - Loss: 49.39319661046995
    Iteration (57700) - Loss: 49.35226094054962
    Iteration (57800) - Loss: 49.28442457548982
    Iteration (57900) - Loss: 49.4056003595044
    h and weredonf, the our dremust asled anderls alf ay and raupfore abilowed linged stapply that to thit
    hathert rovses, assey I an Dotcay. Af and hase noong't
    mes condap."
    
    "Yapest takeichint whife. As 
    Iteration (58000) - Loss: 49.67134411603404
    Iteration (58100) - Loss: 49.61509434710926
    Iteration (58200) - Loss: 49.58136803705094
    Iteration (58300) - Loss: 49.35125622793324
    Iteration (58400) - Loss: 49.26362732218362
    Iteration (58500) - Loss: 49.073546421775085
    Iteration (58600) - Loss: 49.09865573795132
    Iteration (58700) - Loss: 49.16465748030172
    Iteration (58800) - Loss: 49.09019658895063
    Iteration (58900) - Loss: 48.96993222667094
     bam. "Yeatep mofed wald pomled bpark in roiss wafter cavenel. How encon of homablel hattismat buse
    lets batablge am ane a bist wo vy beem,
    souly and used
    down no, wind benidardet cuse the stire of be 
    Iteration (59000) - Loss: 48.94582609576522
    Iteration (59100) - Loss: 48.80970655396174
    Iteration (59200) - Loss: 48.679454257105995
    Iteration (59300) - Loss: 48.58352850903875
    Iteration (59400) - Loss: 48.553689386583166
    Iteration (59500) - Loss: 48.54359073210367
    Iteration (59600) - Loss: 48.74276024110647
    Iteration (59700) - Loss: 48.564435539011896
    Iteration (59800) - Loss: 49.00821859571491
    Iteration (59900) - Loss: 49.1253871741648
    e we would I gruth. She verytelcly to of the for my of no
    mops ip to the to has Iated cofed it I be who tull the stom hos live the form comy of you to sutey. I glam intery reid you dase was him coust b
    Iteration (60000) - Loss: 48.96268671244042
    Iteration (60100) - Loss: 49.14795565456595
    Iteration (60200) - Loss: 49.11747856174245
    Iteration (60300) - Loss: 49.01992232623799
    Iteration (60400) - Loss: 49.01812674887806
    Iteration (60500) - Loss: 48.829100281050174
    Iteration (60600) - Loss: 48.7567887682506
    Iteration (60700) - Loss: 48.805907585407326
    Iteration (60800) - Loss: 48.79985819785196
    Iteration (60900) - Loss: 48.68589832224739
    d alair hand the to coined mit to at bedad imperocl, and who merecy of de wroky a
    light whad thlor the my renly pess mocrone a
    srot deapbed the mabdeds, samely why wabt. I mundet, wews no goome the ay 
    Iteration (61000) - Loss: 48.62549294855689
    Iteration (61100) - Loss: 48.559429944602385
    Iteration (61200) - Loss: 48.58423885073003
    Iteration (61300) - Loss: 48.6464913381672
    Iteration (61400) - Loss: 48.5861635158914
    Iteration (61500) - Loss: 48.91232780980898
    Iteration (61600) - Loss: 48.94989429750652
    Iteration (61700) - Loss: 49.21860895335466
    Iteration (61800) - Loss: 49.552175653560546
    Iteration (61900) - Loss: 49.61846432657916
    row docest, a phaled pance is he this wisced conder's fered
    of sise solkong be, rad out bt I to
    fore thas Luloul hatt a basive, te that Seve a
    guped. Thing, brest betuvinat working lothasr."
    
    "
    here Go
    Iteration (62000) - Loss: 49.512013679569904
    Iteration (62100) - Loss: 49.4305409846865
    Iteration (62200) - Loss: 49.25782232961202
    Iteration (62300) - Loss: 49.229046996050556
    Iteration (62400) - Loss: 49.01653815462142
    Iteration (62500) - Loss: 49.039733119186174
    Iteration (62600) - Loss: 48.99451639199557
    Iteration (62700) - Loss: 49.106743498152916
    Iteration (62800) - Loss: 49.02669513975727
    Iteration (62900) - Loss: 49.07335325873881
    alring and the sed Mr. I had he ardedaald hack a foraghtwe had of enolother, I you but way in him!
    "Nod do, theard andy wadgely of he have Stance rowed
    a gas has
    sork is the aman was lmanbnow a made ma
    Iteration (63000) - Loss: 48.90201226878264
    Iteration (63100) - Loss: 48.79641238364262
    Iteration (63200) - Loss: 48.7548625488673
    Iteration (63300) - Loss: 49.02473217885412
    Iteration (63400) - Loss: 48.91727688744095
    Iteration (63500) - Loss: 49.00676776674546
    Iteration (63600) - Loss: 48.89590457956563
    Iteration (63700) - Loss: 48.91000843819178
    Iteration (63800) - Loss: 48.82539526906072
    Iteration (63900) - Loss: 48.61493240642203
    e alld yeven's osse lyink! I wold 'I sare the dore. "Dits I sthous -than sish a ceald
    up and is and fo'd hive but no
    mise that I dose, beane, 'You dely up estald is locn that of the whall in of and sha
    Iteration (64000) - Loss: 48.41810753361704
    Iteration (64100) - Loss: 48.233717890248855
    Iteration (64200) - Loss: 48.27872875215145
    Iteration (64300) - Loss: 48.31177250803774
    Iteration (64400) - Loss: 48.23399294682449
    Iteration (64500) - Loss: 48.06433573138856
    Iteration (64600) - Loss: 47.92114427179121
    Iteration (64700) - Loss: 47.90653688205386
    Iteration (64800) - Loss: 47.87354835745169
    Iteration (64900) - Loss: 47.843603491074774
    nt, Ally, onrad a
    konderreno
    clowhn pher her him."
    
    "I his hand in to a urssing anterss
    whimt, in the rinten of the wowing hongcention, and cratt Bear it and but ,
    Stentcen the handrare me. Gonmally in
    Iteration (65000) - Loss: 47.776289361290054
    Iteration (65100) - Loss: 47.62005689875859
    Iteration (65200) - Loss: 47.487447131748304
    Iteration (65300) - Loss: 47.80239156335897
    Iteration (65400) - Loss: 48.03223265598065
    Iteration (65500) - Loss: 48.54034797004325
    Iteration (65600) - Loss: 48.54852604450821
    Iteration (65700) - Loss: 48.7644269024008
    Iteration (65800) - Loss: 48.708033675118905
    Iteration (65900) - Loss: 48.93856209687655
    ied of ever shins mistly
    know you derelieght whire."
    A"'sucgolded," seilainly geptweremed encettion rreach were it the lewurely fabked was fordly, buther turqust devick watt. Af saflee maach! Is howin 
    Iteration (66000) - Loss: 48.82578377767026
    Iteration (66100) - Loss: 49.023148265374736
    Iteration (66200) - Loss: 48.933833032869366
    Iteration (66300) - Loss: 48.873817496963476
    Iteration (66400) - Loss: 48.78042218804136
    Iteration (66500) - Loss: 48.727860286987664
    Iteration (66600) - Loss: 48.68748404015018
    Iteration (66700) - Loss: 48.48664436172627
    Iteration (66800) - Loss: 48.29972505273421
    Iteration (66900) - Loss: 48.14311167146016
    now his lough
    I cot in bres fard end sundowe unde with a -and save imlour old acting pape lat was have I darey.'
    
    "My my heard fitselteld frung out like I her alle raig. A it her. And his il de intlish
    Iteration (67000) - Loss: 47.89379205954537
    Iteration (67100) - Loss: 47.920401007161
    Iteration (67200) - Loss: 47.897088532484595
    Iteration (67300) - Loss: 47.92026608148208
    Iteration (67400) - Loss: 47.88528413781537
    Iteration (67500) - Loss: 48.358751265208205
    Iteration (67600) - Loss: 48.482789951227986
    Iteration (67700) - Loss: 48.50994558680255
    Iteration (67800) - Loss: 48.823207996623886
    Iteration (67900) - Loss: 48.84105322445506
    ice eaghy in the en, me," dorpenougher ae." O"I distur derecand Holm
    that anco me as and shis was alened and gothome was to oxnamame intterloter morand that put yam
    demough, take the mast ofe Horres yo
    Iteration (68000) - Loss: 48.91464709631816
    Iteration (68100) - Loss: 49.28173839562848
    Iteration (68200) - Loss: 49.15973122385029
    Iteration (68300) - Loss: 49.08677252992183
    Iteration (68400) - Loss: 49.11081866372582
    Iteration (68500) - Loss: 49.20930485646631
    Iteration (68600) - Loss: 49.00858950959461
    Iteration (68700) - Loss: 49.04126893950443
    Iteration (68800) - Loss: 48.918668032360785
    Iteration (68900) - Loss: 48.80458375219232
    utthich there upold suphened fremousted. Then treand haar tho pilutim the
    sas men
    the that he
    rastures him wrooselad
    you Smalpin! dows yeed
    Amed a chard awters., 'fooumep into cut as
    ferpied of sharl, 
    Iteration (69000) - Loss: 48.64189579269604
    Iteration (69100) - Loss: 48.533693005681066
    Iteration (69200) - Loss: 48.72698307856201
    Iteration (69300) - Loss: 49.11435242741846
    Iteration (69400) - Loss: 49.05311327801277
    Iteration (69500) - Loss: 49.148301712972916
    Iteration (69600) - Loss: 49.5585089683427
    Iteration (69700) - Loss: 49.410167136399664
    Iteration (69800) - Loss: 49.19329038504735
    Iteration (69900) - Loss: 49.10475285989411
    Jorrive
    tho puplowis astonow the has she ent carnouly should of the bpars's.'
    
    "Thay in in
    reanvers?"
    
    "Yeh
    the sille ovore thay, was of shalrne
    'so food crimay, te ans of the so the bange you stisince
    Iteration (70000) - Loss: 48.93578377504278
    Iteration (70100) - Loss: 48.85185634088296
    Iteration (70200) - Loss: 49.159065517485764
    Iteration (70300) - Loss: 48.999564528732456
    Iteration (70400) - Loss: 49.17844948476004
    Iteration (70500) - Loss: 49.25972916346241
    Iteration (70600) - Loss: 49.21864797529367
    Iteration (70700) - Loss: 49.23980765070139
    Iteration (70800) - Loss: 49.201287926941184
    Iteration (70900) - Loss: 49.1222084815926
    as thin I heary of lomeet, e
    thened and mryow
    saig, uster tround the
    have sthersf-o stoed busene
    that with us deart as athiempide setwear fforness out he
    phoandt. It some, he whund laomes dowen where M
    Iteration (71000) - Loss: 48.96139977362781
    Iteration (71100) - Loss: 48.82861702040266
    Iteration (71200) - Loss: 48.656712043838915
    Iteration (71300) - Loss: 48.93386223109971
    Iteration (71400) - Loss: 48.885751899213
    Iteration (71500) - Loss: 48.88695043309844
    Iteration (71600) - Loss: 48.7344211570629
    Iteration (71700) - Loss: 48.573733157515505
    Iteration (71800) - Loss: 48.234695607100576
    Iteration (71900) - Loss: 48.02192869264067
     pratesk," seive hi
    hert, be ta and Op nodres; saigened of uvinds, a end tome, nime be dide fo. He, Fand, this saiy cablich ancerd wave fris fat-evened. The whiak whith rom a
    wive tulle have the haim, 
    Iteration (72000) - Loss: 47.78645438488739
    Iteration (72100) - Loss: 47.94843648990592
    Iteration (72200) - Loss: 48.203305430514426
    Iteration (72300) - Loss: 48.245813558050635
    Iteration (72400) - Loss: 48.21883093650877
    Iteration (72500) - Loss: 48.19097327915182
    Iteration (72600) - Loss: 48.11927757542309
    Iteration (72700) - Loss: 48.01200445560939
    Iteration (72800) - Loss: 48.31186549308101
    Iteration (72900) - Loss: 48.375226139238904
    uctciwpelly gly
    cerme for then wey hownike and the sceking to argher wam sis Masech hisgel.
    
    "We foodtow of wevesstaptor,
    in an of the toore thresed m. "Your
    ofid yeghopentense curesery-egupionson thon
    Iteration (73000) - Loss: 48.4927558544285
    Iteration (73100) - Loss: 48.50149341527242
    Iteration (73200) - Loss: 48.390956033995266
    Iteration (73300) - Loss: 48.332145939989026
    Iteration (73400) - Loss: 48.24372544014962
    Iteration (73500) - Loss: 48.28624695217945
    Iteration (73600) - Loss: 48.18830935534461
    Iteration (73700) - Loss: 48.05457687475934
    Iteration (73800) - Loss: 47.937311218964346
    Iteration (73900) - Loss: 47.80061983694155
    s fall a
    colse chitess, wak On the wrilof
    te thas
    ofroughe busideist
    ro wurtibed, In the acstatith Panger
    seet, and frompst thair that a fo then is of stent. Itwe he, is of to
    the Courting of the Rus a
    Iteration (74000) - Loss: 47.825963779605495
    Iteration (74100) - Loss: 47.906941450088496
    Iteration (74200) - Loss: 47.77066306278856
    Iteration (74300) - Loss: 47.72917530232345
    Iteration (74400) - Loss: 47.83674117533984
    Iteration (74500) - Loss: 47.967969077202774
    Iteration (74600) - Loss: 47.80296493103686
    Iteration (74700) - Loss: 47.83167920485147
    Iteration (74800) - Loss: 47.65058965911368
    Iteration (74900) - Loss: 47.98094465530934
     of the by in exowned lother ufsen your
    entwook ought and oS know."
    
    Holmes weelank, folpwece defleeing pibour ane oncate kundly and a cane you lacke
    muesser dusss.
    
    "Go myich do Is wnep and theod sitt
    Iteration (75000) - Loss: 47.97751827829914
    Iteration (75100) - Loss: 47.887912365491
    Iteration (75200) - Loss: 47.99834433175219
    Iteration (75300) - Loss: 48.07319440571257
    Iteration (75400) - Loss: 48.01172582729194
    Iteration (75500) - Loss: 48.029881415668974
    Iteration (75600) - Loss: 48.05548027060594
    Iteration (75700) - Loss: 48.06457958329479
    Iteration (75800) - Loss: 48.020495272471734
    Iteration (75900) - Loss: 48.30928374618632
    mens expe; sime I mabutco it it sus arrottals
    onet that is exseardes overy whate to lick berlit
     crardencreaste of a muron onchay to nas the esdornas teare upoul filled herely clatireasines un: the sth
    Iteration (76000) - Loss: 48.23403100973168
    Iteration (76100) - Loss: 48.47732287228619
    Iteration (76200) - Loss: 48.57066764244934
    Iteration (76300) - Loss: 48.5901132408607
    Iteration (76400) - Loss: 48.52334505281231
    Iteration (76500) - Loss: 48.9355741395553
    Iteration (76600) - Loss: 48.968495743903496
    Iteration (76700) - Loss: 49.12608276176005
    Iteration (76800) - Loss: 49.02114077838232
    Iteration (76900) - Loss: 49.1550255001444
    er whll whing the gunged to that he maged of cearadesk a very linges niviee bote to soge a soutl that cot whit, theanto mo stwerrete sif the have a maghters my. I
    sive mad liging dinghhy hind. I refe,
    
    Iteration (77000) - Loss: 49.149884063668566
    Iteration (77100) - Loss: 49.038333963848956
    Iteration (77200) - Loss: 48.839346726028765
    Iteration (77300) - Loss: 48.74229204068363
    Iteration (77400) - Loss: 48.43008630159672
    Iteration (77500) - Loss: 48.383342380737005
    Iteration (77600) - Loss: 48.217488723570284
    Iteration (77700) - Loss: 48.14284623864495
    Iteration (77800) - Loss: 48.00921424452907
    Iteration (77900) - Loss: 48.00567416531327
    whep.
    Hown aselycedf.r
    youll had Marlly concur one tithar inting the
    extecs, the gaice."
    
    "Ben Head bilrep rie'
    
    "Herled ramed, ay. Whit a stion and Hollich it ar claint my the hat but of-cos the polcr
    Iteration (78000) - Loss: 47.972474212328706
    Iteration (78100) - Loss: 48.036928419587774
    Iteration (78200) - Loss: 48.10435468959001
    Iteration (78300) - Loss: 48.08640717547363
    Iteration (78400) - Loss: 48.113423857862934
    Iteration (78500) - Loss: 48.1830918860825
    Iteration (78600) - Loss: 48.24713946558697
    Iteration (78700) - Loss: 48.14058866450957
    Iteration (78800) - Loss: 47.952282459168494
    Iteration (78900) - Loss: 48.13178925242765
     ded beared roulds amseres
    putter a rirke yof grom me wribe
    the hiss that alver, was the cat wrouw.
    
    "E
    Attocend paukentt. This inton reisely is you trapser.
    
    "Whimether phanco blase my alpering mist, 
    Iteration (79000) - Loss: 48.22541393082083
    Iteration (79100) - Loss: 48.24754447688881
    Iteration (79200) - Loss: 48.269183037107716
    Iteration (79300) - Loss: 48.46079262178903
    Iteration (79400) - Loss: 48.556833381485305
    Iteration (79500) - Loss: 48.58112799592508
    Iteration (79600) - Loss: 48.489786813519615
    Iteration (79700) - Loss: 48.41337835448562
    Iteration (79800) - Loss: 48.45913513250916
    Iteration (79900) - Loss: 48.16905022391446
    to. Yit dowes our arid I raint
    non swintion wer."
    
    "But the megenep't trice inshow' conge not that this named sich wagked
    ghere?'
    
    You havate! It
    beed. Mllyt And wiggane of the sardy
    my teess, I was ex
    Iteration (80000) - Loss: 47.92052134153264
    Iteration (80100) - Loss: 48.215673298852366
    Iteration (80200) - Loss: 48.16983799917533
    Iteration (80300) - Loss: 48.08946294998183
    Iteration (80400) - Loss: 48.34013948077774
    Iteration (80500) - Loss: 48.45010840782689
    Iteration (80600) - Loss: 48.429108798289946
    Iteration (80700) - Loss: 48.39875414106207
    Iteration (80800) - Loss: 48.25482009737537
    Iteration (80900) - Loss: 48.07079627726726
    is
    out but with that osterly to of Mr. An, whating. He veran," sticling cordeit bus, fathes, ordict acaliascougtsire throe Bare full exined it it neaghin, us, I hig utoncto-wiinlyst's ,
    Afmer."
    
    "That 
    Iteration (81000) - Loss: 47.941815410683404
    Iteration (81100) - Loss: 47.95392661551517
    Iteration (81200) - Loss: 47.98641434846696
    Iteration (81300) - Loss: 47.89082596320636
    Iteration (81400) - Loss: 47.82804547033531
    Iteration (81500) - Loss: 47.821061212058154
    Iteration (81600) - Loss: 47.54636069277259
    Iteration (81700) - Loss: 47.453525182208665
    Iteration (81800) - Loss: 47.43505982993685
    Iteration (81900) - Loss: 47.40324913939813
    t is in of murded I the rempllding who capelathen her forter all the staoses, "You disten fandafked, sher Nor we are the dienps pasd in on the takens
    , bectres, and the setter."
    
    He to wat youn her see
    Iteration (82000) - Loss: 47.42719007724012
    Iteration (82100) - Loss: 47.52581059997929
    Iteration (82200) - Loss: 47.598711952670406
    Iteration (82300) - Loss: 47.74928093423792
    Iteration (82400) - Loss: 47.849249861474696
    Iteration (82500) - Loss: 47.74117007006235
    Iteration (82600) - Loss: 47.89163714209473
    Iteration (82700) - Loss: 47.88356113403685
    Iteration (82800) - Loss: 47.745802318719065
    Iteration (82900) - Loss: 47.692353653888965
    e no
    gandelver mew. The brive-cut which drom her the sart was onre all mad come then down in as in tis gondlod, pratted it to fithe, por that? I comeenther of with he wo
    loo saif of the ocke, oto is is
    Iteration (83000) - Loss: 47.5632259908679
    Iteration (83100) - Loss: 47.57304672264192
    Iteration (83200) - Loss: 47.61188541387925
    Iteration (83300) - Loss: 47.58149391461268
    Iteration (83400) - Loss: 47.53190180238981
    Iteration (83500) - Loss: 47.454765014478745
    Iteration (83600) - Loss: 47.37553390605216
    Iteration (83700) - Loss: 47.56154747264696
    Iteration (83800) - Loss: 47.40827976346881
    Iteration (83900) - Loss: 47.46804755806074
    hall, arrew I a ut over arterall, sfoctilt as to you ciloughinains for cam a cerpon the shad, and
    sore which nom aball ans of he beam of swat whe Brabling me a down is
    were tryoune in pench that indorn
    Iteration (84000) - Loss: 47.75740826000093
    Iteration (84100) - Loss: 47.84688012899865
    Iteration (84200) - Loss: 48.2083956031004
    Iteration (84300) - Loss: 48.37101175134641
    Iteration (84400) - Loss: 48.425812455418885
    Iteration (84500) - Loss: 48.344810018491835
    Iteration (84600) - Loss: 48.25996208534375
    Iteration (84700) - Loss: 48.00737368087296
    Iteration (84800) - Loss: 47.97137056157116
    Iteration (84900) - Loss: 47.84869902440071
     the forgeatally is to yeming
    nood the capaly on phad-all jotefarg. I mardent of St flomes
    fatting natgeced, forayked cinoid is us and of a net. Eutad the
    has weres "the lady. He camperson feainad a nl
    Iteration (85000) - Loss: 47.81639055396493
    Iteration (85100) - Loss: 47.90382393053668
    Iteration (85200) - Loss: 47.9260483215898
    Iteration (85300) - Loss: 47.90695858210599
    Iteration (85400) - Loss: 47.81301548810895
    Iteration (85500) - Loss: 47.60684155594431
    Iteration (85600) - Loss: 47.51307800183339
    Iteration (85700) - Loss: 47.57920403472166
    Iteration (85800) - Loss: 47.78873667489575
    Iteration (85900) - Loss: 47.75371446495202
    f efat 'Vlu steacutaly tor watles oo fet fedimasao,
    bring. sxat in
    bley fate dise tricked Shtyicuchy you dates
    with bage than Whi duster that of homed oncal exaont a farss afe loid a peaen-to. 
    oreard 
    Iteration (86000) - Loss: 47.754103345612265
    Iteration (86100) - Loss: 47.64028092027491
    Iteration (86200) - Loss: 47.668383219885584
    Iteration (86300) - Loss: 47.702293883458886
    Iteration (86400) - Loss: 47.43768145222836
    Iteration (86500) - Loss: 47.264534316319626
    Iteration (86600) - Loss: 47.221740966268506
    Iteration (86700) - Loss: 47.18620231587025
    Iteration (86800) - Loss: 47.20617167500559
    Iteration (86900) - Loss: 47.03360468833726
     upor?"
    
    "'Ere chase? "My the drewtlor mank the gakesper-Mrokated net which that liensed it buch forich
    rade. And be infers an trough could a canss ovave, ha blaien ingining in enited tof of gake rithe
    Iteration (87000) - Loss: 46.957615761112834
    Iteration (87100) - Loss: 46.85990770448187
    Iteration (87200) - Loss: 46.73623570803197
    Iteration (87300) - Loss: 46.79071795914517
    Iteration (87400) - Loss: 46.6908474069662
    Iteration (87500) - Loss: 46.63559900948458
    Iteration (87600) - Loss: 46.4493613442727
    Iteration (87700) - Loss: 46.501578021750014
    Iteration (87800) - Loss: 46.83644129098901
    Iteration (87900) - Loss: 46.94810036624741
    n encures to the kin, 2ther dling a prito, to the nes
    afpenced I al reen and feess of a larver, aw
    or offers. rut we call for Mrime to gomsan us ence to ne afsure.
    
    "froums thoughly, breald in of his t
    Iteration (88000) - Loss: 47.358885228411886
    Iteration (88100) - Loss: 47.53155197408615
    Iteration (88200) - Loss: 47.631327167054145
    Iteration (88300) - Loss: 47.63780555499834
    Iteration (88400) - Loss: 47.71464175087423
    Iteration (88500) - Loss: 47.70352092849419
    Iteration (88600) - Loss: 47.8610592895189
    Iteration (88700) - Loss: 47.73230809655342
    Iteration (88800) - Loss: 47.687373579639704
    Iteration (88900) - Loss: 47.637005106047184
    'Lepiry cafl
    enchive exore will
    iiplelcey have for
    my a verucheld frackinid in in the mad and slampeding carfel, this cut boughinard to my me withen for I mays, "hat hee, dinthor, I han
    sil froyd whoan
    Iteration (89000) - Loss: 47.61950380115901
    Iteration (89100) - Loss: 47.632501415471054
    Iteration (89200) - Loss: 47.30833362131499
    Iteration (89300) - Loss: 47.1699494512136
    Iteration (89400) - Loss: 47.02663119208522
    Iteration (89500) - Loss: 46.9050553195392
    Iteration (89600) - Loss: 46.95326503455927
    Iteration (89700) - Loss: 46.91325166655104
    Iteration (89800) - Loss: 46.882493938304656
    Iteration (89900) - Loss: 47.10583114133891
     mome old thet of think, the
    sflens in Hollrent
    inlerd, which gase or peritineath
    preme hare alloughther and been dock me, my apon ineeds; gasly there prieted gure intyess treaplect has stace his rerpa
    Iteration (90000) - Loss: 47.39470558167736
    Iteration (90100) - Loss: 47.46520688243089
    Iteration (90200) - Loss: 47.42748376538343
    Iteration (90300) - Loss: 47.76766970211753
    Iteration (90400) - Loss: 47.74628653368707
    Iteration (90500) - Loss: 47.99193084917899
    Iteration (90600) - Loss: 48.18431327894739
    Iteration (90700) - Loss: 48.06634539571846
    Iteration (90800) - Loss: 48.0829939957634
    Iteration (90900) - Loss: 48.17222386338953
    ere a mon I sthesfwo a know honle ierted froping't there thomly. Thoult, and it whet to onh vemoling gut Goon moring with in wavendly sapesurt of Thin were up peming! glrash to howed the sain catongo't
    Iteration (91000) - Loss: 48.11855333016273
    Iteration (91100) - Loss: 47.93633759457585
    Iteration (91200) - Loss: 48.02475426514572
    Iteration (91300) - Loss: 47.89604019969489
    Iteration (91400) - Loss: 47.70081405024911
    Iteration (91500) - Loss: 47.544671657773044
    Iteration (91600) - Loss: 47.5113235847405
    Iteration (91700) - Loss: 47.97218135703456
    Iteration (91800) - Loss: 48.050063064164235
    Iteration (91900) - Loss: 47.978524400604876
    ou trioking, and Aider apon which so Bast of nersfe it insind mat, and esstoret, withiny that up hove that it ill he ane beack tomer
    I bering nos, that stunged frokm Secintl."
    
    "At intihe
    persshed frop
    Iteration (92000) - Loss: 48.12466167643354
    Iteration (92100) - Loss: 48.455042008569194
    Iteration (92200) - Loss: 48.386321294991625
    Iteration (92300) - Loss: 48.155910806816244
    Iteration (92400) - Loss: 48.068078029153284
    Iteration (92500) - Loss: 47.96197521627819
    Iteration (92600) - Loss: 48.15952610842545
    Iteration (92700) - Loss: 48.05342434409617
    Iteration (92800) - Loss: 47.89056038305656
    Iteration (92900) - Loss: 48.020917736376475
     up courdremed noubulved was loin I saal him mad areloked remald ention, at rught aisutot noom and Lofk mithy
    alching make
    pommat
    which."
    
    "Frailt forled in his coudy a maid it to strock wnot metlown f
    Iteration (93000) - Loss: 48.22343543987889
    Iteration (93100) - Loss: 48.17390530962866
    Iteration (93200) - Loss: 48.18204337357366
    Iteration (93300) - Loss: 48.07087133868091
    Iteration (93400) - Loss: 48.04579584605202
    Iteration (93500) - Loss: 47.895679759864045
    Iteration (93600) - Loss: 47.87267023088185
    Iteration (93700) - Loss: 47.58868675785502
    Iteration (93800) - Loss: 47.971718294859265
    Iteration (93900) - Loss: 47.790792798350694
    u cafe to the mashed then have hee exphe fall end mutuf I
    clighing was chary oun here to bed,
    bound Cload whink chaas's
    outh as upor whock Whene fromed
    I smever hive which te horo, "Ohise, the
    have. Ho
    Iteration (94000) - Loss: 47.886154768851114
    Iteration (94100) - Loss: 47.772396459118845
    Iteration (94200) - Loss: 47.574902251808446
    Iteration (94300) - Loss: 47.26038573596718
    Iteration (94400) - Loss: 46.96007182428559
    Iteration (94500) - Loss: 46.85791380224713
    Iteration (94600) - Loss: 47.033442177568325
    Iteration (94700) - Loss: 47.27975017866287
    Iteration (94800) - Loss: 47.240391127186065
    Iteration (94900) - Loss: 47.23536947236847
    re po paratershted tatcingell noverserated.
    
    "What band, Mave to fed an," saice ofority, buse inceed."
    
    "'I ham, and fonderly triugher osher ha-chake marly laddistamann of he dor to werre,
    subpedet, ro
    Iteration (95000) - Loss: 47.132743630762604
    Iteration (95100) - Loss: 47.17685923918712
    Iteration (95200) - Loss: 47.035169472414054
    Iteration (95300) - Loss: 47.37227004207218
    Iteration (95400) - Loss: 47.50369365365578
    Iteration (95500) - Loss: 47.462911985008006
    Iteration (95600) - Loss: 47.52637292946112
    Iteration (95700) - Loss: 47.39522566307821
    Iteration (95800) - Loss: 47.37154756023225
    Iteration (95900) - Loss: 47.19154341329283
    aturioh?"
    
    ""I real of the blon I muy to mackished Bant houlo. Holmestore oos?"
    
    "Will masto's all frimel. I veeson ser. The
    rase myontous to bmathint I wnimestiog anvery than up in the a
    kille miy. Ee
    Iteration (96000) - Loss: 47.326464133150594
    Iteration (96100) - Loss: 47.15448674726534
    Iteration (96200) - Loss: 46.95814211932578
    Iteration (96300) - Loss: 46.95574504646959
    Iteration (96400) - Loss: 46.83511194142509
    Iteration (96500) - Loss: 46.87515504864421
    Iteration (96600) - Loss: 46.88781199634118
    Iteration (96700) - Loss: 46.80152342868821
    Iteration (96800) - Loss: 46.82874694654319
    Iteration (96900) - Loss: 46.853974276837135
    oured mugl of the ang one
    rean os louble."
    ""You stherang lonkep anly shou?" It mit had pofing at the his with his litted riss
    y, lidt stillany of an the wore buenille to I here fourt wistioufs gear, a
    Iteration (97000) - Loss: 46.95525824398567
    Iteration (97100) - Loss: 46.882448822637144
    Iteration (97200) - Loss: 46.79015982457628
    Iteration (97300) - Loss: 46.90341506252832
    Iteration (97400) - Loss: 46.993662716115224
    Iteration (97500) - Loss: 46.95676854791471
    Iteration (97600) - Loss: 46.9612790438574
    Iteration (97700) - Loss: 47.079120004203105
    Iteration (97800) - Loss: 47.02107573976713
    Iteration (97900) - Loss: 47.05673223486153
    keer at of ghank in upre ro his nard, you have, and it
    his dacked greaken antabpeverlysield the him the
    spoker his om. If everying interes'd Heret. "I vince, that
    is mose be tof nox inat. It his lost
    l
    Iteration (98000) - Loss: 47.116822875921045
    Iteration (98100) - Loss: 47.05877246297264
    Iteration (98200) - Loss: 47.08335293758114
    Iteration (98300) - Loss: 47.10076120968032
    Iteration (98400) - Loss: 47.4078195131841
    Iteration (98500) - Loss: 47.432150252159914
    Iteration (98600) - Loss: 47.63429089743154
    Iteration (98700) - Loss: 47.650553446002014
    Iteration (98800) - Loss: 47.76197806260025
    Iteration (98900) - Loss: 48.021853839917526
    . This bat at. It fere than nepod!"
    hoss-sinich
    the meind what you abmom
    lark be
    handtindce fropok to pobest fao,
    what down underall, bet shim beswucing, St pharo."
     "Goubon in con indow beved
    Undard, 
    Iteration (99000) - Loss: 48.08606143983557
    Iteration (99100) - Loss: 48.174007046608395
    Iteration (99200) - Loss: 48.244362362787804
    Iteration (99300) - Loss: 48.15809762497233
    Iteration (99400) - Loss: 48.2062786424727
    Iteration (99500) - Loss: 48.30392027298974
    Iteration (99600) - Loss: 48.116690166703556
    Iteration (99700) - Loss: 47.96798859845185
    Iteration (99800) - Loss: 47.707151222071644
    Iteration (99900) - Loss: 47.42529030868038
    n," the
    lity loosined ponly an,
    facaly Has elflan drieg non
    as housh
    rome arsor tapmn made to-nokeats my sed hom dous no had dearcher math a qut.
    
    "700nor, was
    of
    saintice, Sncaf-never hid frowavead ar
    Iteration (100000) - Loss: 47.49174871935582
    Iteration (100100) - Loss: 47.34047628460603
    Iteration (100200) - Loss: 47.229952461789196
    Iteration (100300) - Loss: 47.16665987538763
    Iteration (100400) - Loss: 47.158206916068885
    Iteration (100500) - Loss: 47.143643306282044
    Iteration (100600) - Loss: 47.163907568731574
    Iteration (100700) - Loss: 47.23807067136488
    Iteration (100800) - Loss: 47.2112485809383
    Iteration (100900) - Loss: 47.28823790280271
    soor mation.
    "And
    now hass out sthort had oponen pongering and of kfothist rivagever."
    
    "I was wideriss to deind afcome reamed."
    f of you helad the arcile-waseart, lino thations theaksed cruse fat it
    h
    Iteration (101000) - Loss: 47.3759923391704
    Iteration (101100) - Loss: 47.373469113475664
    Iteration (101200) - Loss: 47.23873521193258
    Iteration (101300) - Loss: 47.14181414816672
    Iteration (101400) - Loss: 47.33797861573893
    Iteration (101500) - Loss: 47.39623564178197
    Iteration (101600) - Loss: 47.42007586270081
    Iteration (101700) - Loss: 47.45819445674102
    Iteration (101800) - Loss: 47.666162512224425
    Iteration (101900) - Loss: 47.64536983148905
    of put
    geasing.
    
    FAre fourt allerel,
    and in the keingengach tave as's the markou!" the murneved you onnt, the havt be in thewef, that and that Sterinid ?"Which now think."
    
    "What your in we frops al hi
    Iteration (102000) - Loss: 47.776150878666535
    Iteration (102100) - Loss: 47.62579744757757
    Iteration (102200) - Loss: 47.62354211699451
    Iteration (102300) - Loss: 47.579045902555606
    Iteration (102400) - Loss: 47.33426288217438
    Iteration (102500) - Loss: 47.07617750746254
    Iteration (102600) - Loss: 47.31126003459625
    Iteration (102700) - Loss: 47.24397023726686
    Iteration (102800) - Loss: 47.20014894518788
    Iteration (102900) - Loss: 47.53495229379835
    nd a supilese to
    rerettirising-was aghorss the mouke afpered, afk we bring overs of
    they higrat andeewn bane is him uponed with fy and, and the man woulder arloger al sme be lingantllw roosentrles, poo
    Iteration (103000) - Loss: 47.62590515075057
    Iteration (103100) - Loss: 47.5274669726782
    Iteration (103200) - Loss: 47.44643519737383
    Iteration (103300) - Loss: 47.36742763381409
    Iteration (103400) - Loss: 47.153454228104515
    Iteration (103500) - Loss: 47.103241117860165
    Iteration (103600) - Loss: 47.140638142143374
    Iteration (103700) - Loss: 47.1669574507661
    Iteration (103800) - Loss: 47.00532286214056
    Iteration (103900) - Loss: 46.97328294839396
    s of
    basing well ma ald is slienf was said his to you whiet. To the enew."
    '"'"I knor turnew, whation mist be abelled beparr," said I had my the co-ctang the quich the listher. You that hime blaonsed o
    Iteration (104000) - Loss: 46.996156735672876
    Iteration (104100) - Loss: 46.72940939944357
    Iteration (104200) - Loss: 46.5765432065993
    Iteration (104300) - Loss: 46.66947480801459
    Iteration (104400) - Loss: 46.58629983901931
    Iteration (104500) - Loss: 46.77754299585897
    Iteration (104600) - Loss: 46.714555685038
    Iteration (104700) - Loss: 46.940930504482544
    Iteration (104800) - Loss: 47.138011359316074
    Iteration (104900) - Loss: 47.11566905246011
    
    
    "Aotergach seangos coublested sufe to think the arele of farl of
    the fir! lake is of she ostabler thied dill Holmes, ar' roughterer.
    
    " he promare to ane as -op shere I cried widous roond indeaes hea
    Iteration (105000) - Loss: 47.02640892794436
    Iteration (105100) - Loss: 47.11703718004599
    Iteration (105200) - Loss: 47.14147651967748
    Iteration (105300) - Loss: 46.97368793798898
    Iteration (105400) - Loss: 46.873653175476754
    Iteration (105500) - Loss: 46.69267211255833
    Iteration (105600) - Loss: 46.80148244062216
    Iteration (105700) - Loss: 46.808710980373824
    Iteration (105800) - Loss: 46.76393787509189
    Iteration (105900) - Loss: 46.69934594310858
    r his whepurd for herer. It ois bored if he has had whing mistled
    was laint, "nes, bubbso bbersers, and my abunt.
    
    "For is youfhad remard in the was be, besftersher. And that ded Snament a blow, wend, 
    Iteration (106000) - Loss: 46.65508606128085
    Iteration (106100) - Loss: 46.550619206790124
    Iteration (106200) - Loss: 46.74960063705567
    Iteration (106300) - Loss: 46.67036372173516
    Iteration (106400) - Loss: 46.752239733315335
    Iteration (106500) - Loss: 47.04219315281101
    Iteration (106600) - Loss: 47.082105576465096
    Iteration (106700) - Loss: 47.459529361564876
    Iteration (106800) - Loss: 47.561609602575004
    Iteration (106900) - Loss: 47.56416781194579
     laaning on in at downyss inay and which ace, and tixtejorengtisery."
    
    "Him upon my docken,."
    
    I carghen siid abpeens, and a very. Then I ather atiof."
    
    "The valr and
    praping bring. Ang "gomution houff
    Iteration (107000) - Loss: 47.58110641723742
    Iteration (107100) - Loss: 47.501597514785125
    Iteration (107200) - Loss: 47.31969110035458
    Iteration (107300) - Loss: 47.22812488025391
    Iteration (107400) - Loss: 47.18599392387909
    Iteration (107500) - Loss: 47.05594395827899
    Iteration (107600) - Loss: 47.11614804340131
    Iteration (107700) - Loss: 47.13455573403287
    Iteration (107800) - Loss: 47.17114994111862
    Iteration (107900) - Loss: 47.011191481837756
    horher, trimbend-by be justined the en,
    Lose
    or to gas in putalowring becafist ocest oo there? I dritle her befrighidoun. Not chist up firm which Stico, enter sime I Froall. Limp roo list to hy capjnep
    Iteration (108000) - Loss: 46.849993530869455
    Iteration (108100) - Loss: 46.733993986100735
    Iteration (108200) - Loss: 46.9632770477824
    Iteration (108300) - Loss: 46.96642664332404
    Iteration (108400) - Loss: 46.966947293649994
    Iteration (108500) - Loss: 47.01192323500664
    Iteration (108600) - Loss: 46.87711236748242
    Iteration (108700) - Loss: 46.90961179108641
    Iteration (108800) - Loss: 46.83240320522938
    Iteration (108900) - Loss: 46.660446997212645
    d anible to all sum. I awyed
    wolnan the cweved showhy, but I for in tere mad have
    lact facn opreasibuld bryock mamy I what his ane-out an the ste candiontide, that
    as rever was blangeread ona callee pr
    Iteration (109000) - Loss: 46.3845010563414
    Iteration (109100) - Loss: 46.398514084145816
    Iteration (109200) - Loss: 46.40785796942853
    Iteration (109300) - Loss: 46.42949873124363
    Iteration (109400) - Loss: 46.23376485672649
    Iteration (109500) - Loss: 46.099250818046016
    Iteration (109600) - Loss: 46.09237837556274
    Iteration (109700) - Loss: 45.931239119289195
    Iteration (109800) - Loss: 45.94055956343021
    Iteration (109900) - Loss: 45.955029917380564
    d
    am ofe He had arsibe. No thottermone blabone shounthed lifed resalles, sberwarsfmopejure boughane
    yetther you canfor
    twere dosmat I there
    one it swooen, ssentsibgaples
    know whore, you sork? It ofed t
    Iteration (110000) - Loss: 45.904151704454655
    Iteration (110100) - Loss: 45.70380472880199
    Iteration (110200) - Loss: 45.73783565870939
    Iteration (110300) - Loss: 46.08497699727038
    Iteration (110400) - Loss: 46.2890435040301
    Iteration (110500) - Loss: 46.604592681668834
    Iteration (110600) - Loss: 46.7098585782372
    Iteration (110700) - Loss: 46.806157488895586
    Iteration (110800) - Loss: 46.90898133038731
    Iteration (110900) - Loss: 46.80736772345058
    ," notle you momenfpectainy abourded am that what be in ou the smagned found, requed quich sfowads a
    tomend swoted be that is be.'
    
    "'Holl?"
    
    "And
    he. "Sytom I itsing them I staings perceatorowe
    the lo
    Iteration (111000) - Loss: 46.92037561107965
    Iteration (111100) - Loss: 47.03989466494837
    Iteration (111200) - Loss: 46.94761103858183
    Iteration (111300) - Loss: 46.84097022957768
    Iteration (111400) - Loss: 46.779290270550625
    Iteration (111500) - Loss: 46.734638788231884
    Iteration (111600) - Loss: 46.739107812671485
    Iteration (111700) - Loss: 46.490342610176995
    Iteration (111800) - Loss: 46.315685685946356
    Iteration (111900) - Loss: 46.14860611403726
    ch ally. Thes, longhior. Theyed to sell who af a worters, out werrestly that so the it
    untaluls susting, will, and there will-camightsed cains. Burds ever
    loobleend. "Ber ard of and hurturle seighthy n
    Iteration (112000) - Loss: 46.047081027379264
    Iteration (112100) - Loss: 46.061726269158946
    Iteration (112200) - Loss: 46.11221813939209
    Iteration (112300) - Loss: 46.00698398357929
    Iteration (112400) - Loss: 46.369648655299045
    Iteration (112500) - Loss: 46.64165623052922
    Iteration (112600) - Loss: 46.71883477020208
    Iteration (112700) - Loss: 46.69867937142322
    Iteration (112800) - Loss: 47.025345602365505
    Iteration (112900) - Loss: 47.059818873940685
    oughist it
    srecgiout me and grealdevivoring op
    this dight
    as as was handing jucl by in Friestilt
    om of
    said iy sactfeen.-"You moon.
    
    "There he toun my and even mims I dofe hein, into sode yourdensuved 
    Iteration (113000) - Loss: 47.471157674278906
    Iteration (113100) - Loss: 47.4011070812018
    Iteration (113200) - Loss: 47.38051510231421
    Iteration (113300) - Loss: 47.48625074666872
    Iteration (113400) - Loss: 47.547693741199495
    Iteration (113500) - Loss: 47.437278391560696
    Iteration (113600) - Loss: 47.28813691544831
    Iteration (113700) - Loss: 47.256404021104004
    Iteration (113800) - Loss: 47.20182632833257
    Iteration (113900) - Loss: 46.92242098332577
    . There faared up in the. "he have the lissibage,
    bether geturh the
    rither rem be beetionelak which we wrapter, I cowmer to the stomes and an
    hill neeting
    sher wyerter al
    whee hatkeitgel. My cheature c
    Iteration (114000) - Loss: 46.81885167251374
    Iteration (114100) - Loss: 46.961561113962794
    Iteration (114200) - Loss: 47.191242557883
    Iteration (114300) - Loss: 47.266588406943654
    Iteration (114400) - Loss: 47.33119051817879
    Iteration (114500) - Loss: 47.575129894308404
    Iteration (114600) - Loss: 47.664161883987816
    Iteration (114700) - Loss: 47.520859839349995
    Iteration (114800) - Loss: 47.300580074485254
    Iteration (114900) - Loss: 47.215012992404645
    t wel gurmen age, wmoy 18t to the mess it
    how
    of I was of is Tatide the lont."
    
    'And of I memanvide," thell count, and at home driens in
    thimemoot Helter and, buther beforets, nowes tait the knested th
    Iteration (115000) - Loss: 47.14188583622128
    Iteration (115100) - Loss: 47.288543534396396
    Iteration (115200) - Loss: 47.1756490014525
    Iteration (115300) - Loss: 47.152314767861476
    Iteration (115400) - Loss: 47.270917058072754
    Iteration (115500) - Loss: 47.40897655542672
    Iteration (115600) - Loss: 47.46673319842065
    Iteration (115700) - Loss: 47.43525398440322
    Iteration (115800) - Loss: 47.339263353460744
    Iteration (115900) - Loss: 47.21196323616999
    my
    I you my at he thoneln which wave the ban your the younks mat fouldalst any not antarm opeds of I hocrused was astore maf then clearn agary yand. Jack here that for a ose wassed," sonk they to had c
    Iteration (116000) - Loss: 47.1102501607772
    Iteration (116100) - Loss: 46.96423292367542
    Iteration (116200) - Loss: 47.12000858191688
    Iteration (116300) - Loss: 47.186893531077146
    Iteration (116400) - Loss: 47.11361264291034
    Iteration (116500) - Loss: 47.06841608175527
    Iteration (116600) - Loss: 46.97222738979101
    Iteration (116700) - Loss: 46.65222132385205
    Iteration (116800) - Loss: 46.36536615589022
    Iteration (116900) - Loss: 46.10140843914516
     Ny heowsed slise hid Mralvelmm bade; the bear shaple, and fall opevined ofed have that non able the can spiow upon, and has a hered, myter
    hoursed
    I hat It one rematestingast into."
    
    "Yous thigkebre o
    Iteration (117000) - Loss: 46.028233125773944
    Iteration (117100) - Loss: 46.28604559078223
    Iteration (117200) - Loss: 46.45305981676773
    Iteration (117300) - Loss: 46.51188730126385
    Iteration (117400) - Loss: 46.54363680968484
    Iteration (117500) - Loss: 46.43801786754807
    Iteration (117600) - Loss: 46.324769532400715
    Iteration (117700) - Loss: 46.32450824551334
    Iteration (117800) - Loss: 46.64021922280394
    Iteration (117900) - Loss: 46.804407586956344
    onvergarstereby in than safsucour act the mood use maidong of and traid oft koin. Cocen deet an a
    qappeen then the was and fion, and efle."
    
    "There his the carrounce the veriched scopt mond,
    wrred. She
    Iteration (118000) - Loss: 46.70100127811656
    Iteration (118100) - Loss: 46.75700144875052
    Iteration (118200) - Loss: 46.605722020674065
    Iteration (118300) - Loss: 46.58479197471778
    Iteration (118400) - Loss: 46.41271185358628
    Iteration (118500) - Loss: 46.5196396989622
    Iteration (118600) - Loss: 46.35234755835396
    Iteration (118700) - Loss: 46.21079108615512
    Iteration (118800) - Loss: 46.134897460743076
    Iteration (118900) - Loss: 46.05023338614825
    ply oo to quith sase buem, and at his
    salrusk from knored, willed was sho.
    Hormet?"
    Horweder twas osfither her a so to gress. Het san hat this not-cealine well, wno
    waid on the last as; somrolenwelf.'
    
    Iteration (119000) - Loss: 46.095751114371645
    Iteration (119100) - Loss: 46.112014691398635
    Iteration (119200) - Loss: 46.023537498965894
    Iteration (119300) - Loss: 46.095527112132956
    Iteration (119400) - Loss: 46.214003134496856
    Iteration (119500) - Loss: 46.15359757752655
    Iteration (119600) - Loss: 46.13338701671202
    Iteration (119700) - Loss: 46.10293220458567
    Iteration (119800) - Loss: 46.23549736017492
    Iteration (119900) - Loss: 46.3879872443891
     to come see. "I in pritelffer."
    
    "No, and, aid from you or
    the sirton to you colpers At
    husht sill."
    
    E'Clattaly shous forming was fack," the coubler have sticl his we had wore a
    bad-ars me mubation i
    Iteration (120000) - Loss: 46.19692262712558
    Iteration (120100) - Loss: 46.33313605992208
    Iteration (120200) - Loss: 46.49794785315083
    Iteration (120300) - Loss: 46.41214762105248
    Iteration (120400) - Loss: 46.48541035758675
    Iteration (120500) - Loss: 46.45659994425919
    Iteration (120600) - Loss: 46.400460045714816
    Iteration (120700) - Loss: 46.39422814808782
    Iteration (120800) - Loss: 46.56279571616736
    Iteration (120900) - Loss: 46.76123720949845
    is gettered."
    
    "This did my ind wills, of Ky obfopniked. Nown you thincer to that intear-a corncfed and-bidat. Fulk that the
    shay?"
    
    "He the sained. Wastive desire lidivery-Kbeal befevers it exsoned to
    Iteration (121000) - Loss: 46.962426838099105
    Iteration (121100) - Loss: 46.9498576616089
    Iteration (121200) - Loss: 46.99973562391087
    Iteration (121300) - Loss: 47.07736478790202
    Iteration (121400) - Loss: 47.42087317139995
    Iteration (121500) - Loss: 47.40745111000494
    Iteration (121600) - Loss: 47.62918082183615
    Iteration (121700) - Loss: 47.5757635466187
    Iteration (121800) - Loss: 47.59699021579441
    Iteration (121900) - Loss: 47.607392019631476
     rouked coiman at cCfone and who campelesed latine," simencriched a coms. Youns. Well the horel, 4n howeningy.
    
    "TherningCals, his obs:
    smo antion. Wet howed, no
    foresfion guC braw reess, HolmeI
    lipere
    Iteration (122000) - Loss: 47.64484077108369
    Iteration (122100) - Loss: 47.455775834419725
    Iteration (122200) - Loss: 47.40005790656913
    Iteration (122300) - Loss: 47.14282720811594
    Iteration (122400) - Loss: 46.82607842218344
    Iteration (122500) - Loss: 46.793084165014065
    Iteration (122600) - Loss: 46.72046022297596
    Iteration (122700) - Loss: 46.487512612458424
    Iteration (122800) - Loss: 46.50023085037496
    Iteration (122900) - Loss: 46.50040955366682
    ow
    stery, is his had say exapin, me when, and inthosghed we emple ih." Hould it of thecs of the ground of Leoccaled of high! Ieper had a hanjommer it. CUs sonted at can, mur seener
    she trough
    me sthowe
    Iteration (123000) - Loss: 46.58442648832375
    Iteration (123100) - Loss: 46.55007178835211
    Iteration (123200) - Loss: 46.636655897343694
    Iteration (123300) - Loss: 46.55489838920173
    Iteration (123400) - Loss: 46.58050534520037
    Iteration (123500) - Loss: 46.71157394612788
    Iteration (123600) - Loss: 46.66512136786725
    Iteration (123700) - Loss: 46.54717109806873
    Iteration (123800) - Loss: 46.52697271548428
    Iteration (123900) - Loss: 46.69903447535275
     in beption be; Bhan Pomnous wasser, of the may hisk as a betucured bome, is the silk who he have he workdyst tiviredss in a troid and your iy it. Hetherpel," 000 casjeul, emprearixtland you diry.
    But 
    Iteration (124000) - Loss: 46.6998383938108
    Iteration (124100) - Loss: 46.71372799774038
    Iteration (124200) - Loss: 46.81019559576393
    Iteration (124300) - Loss: 47.02972678680906
    Iteration (124400) - Loss: 46.96316830483973
    Iteration (124500) - Loss: 47.08056701452313
    Iteration (124600) - Loss: 46.9702237599485
    Iteration (124700) - Loss: 46.92190694951392
    Iteration (124800) - Loss: 46.84367806145509
    Iteration (124900) - Loss: 46.603950194084454
    e reames, I
    was a caugake-. As. Heard.
    
    "Aglt forbsalfe. Ostaings
    mined in jut then head an astary of the atf and and at foones tornes, 1he bave tork you, if is seash, and thracan.
    THolilatiley for to 
    Iteration (125000) - Loss: 46.47838234394237
    Iteration (125100) - Loss: 46.63565671620991
    Iteration (125200) - Loss: 46.630334751975745
    Iteration (125300) - Loss: 46.64799152647741
    Iteration (125400) - Loss: 47.013297636459704
    Iteration (125500) - Loss: 47.01111177037607
    Iteration (125600) - Loss: 46.95380146348021
    Iteration (125700) - Loss: 46.82093563750791
    Iteration (125800) - Loss: 46.71160566360215
    Iteration (125900) - Loss: 46.487681931055
     amard-beinge my reefs, in
    nealls offore upralmed mut
    is there yould frove, beanie and for bupl forms to fastion some peaks of so a carf. I had verod os is a sthastly whith enverf.r
    the serearding his 
    Iteration (126000) - Loss: 46.50713256824544
    Iteration (126100) - Loss: 46.613170602903836
    Iteration (126200) - Loss: 46.46734407508579
    Iteration (126300) - Loss: 46.400949751781326
    Iteration (126400) - Loss: 46.26805258799616
    Iteration (126500) - Loss: 46.29341268633881
    Iteration (126600) - Loss: 46.16768151680307
    Iteration (126700) - Loss: 46.001458373847974
    Iteration (126800) - Loss: 45.93043475717799
    Iteration (126900) - Loss: 45.894174737382464
    epiattaler hap the stlest of down and with a
    stry there, Proces. That acrid?"
    
    "Whay," hey a slasioy torching one
    moomury
    so then, ofone hightt with to doncal
    the soor,
    and mide.
    "Shmat bitare
    dore in 
    Iteration (127000) - Loss: 46.16325567856788
    Iteration (127100) - Loss: 46.039700194666615
    Iteration (127200) - Loss: 46.30016005928619
    Iteration (127300) - Loss: 46.416026596939936
    Iteration (127400) - Loss: 46.26285182734541
    Iteration (127500) - Loss: 46.38336638853341
    Iteration (127600) - Loss: 46.420732249166264
    Iteration (127700) - Loss: 46.35115742815224
    Iteration (127800) - Loss: 46.277126028852095
    Iteration (127900) - Loss: 46.19496002872227
    keron
    ancerr which lastle crase owe naghen which he shere, drabvaasing the. "It in that that wighern. The aming that mistlystlery hler frofite whate is silk-carery to of a fwruck, and street quith here
    Iteration (128000) - Loss: 46.02353088855665
    Iteration (128100) - Loss: 46.161097114819604
    Iteration (128200) - Loss: 46.19759267580977
    Iteration (128300) - Loss: 46.14074224642898
    Iteration (128400) - Loss: 46.07591275056632
    Iteration (128500) - Loss: 46.01039344684025
    Iteration (128600) - Loss: 45.9500724519837
    Iteration (128700) - Loss: 46.05610532302198
    Iteration (128800) - Loss: 46.069317687339534
    Iteration (128900) - Loss: 46.14815946976409
    
    have sture that of ladselves, he, where to bucl to yourdarcher see is mo lasatiot, were
    ex to huther:s the motteety and grimention rantave a fithong
    upon
    companily. There defose this by out was a pryo
    Iteration (129000) - Loss: 46.316410273627035
    Iteration (129100) - Loss: 46.65879248167521
    Iteration (129200) - Loss: 46.91524922907147
    Iteration (129300) - Loss: 47.060367247565544
    Iteration (129400) - Loss: 46.882550252484464
    Iteration (129500) - Loss: 46.830910436474774
    Iteration (129600) - Loss: 46.687817412130464
    Iteration (129700) - Loss: 46.61155208060292
    Iteration (129800) - Loss: 46.4519011493536
    Iteration (129900) - Loss: 46.46236545736636
    pcas, indice."
    
    "He enfist one the derectine
    Holme. "What yeid--gotrest, Pad. Tho.
    Save rourreather he all to past ate indist twithinidy fresh rothing, my nargherind shom, pasels stird of coust intelve
    Iteration (130000) - Loss: 46.401117963995226
    Iteration (130100) - Loss: 46.54996501452268
    Iteration (130200) - Loss: 46.46765328915491
    Iteration (130300) - Loss: 46.51710045002858
    Iteration (130400) - Loss: 46.29905042184648
    Iteration (130500) - Loss: 46.10445493469548
    Iteration (130600) - Loss: 46.082992210026156
    Iteration (130700) - Loss: 46.25739037155997
    Iteration (130800) - Loss: 46.20860853911174
    Iteration (130900) - Loss: 46.318093152846835
     caint, mishor I but of say is thourdperesher enowend to your pally
    thast he a kount.'
    
    "'But of
    o'd eass obes to miaked very
    and morector of I semay that in the should he.
    I hound is comakiny's of Dar
    Iteration (131000) - Loss: 46.26307449495622
    Iteration (131100) - Loss: 46.25842410154294
    Iteration (131200) - Loss: 46.277392327596
    Iteration (131300) - Loss: 46.111558837711634
    Iteration (131400) - Loss: 45.98721365476168
    Iteration (131500) - Loss: 45.749413206497486
    Iteration (131600) - Loss: 45.76886142617323
    Iteration (131700) - Loss: 45.859700748326766
    Iteration (131800) - Loss: 45.79253791048373
    Iteration (131900) - Loss: 45.60792190086197
    re to ress."
    
    "I ders the
    bapk rough, excreat eicual never if the peask.'
    
    "At?"
    
    "The. As theel, and in that wouvier ay he wen of poant ve proult no nown
    owind
    nom searly tingo's hope, and is thone
    an
    Iteration (132000) - Loss: 45.46214024211665
    Iteration (132100) - Loss: 45.4279344104374
    Iteration (132200) - Loss: 45.40154950154802
    Iteration (132300) - Loss: 45.38515054084205
    Iteration (132400) - Loss: 45.28363385385508
    Iteration (132500) - Loss: 45.211548694972784
    Iteration (132600) - Loss: 45.042598153446846
    Iteration (132700) - Loss: 45.12270419834326
    Iteration (132800) - Loss: 45.452020408881346
    Iteration (132900) - Loss: 45.93754386786861
    al nottates: He be in are
    swough, rean to wrount makle. I ghats that and fithone
    susiched ind maketsile that was stemponsod. She your droly Street wisegly I word-for the say fult to the sise when -pown
    Iteration (133000) - Loss: 45.89548050930033
    Iteration (133100) - Loss: 46.05316551444582
    Iteration (133200) - Loss: 46.05048352899134
    Iteration (133300) - Loss: 46.27549599353443
    Iteration (133400) - Loss: 46.110508201759856
    Iteration (133500) - Loss: 46.31660037335544
    Iteration (133600) - Loss: 46.35792695649736
    Iteration (133700) - Loss: 46.248126596960375
    Iteration (133800) - Loss: 46.25034772722802
    Iteration (133900) - Loss: 46.186060802412214
    to
    the con prows-sate as upuas and in any
    slell.
    
    "'00 Clumber unsaik
    which we temend."
    
    
    "The paidontringe hourd byed lighture the siled sopoled founsture a ther wring of at ournes all my is in, he fa
    Iteration (134000) - Loss: 46.08573502490278
    Iteration (134100) - Loss: 46.07144405282827
    Iteration (134200) - Loss: 45.840939005840305
    Iteration (134300) - Loss: 45.72115401371528
    Iteration (134400) - Loss: 45.51515649080417
    Iteration (134500) - Loss: 45.53419158791621
    Iteration (134600) - Loss: 45.571255287699245
    Iteration (134700) - Loss: 45.55050627234998
    Iteration (134800) - Loss: 45.45939411022533
    Iteration (134900) - Loss: 45.861964079455895
    , we well is with iever has sprait mo Sas and
    the and hom thinked a wush? It ingak sut it to prilon been
    exfoir I was nan
    awe, sece of a croze jon's at a her with a mast some, his to headed a
    forcamori
    Iteration (135000) - Loss: 46.06174647920017
    Iteration (135100) - Loss: 46.05674252625465
    Iteration (135200) - Loss: 46.38501492024099
    Iteration (135300) - Loss: 46.43234369732589
    Iteration (135400) - Loss: 46.46558176414976
    Iteration (135500) - Loss: 46.906998182186136
    Iteration (135600) - Loss: 46.86685805447734
    Iteration (135700) - Loss: 46.809279228658774
    Iteration (135800) - Loss: 46.81571792097993
    Iteration (135900) - Loss: 46.850774195678945
    ver it
    of I ligkeratnes
    was you sover of what and refinitin refaome."
    
    "'I droupled would the soentibrieg the laence and my sutcer
    eirffors un the thinver," Thankosion bllat yach better threapparyent w
    Iteration (136000) - Loss: 46.73003946916089
    Iteration (136100) - Loss: 46.66430223498779
    Iteration (136200) - Loss: 46.617020917295434
    Iteration (136300) - Loss: 46.46559456652575
    Iteration (136400) - Loss: 46.32112545257593
    Iteration (136500) - Loss: 46.17457819697554
    Iteration (136600) - Loss: 46.299163659442904
    Iteration (136700) - Loss: 46.53357713636519
    Iteration (136800) - Loss: 46.479641091439596
    Iteration (136900) - Loss: 46.58863644427671
     littom pake, ruszing uf?"
    
    "Nom. Iglys face the quitine thaved
    here the chaping at wiun is, ant it a
    pech couruals
    prantuy.
    
    UThilk to ut girryine, to the nomplonied over, but his
    not and
    luan."
    
    "4Ya
    Iteration (137000) - Loss: 47.00799294628415
    Iteration (137100) - Loss: 46.93794270830885
    Iteration (137200) - Loss: 46.720350042604544
    Iteration (137300) - Loss: 46.62915122330583
    Iteration (137400) - Loss: 46.39119898106707
    Iteration (137500) - Loss: 46.413244445027175
    Iteration (137600) - Loss: 46.600518765807365
    Iteration (137700) - Loss: 46.40375344209428
    Iteration (137800) - Loss: 46.62324007203879
    Iteration (137900) - Loss: 46.66102261293277
    e takene exconven ham shand. Womer."
    
    "There of ir
    eners,
    which just
    jus, in gelly heesedfext for gevering
    on that is and opr for a Storned. I
    "'Dad
    with I came on mith
    his lace and hosen to the unced 
    Iteration (138000) - Loss: 46.6798814634038
    Iteration (138100) - Loss: 46.74638040863548
    Iteration (138200) - Loss: 46.72977754085499
    Iteration (138300) - Loss: 46.61298689083392
    Iteration (138400) - Loss: 46.520036504403
    Iteration (138500) - Loss: 46.46604600812078
    Iteration (138600) - Loss: 46.28278218721747
    Iteration (138700) - Loss: 46.50875192670831
    Iteration (138800) - Loss: 46.5947579818829
    Iteration (138900) - Loss: 46.55410569851677
    Here to trlowe, my urog the repioking,
    an detam, than "Vhorbreagenerly Con
    his whres, what he save had primine cut in
    ubsers. What what I'nd is, and onfeigase his hainy your
    Lortsing cuthy a whomenchin
    Iteration (139000) - Loss: 46.39681134953753
    Iteration (139100) - Loss: 46.32877499227452
    Iteration (139200) - Loss: 46.00264391969049
    Iteration (139300) - Loss: 45.78353360796022
    Iteration (139400) - Loss: 45.53602090175095
    Iteration (139500) - Loss: 45.59896649346222
    Iteration (139600) - Loss: 45.70334009515047
    Iteration (139700) - Loss: 45.897325627028096
    Iteration (139800) - Loss: 45.87828187012274
    Iteration (139900) - Loss: 45.85782979764148
    af some site come me upensed, bufter
    soame
    he very."
    
    "At is no
    quames of that you has no Binhirid to, which recay.
    He which to naw it the raor, ver
    drimanttle the, that lias, avatiled pugner, Bmant
    wf
    Iteration (140000) - Loss: 45.81852355647091
    Iteration (140100) - Loss: 45.735079246678325
    Iteration (140200) - Loss: 45.98996558162874
    Iteration (140300) - Loss: 45.96231437258532
    Iteration (140400) - Loss: 46.166166528455676
    Iteration (140500) - Loss: 46.175260876181476
    Iteration (140600) - Loss: 46.03896984170807
    Iteration (140700) - Loss: 45.970766649520186
    Iteration (140800) - Loss: 45.92706337092967
    Iteration (140900) - Loss: 45.84915290244407
    Mall it, municame which,' y fore as Closk?"
    
    "He neventine lids. I to smoant of thiint
    jungoo aining the sritted and the evenched youn eather
    upon in to his deel goury, Whur
    ut wish of
    than his very to
    Iteration (141000) - Loss: 45.927867151444005
    Iteration (141100) - Loss: 45.741064131992914
    Iteration (141200) - Loss: 45.59809850503201
    Iteration (141300) - Loss: 45.54496422690493
    Iteration (141400) - Loss: 45.50703668985677
    Iteration (141500) - Loss: 45.60751128508333
    Iteration (141600) - Loss: 45.60937092927101
    Iteration (141700) - Loss: 45.52203681570995
    Iteration (141800) - Loss: 45.635551031175375
    Iteration (141900) - Loss: 45.69103166217423
    o
    bachen--oblet anters to to you sbee srap the is the maided aAk, all glancuirised opat feing let, not I dresed the. The upotion has the there reash even Holmes
    with Holto ring-pet."
    
    "Mo gluztealy you
    Iteration (142000) - Loss: 45.532986203351676
    Iteration (142100) - Loss: 45.601702428680355
    Iteration (142200) - Loss: 45.45552768996391
    Iteration (142300) - Loss: 45.698124595545536
    Iteration (142400) - Loss: 45.735906116194556
    Iteration (142500) - Loss: 45.62026933958379
    Iteration (142600) - Loss: 45.73591829263834
    Iteration (142700) - Loss: 45.87109221623938
    Iteration (142800) - Loss: 45.793290165617016
    Iteration (142900) - Loss: 45.8635180396556
    tter the of my out in he fornan--oo hadd foring the contif
    me
    to fuer. Ol'ss. Ke. is
    niving had '8rsaftur is.'
    
    LoothinKy ang onninigae. That I hooke and have and oveals, got
    upons
    that orame, and resc
    Iteration (143000) - Loss: 45.94031706429923
    Iteration (143100) - Loss: 45.997892619565086
    Iteration (143200) - Loss: 45.853297641420625
    Iteration (143300) - Loss: 46.10415752712559
    Iteration (143400) - Loss: 46.153547430637424
    Iteration (143500) - Loss: 46.44797654050063
    Iteration (143600) - Loss: 46.516793508073
    Iteration (143700) - Loss: 46.42049374466284
    Iteration (143800) - Loss: 46.49518659730312
    Iteration (143900) - Loss: 46.94276503030927
     to mong indiatth a por to
    muse the fand, as the Pake it
    you arever, was lice
    liggess her of a cosets rid him then heme toon abpere. It smor to
    comn coid,' thim Lome, this. But Lade whion, are two for 
    Iteration (144000) - Loss: 46.93136626118831
    Iteration (144100) - Loss: 47.14859221202238
    Iteration (144200) - Loss: 47.04586420433853
    Iteration (144300) - Loss: 47.08038674121458
    Iteration (144400) - Loss: 47.12131026141947
    Iteration (144500) - Loss: 47.08739720222778
    Iteration (144600) - Loss: 46.82798626417135
    Iteration (144700) - Loss: 46.79767926198807
    Iteration (144800) - Loss: 46.45606644358379
    Iteration (144900) - Loss: 46.33972671547166
     had
    emiatrive astought. And a do aspefan the
    pasing of then the lrow at I ""he Basting a slat on I Nayden for he the creing the
    fin theard had aAdaoken out the sten own. shis wat agree of com bemore m
    Iteration (145000) - Loss: 46.23780192508329
    Iteration (145100) - Loss: 46.19088860922695
    Iteration (145200) - Loss: 45.924411901910695
    Iteration (145300) - Loss: 45.934300108602855
    Iteration (145400) - Loss: 45.88227166595763
    Iteration (145500) - Loss: 46.09340388871163
    Iteration (145600) - Loss: 46.10348569532254
    Iteration (145700) - Loss: 46.13393096825408
    Iteration (145800) - Loss: 46.04552192955192
    Iteration (145900) - Loss: 46.077014654270215
    dmen! Heren inding-Clente refich ane tnon a desreened your to comed my shyientiries, ubloirt himnes he, which at of you abtets upto gripperoonlbought."
    
    "Nindregtrittisgate Ofinghint tastuy tase that t
    Iteration (146000) - Loss: 46.20699373287593
    Iteration (146100) - Loss: 46.14314160771512
    Iteration (146200) - Loss: 45.95373531806879
    Iteration (146300) - Loss: 46.05083168226802
    Iteration (146400) - Loss: 46.24641318483555
    Iteration (146500) - Loss: 46.31171059621342
    Iteration (146600) - Loss: 46.27945595915343
    Iteration (146700) - Loss: 46.3707651640719
    Iteration (146800) - Loss: 46.60319227709758
    Iteration (146900) - Loss: 46.470573270333325
    y, there of gout
    furnog?"
    
    "Now now then of acauld ang iplatay the handoble fally to go rangelvecrat, from, wished wak esirauysombl off
    are chood, with the larder. I who doctition
    onler fataw in
    that
    b
    Iteration (147000) - Loss: 46.59059379531216
    Iteration (147100) - Loss: 46.443007421330094
    Iteration (147200) - Loss: 46.42603853727244
    Iteration (147300) - Loss: 46.27715410926582
    Iteration (147400) - Loss: 45.9988520421198
    Iteration (147500) - Loss: 46.16971394446386
    Iteration (147600) - Loss: 46.092621946168556
    Iteration (147700) - Loss: 46.020699391662106
    Iteration (147800) - Loss: 46.200986916906594
    Iteration (147900) - Loss: 46.476465748997136
    sts.
    
    "Oh, and aty?' ston the bald bach marvelby the lost in
    I jemoring-de' I how couts Brestingly some comitie.'
    
    "Were matse of of had night noke il
    jusshrong
    That anpy looke?"
    
    OfE's in sid paceve
    l
    Iteration (148000) - Loss: 46.438409419552826
    Iteration (148100) - Loss: 46.38394447823979
    Iteration (148200) - Loss: 46.214212361441135
    Iteration (148300) - Loss: 46.12116647288767
    Iteration (148400) - Loss: 45.930900425284335
    Iteration (148500) - Loss: 45.99402804344713
    Iteration (148600) - Loss: 46.066337260990274
    Iteration (148700) - Loss: 45.94592318427575
    Iteration (148800) - Loss: 45.853533404980055
    Iteration (148900) - Loss: 45.816398725744214
    is doan seisther the nle?"
    
    Jannenty now."
    
    "That my sedid ser.
    Holmes that him."
    
    "I
    must for himen?s do nown in into the statyonal.
    I hel?"
    
    "Vave this corniven of cornived so that this and
    which eas
    Iteration (149000) - Loss: 45.659429595599164
    Iteration (149100) - Loss: 45.52038075695798
    Iteration (149200) - Loss: 45.4611975889568
    Iteration (149300) - Loss: 45.442082900958816
    Iteration (149400) - Loss: 45.483818077182846
    Iteration (149500) - Loss: 45.640915043320426
    Iteration (149600) - Loss: 45.63986653774637
    Iteration (149700) - Loss: 45.843002449143846
    Iteration (149800) - Loss: 45.99616454145838
    Iteration (149900) - Loss: 45.7976401517981
    gty. Oh. I have crieg
    I was sontible criel-up tond, bomin,
    I primonlyong os sacked busing wers of pack of kich urled ask iy
    proulon wass tood for this. Have baster with bust bead you
    smaave thin, whess
    Iteration (150000) - Loss: 45.89945063155764
    Iteration (150100) - Loss: 45.90484475750991
    Iteration (150200) - Loss: 45.75034986992244
    Iteration (150300) - Loss: 45.72649362579531
    Iteration (150400) - Loss: 45.546377274329934
    Iteration (150500) - Loss: 45.563913725541816
    Iteration (150600) - Loss: 45.610267510118504
    Iteration (150700) - Loss: 45.656663172830626
    Iteration (150800) - Loss: 45.534102924933464
    Iteration (150900) - Loss: 45.48917447240092
    he chiouln be upqe in ferosh of all on a knor, and which at how ar.
    "A mefpresending St in a coto lo gled suk than veryt prentine pufgelt, and Yound as then I le the unkes had thit solfard
    havainfed wh
    Iteration (151000) - Loss: 45.46698435273174
    Iteration (151100) - Loss: 45.54583140788283
    Iteration (151200) - Loss: 45.552398229896646
    Iteration (151300) - Loss: 45.47231990805676
    Iteration (151400) - Loss: 45.78945021212243
    Iteration (151500) - Loss: 45.75304050360407
    Iteration (151600) - Loss: 46.079835577628394
    Iteration (151700) - Loss: 46.44921258498659
    Iteration (151800) - Loss: 46.494063054100025
    Iteration (151900) - Loss: 46.478755371315664
    d sinity," Cans unticky is fron pint the had was and but. Oh thrat a metion."
    
    "Welpsly
    be altaok to he wres, rut
    a reding, and that he duccon-bagats wisking swad no Leht is exenge his fale a cerkount,
    Iteration (152000) - Loss: 46.38766795759968
    Iteration (152100) - Loss: 46.20783006498807
    Iteration (152200) - Loss: 46.16353866727002
    Iteration (152300) - Loss: 45.949662140639006
    Iteration (152400) - Loss: 45.99127092277372
    Iteration (152500) - Loss: 46.04308843052594
    Iteration (152600) - Loss: 46.03033534536093
    Iteration (152700) - Loss: 45.97455100501913
    Iteration (152800) - Loss: 45.94857562879584
    Iteration (152900) - Loss: 45.77491833950735
    ech the ame, be for expheral for cane, and been werven asseedone that Hordes in
    the hill-bring tollet all a dised of we had ne thick any colernooly to bectatist bectrientaght be for pinging fore the is
    Iteration (153000) - Loss: 45.6361451057006
    Iteration (153100) - Loss: 45.66163211419188
    Iteration (153200) - Loss: 45.83938692501053
    Iteration (153300) - Loss: 45.71625479354743
    Iteration (153400) - Loss: 45.778693786887985
    Iteration (153500) - Loss: 45.7417707274144
    Iteration (153600) - Loss: 45.709798382800464
    Iteration (153700) - Loss: 45.67750093129016
    Iteration (153800) - Loss: 45.453739685170696
    Iteration (153900) - Loss: 45.31113552259079
    s it. I corlough indedicn
    strichalling lest to swert hind now fornemed as of I veen stult.f
    "'Quyt-for sutur to that undealp!' 'Holmet sastion-'sleen for withould hell oas you hourdeny a
    very bands war
    Iteration (154000) - Loss: 45.1989669364128
    Iteration (154100) - Loss: 45.24161138280131
    Iteration (154200) - Loss: 45.32228286438256
    Iteration (154300) - Loss: 45.16819521815991
    Iteration (154400) - Loss: 45.01612810490384
    Iteration (154500) - Loss: 44.853485369281636
    Iteration (154600) - Loss: 44.83799199897837
    Iteration (154700) - Loss: 44.83694829278588
    Iteration (154800) - Loss: 44.8080486637996
    Iteration (154900) - Loss: 44.75302778812545
    I attered than
    clidiemand real. Halry
    down,
    wtoudcher, and the had, wisle to loor."
    
    "Ohe lital yohard not man therr wollow up an ander shay
    been might Stens had to tweekh and the supsed, and his noce 
    Iteration (155000) - Loss: 44.66805797238315
    Iteration (155100) - Loss: 44.611529543736836
    Iteration (155200) - Loss: 44.8832434775433
    Iteration (155300) - Loss: 45.06091068500282
    Iteration (155400) - Loss: 45.36692432891271
    Iteration (155500) - Loss: 45.43552546936506
    Iteration (155600) - Loss: 45.598676932177405
    Iteration (155700) - Loss: 45.58450091511634
    Iteration (155800) - Loss: 45.62411070447952
    Iteration (155900) - Loss: 45.59956152665445
    y toakes whith to aline the upon not of the rowered one excrive no is of the quspione to the
    knous tis of the fruet you a nept to my from somples, besiover man you haid a for Premince-Asoneed?" Im Eced
    Iteration (156000) - Loss: 45.78319344705723
    Iteration (156100) - Loss: 45.75632758748301
    Iteration (156200) - Loss: 45.715131835562325
    Iteration (156300) - Loss: 45.66809801239533
    Iteration (156400) - Loss: 45.59233954469305
    Iteration (156500) - Loss: 45.58451128183761
    Iteration (156600) - Loss: 45.38045708678513
    Iteration (156700) - Loss: 45.20953837473035
    Iteration (156800) - Loss: 45.11327946084969
    Iteration (156900) - Loss: 44.87672119066855
    y we mough.
    
    'I had inneg of the very, your ug until to he reilon all back she yelimand up a fa siokle of we. loceming, buch I cack whec he put he
    may his sten of thing thoulony lite than a confe. Agra
    Iteration (157000) - Loss: 44.91334632364385
    Iteration (157100) - Loss: 44.87436288835823
    Iteration (157200) - Loss: 44.886117218636606
    Iteration (157300) - Loss: 44.947888707819395
    Iteration (157400) - Loss: 45.41863486266289
    Iteration (157500) - Loss: 45.52868287046419
    Iteration (157600) - Loss: 45.5103663348848
    Iteration (157700) - Loss: 45.89461362105233
    Iteration (157800) - Loss: 45.91828384785207
    Iteration (157900) - Loss: 46.01959498864039
    to besifiling we qainodebred inthived, caindelly timack-eate twom how ligasuy, put my inlicbent sumanded of the plong up. When macins."
    
    "Weal," somroubso resing in of tomess."
    
    "This of the bacryven b
    Iteration (158000) - Loss: 46.42262272667364
    Iteration (158100) - Loss: 46.27228777401368
    Iteration (158200) - Loss: 46.2738608049675
    Iteration (158300) - Loss: 46.307326216043656
    Iteration (158400) - Loss: 46.35895095365249
    Iteration (158500) - Loss: 46.111040991670016
    Iteration (158600) - Loss: 46.24037440759387
    Iteration (158700) - Loss: 46.13465474595328
    Iteration (158800) - Loss: 46.02171323954509
    Iteration (158900) - Loss: 45.814832832199876
    iour and riunds oo
    lound was shoughly fids."
    
    "I I drleg. Acmocojy
    been prompan, to hive
    cous!
    Sherle?"
    
    "Bo mirring a pechone all concinging
    owe flook. Suf bettock wurted hil. She
    is there fach sight 
    Iteration (159000) - Loss: 45.74612717433545
    Iteration (159100) - Loss: 45.99266955773931
    Iteration (159200) - Loss: 46.023407348298804
    Iteration (159300) - Loss: 45.96561411520722
    Iteration (159400) - Loss: 46.140200569961344
    Iteration (159500) - Loss: 46.483885923522394
    Iteration (159600) - Loss: 46.34809269021808
    Iteration (159700) - Loss: 46.203776699219915
    Iteration (159800) - Loss: 46.10915951675728
    Iteration (159900) - Loss: 45.93850234195501
    itay puspoid of the quith mmaven sautthonbasing tursiouts geem
    have a evered and os the
    panet youre feal."
    
    Stherul contly affonely baard, g ot heapediwantilg tolwer, with "sher unt inkesly wordingeven
    Iteration (160000) - Loss: 45.96094474541169
    Iteration (160100) - Loss: 45.978098151540735
    Iteration (160200) - Loss: 45.838431355073716
    Iteration (160300) - Loss: 46.05646481120671
    Iteration (160400) - Loss: 46.24389331929218
    Iteration (160500) - Loss: 46.20680446157485
    Iteration (160600) - Loss: 46.205903255741134
    Iteration (160700) - Loss: 46.14999357361926
    Iteration (160800) - Loss: 46.13516826699124
    Iteration (160900) - Loss: 46.0250484598185
    obur licning it howge up yeo, besurtled besexs oa greary upon. You come thoubuarn consead, I maykate-seeptering,
    mearttibed forf."
    Sut
    colmouve brswigh.
    
    "Verusted sherefuly is,
    whrapine, to plite. "I 
    Iteration (161000) - Loss: 45.947144758243155
    Iteration (161100) - Loss: 45.75790982179536
    Iteration (161200) - Loss: 46.064571387407845
    Iteration (161300) - Loss: 46.04441463462876
    Iteration (161400) - Loss: 46.08932546686293
    Iteration (161500) - Loss: 45.984403640229694
    Iteration (161600) - Loss: 45.72575727210285
    Iteration (161700) - Loss: 45.434930711610384
    Iteration (161800) - Loss: 45.221827376733465
    Iteration (161900) - Loss: 44.98360839257514
    le whighelter."
    
    "Oh, Mive the to
    vemary clutcling it was me wher, ace dran," I had ofme me. The frimads fale fall
    allouge. The
    was swear and aboutled anttchan over ingenemesay that not
    draw like befer
    Iteration (162000) - Loss: 45.1460671229997
    Iteration (162100) - Loss: 45.401951599633634
    Iteration (162200) - Loss: 45.43497689247593
    Iteration (162300) - Loss: 45.46546549836333
    Iteration (162400) - Loss: 45.38251756641625
    Iteration (162500) - Loss: 45.355148475764
    Iteration (162600) - Loss: 45.17100524002257
    Iteration (162700) - Loss: 45.44876278199607
    Iteration (162800) - Loss: 45.54164474920611
    Iteration (162900) - Loss: 45.64020933210592
     for that he carpion;
    Stuge. For to Boy. She to the caing one to hell wincel. The sawis into list to the. "Pep ones his abkesting beens but can pent peat it ace, I had thair you alrelgenwort that wrigh
    Iteration (163000) - Loss: 45.65006984582009
    Iteration (163100) - Loss: 45.55717064469919
    Iteration (163200) - Loss: 45.47617181676043
    Iteration (163300) - Loss: 45.37498674483453
    Iteration (163400) - Loss: 45.440629824253456
    Iteration (163500) - Loss: 45.324038323890385
    Iteration (163600) - Loss: 45.170916157651995
    Iteration (163700) - Loss: 45.11682948256048
    Iteration (163800) - Loss: 44.96080550012266
    Iteration (163900) - Loss: 44.98744874703825
    glist rosed beent
    whach owe, night the and they then out with bemage neenwely mury, ond slony the daid, he whow no kedver not man in, Cottle would that the had show werk is conotionseded knewver bemard
    Iteration (164000) - Loss: 45.123054524568914
    Iteration (164100) - Loss: 45.035034052043315
    Iteration (164200) - Loss: 44.97543979326945
    Iteration (164300) - Loss: 45.0800565669487
    Iteration (164400) - Loss: 45.2509033207509
    Iteration (164500) - Loss: 45.071402427823934
    Iteration (164600) - Loss: 45.1375789026357
    Iteration (164700) - Loss: 45.032830310835216
    Iteration (164800) - Loss: 45.30874111414348
    Iteration (164900) - Loss: 45.31198568537726
    e your come and time mi have wheterpon,
    and with was nould which I knentions forthas and of will.
    
    It his creaymed. The amnarly to miner thas indan come the Roriloned, Mr. "He paked of ey seas cane cra
    Iteration (165000) - Loss: 45.29692787591413
    Iteration (165100) - Loss: 45.430496659063834
    Iteration (165200) - Loss: 45.47765400224728
    Iteration (165300) - Loss: 45.44582622347524
    Iteration (165400) - Loss: 45.49125554006426
    Iteration (165500) - Loss: 45.43523872078779
    Iteration (165600) - Loss: 45.48182190388951
    Iteration (165700) - Loss: 45.4285926098678
    Iteration (165800) - Loss: 45.77701381008712
    Iteration (165900) - Loss: 45.6961647852847
     "thim, an) the pricioss bryss intunttiser youn of my rlunce be a comes, bber. To his shad in the served
    bumathae ut mave pasesich. I thandsic1sure the willleded merain, and comserve to he havl the edl
    Iteration (166000) - Loss: 46.04634524092199
    Iteration (166100) - Loss: 46.079980789296755
    Iteration (166200) - Loss: 46.08853777436678
    Iteration (166300) - Loss: 46.256252129549985
    Iteration (166400) - Loss: 46.46067039135081
    Iteration (166500) - Loss: 46.57108435223748
    Iteration (166600) - Loss: 46.71211253243825
    Iteration (166700) - Loss: 46.589984190908176
    Iteration (166800) - Loss: 46.695762056643055
    Iteration (166900) - Loss: 46.667274613783604
    ret in
    this Nuss reaps ale fit, we deal the yeet the where lattle the rean where cellock be a paid herse ser. Holmes that occted she was a storn, when his deen edeite scanding houmey in mituls larn, Lo
    Iteration (167000) - Loss: 46.59673784993148
    Iteration (167100) - Loss: 46.46886175146297
    Iteration (167200) - Loss: 46.23948632367628
    Iteration (167300) - Loss: 45.95195855081958
    Iteration (167400) - Loss: 45.92101908476688
    Iteration (167500) - Loss: 45.739358619324655
    Iteration (167600) - Loss: 45.650081298047695
    Iteration (167700) - Loss: 45.53570222325967
    Iteration (167800) - Loss: 45.52876054070531
    Iteration (167900) - Loss: 45.54978355417235
     dode, and the ittide, whockeen with in to," surl--his sinked agontred wajkan pire whothercamar comes!" I sobing the waft in the ssingly have do rightrearseathing."
    cA, and the matt am aver.'
    
    "Gelbave
    Iteration (168000) - Loss: 45.56073134761066
    Iteration (168100) - Loss: 45.60354871104894
    Iteration (168200) - Loss: 45.614872840982734
    Iteration (168300) - Loss: 45.60277981732068
    Iteration (168400) - Loss: 45.63762677148569
    Iteration (168500) - Loss: 45.74654491587958
    Iteration (168600) - Loss: 45.60992082292311
    Iteration (168700) - Loss: 45.5005148381974
    Iteration (168800) - Loss: 45.723215343501266
    Iteration (168900) - Loss: 45.79889622625769
    
    
    "This site
    out him!" Hy
    mints wattle was startort, wurtiy He coutse for a mad the praw, howesh aamovily it inding notnes upear of a comsping toching, and pran theod, we coustrried hax
    the haz
    proite 
    Iteration (169000) - Loss: 45.809804637255176
    Iteration (169100) - Loss: 45.81157398690926
    Iteration (169200) - Loss: 46.04364424101307
    Iteration (169300) - Loss: 46.071472922241725
    Iteration (169400) - Loss: 46.157492453977994
    Iteration (169500) - Loss: 46.035706733958015
    Iteration (169600) - Loss: 45.93359325065783
    Iteration (169700) - Loss: 45.977598167965525
    Iteration (169800) - Loss: 45.752067302706195
    Iteration (169900) - Loss: 45.569272147440444
    to the sumine midt go dad with ransher at cee pital said-timed were thotter which, bot three, so aich for anty, -he ssoratale. I have have to dising come gloctplt your, and me it iclon dim?" 
    4Sere
    of 
    Iteration (170000) - Loss: 45.727842114221346
    Iteration (170100) - Loss: 45.65786871911778
    Iteration (170200) - Loss: 45.63814485432639
    Iteration (170300) - Loss: 45.92364197628922
    Iteration (170400) - Loss: 46.077764707817266
    Iteration (170500) - Loss: 45.95590903871361
    Iteration (170600) - Loss: 45.927324534607116
    Iteration (170700) - Loss: 45.82808081334506
    Iteration (170800) - Loss: 45.592715785737994
    Iteration (170900) - Loss: 45.52038018396883
    ow and been his then," surt of the plaid winden be was a de-nied shen,
    concled, but are in gaived a teceneringent sod herf-a bin, butur Holmes, Jome be his strent?"
    
    Mreest, and the burn deevose inded.
    Iteration (171000) - Loss: 45.528936357873896
    Iteration (171100) - Loss: 45.56893515157913
    Iteration (171200) - Loss: 45.43816058304182
    Iteration (171300) - Loss: 45.43282996139538
    Iteration (171400) - Loss: 45.381389814696846
    Iteration (171500) - Loss: 45.11453478550216
    Iteration (171600) - Loss: 45.022793658806805
    Iteration (171700) - Loss: 45.003853748073944
    Iteration (171800) - Loss: 44.98628617164954
    Iteration (171900) - Loss: 45.05741231101998
     and to triikopergmater wam pure ha. Thomator I wourding 'Dappe-dear, was so
    that the polent wor, arrasssice
    bany. I anich primes
    ins of 3f condudy,' hat mencilbe, as meaple. Thincard-silence for ous o
    Iteration (172000) - Loss: 45.117530886683895
    Iteration (172100) - Loss: 45.15650265802279
    Iteration (172200) - Loss: 45.2858984702117
    Iteration (172300) - Loss: 45.41147115098954
    Iteration (172400) - Loss: 45.28426258373472
    Iteration (172500) - Loss: 45.39461124140921
    Iteration (172600) - Loss: 45.426619424247384
    Iteration (172700) - Loss: 45.274436690649026
    Iteration (172800) - Loss: 45.205249352355324
    Iteration (172900) - Loss: 45.08058974658945
    hen us a
    cenatle abmes, think,
    and blatimppines avlow, and was of and in to in
    the was the what out."
    
    "No, one mat
    reen is beh. This the co, but het to abdory, thedes time a seston efprapelf; a ceet t
    Iteration (173000) - Loss: 45.14978692933893
    Iteration (173100) - Loss: 45.20299235091609
    Iteration (173200) - Loss: 45.22196711362358
    Iteration (173300) - Loss: 45.090595636604924
    Iteration (173400) - Loss: 45.033463538702065
    Iteration (173500) - Loss: 44.948784005084306
    Iteration (173600) - Loss: 45.1177166725497
    Iteration (173700) - Loss: 45.04305612020178
    Iteration (173800) - Loss: 45.120702357523946
    Iteration (173900) - Loss: 45.3768188754582
    hat Rad thought sumy the
    sher Holmes aster sever, and Cood hemmemed has buce tolsines in latttasy a have alpould and proRly enired cave, which himuE have her up a cre docked on and.
    
    "Nondersiok ave fa
    Iteration (174000) - Loss: 45.362253583460955
    Iteration (174100) - Loss: 45.7891446961176
    Iteration (174200) - Loss: 45.97493259578223
    Iteration (174300) - Loss: 45.98980321026109
    Iteration (174400) - Loss: 45.94176804417591
    Iteration (174500) - Loss: 45.87934693702793
    Iteration (174600) - Loss: 45.623474936847785
    Iteration (174700) - Loss: 45.582941081908665
    Iteration (174800) - Loss: 45.418396562240346
    Iteration (174900) - Loss: 45.43982837837318
    ot?"
    
    "Ye?"
    
    "Yesstro potnep wheres case ond in the on and surs whine hould up wars. Nown my has prear sis so thre of the
    beardnerr or six ttHer.' Had head the jut inseld anas had," fathe right, not-we
    Iteration (175000) - Loss: 45.50707585409081
    Iteration (175100) - Loss: 45.5396343524673
    Iteration (175200) - Loss: 45.52881035448272
    Iteration (175300) - Loss: 45.38156636404957
    Iteration (175400) - Loss: 45.158988563982476
    Iteration (175500) - Loss: 45.05858375550815
    Iteration (175600) - Loss: 45.173692140985715
    Iteration (175700) - Loss: 45.335175614294116
    Iteration (175800) - Loss: 45.307751537755365
    Iteration (175900) - Loss: 45.276191742003796
     ou therly a brotlivie. Dut of you must bucat.
    
    "Noscers to as surd is the face, you sout teary, to could in in he otedf to lom the os mesy to
    so rety case it
    capthis It is as do aw stoidente
    by poren 
    Iteration (176000) - Loss: 45.24259344557699
    Iteration (176100) - Loss: 45.25401899218719
    Iteration (176200) - Loss: 45.277803929322936
    Iteration (176300) - Loss: 45.03770277859883
    Iteration (176400) - Loss: 44.86204660038462
    Iteration (176500) - Loss: 44.849212146231636
    Iteration (176600) - Loss: 44.83915741272973
    Iteration (176700) - Loss: 44.860161424698006
    Iteration (176800) - Loss: 44.69156412833088
    Iteration (176900) - Loss: 44.56606404266612
    his thas that comply by that I ged mish a ve untar a lut me agass, ald tuble preet of tyosetire the, which swed him who
    doint.
    
    "I sed in han Dup a glam lise binly, and you or
    wits I staming it for the
    Iteration (177000) - Loss: 44.47222870182285
    Iteration (177100) - Loss: 44.37322731927541
    Iteration (177200) - Loss: 44.408876006312006
    Iteration (177300) - Loss: 44.38175527596172
    Iteration (177400) - Loss: 44.32157150594709
    Iteration (177500) - Loss: 44.1431945799481
    Iteration (177600) - Loss: 44.198073499680845
    Iteration (177700) - Loss: 44.49066006499553
    Iteration (177800) - Loss: 44.58505182414694
    Iteration (177900) - Loss: 44.81311720092743
     to thick
    who olimange this whoce ancing a froarys rade'r ey her which
    my wat sausting
    holmess shather lied brimentel. Jould stoud the dywancy hill ound, and lone unceop, my wilk tcany I
    dimang, and si
    Iteration (178000) - Loss: 44.95576464308034
    Iteration (178100) - Loss: 45.04328019991097
    Iteration (178200) - Loss: 45.13412955647382
    Iteration (178300) - Loss: 45.085616835087926
    Iteration (178400) - Loss: 45.12119341613851
    Iteration (178500) - Loss: 45.36596818626646
    Iteration (178600) - Loss: 45.261810973625224
    Iteration (178700) - Loss: 45.27589348082058
    Iteration (178800) - Loss: 45.1691510218462
    Iteration (178900) - Loss: 45.159241024090534
     and me a lard opemend shere have a corved or Yever nervegal leal coul ga
    for lase nees. Iters seak concemeve an underingtronycet."
    
    "'Oh a wood we lain. I malding I has Efpance."
    
    "I asteas buinttie, 
    Iteration (179000) - Loss: 45.1623986981246
    Iteration (179100) - Loss: 44.91654804397825
    Iteration (179200) - Loss: 44.779317346126966
    Iteration (179300) - Loss: 44.65063193614743
    Iteration (179400) - Loss: 44.50957771514407
    Iteration (179500) - Loss: 44.597561820983046
    Iteration (179600) - Loss: 44.598647315264
    Iteration (179700) - Loss: 44.553781347565426
    Iteration (179800) - Loss: 44.81295748602295
    Iteration (179900) - Loss: 45.07158369833861
    brits not enie rubiled," suvessed face, very druadingaterans would shooning he dourorttear, is man had upor., which the kecthositan mire not
    exseenout
    only suel abloniot here was prominaldathy here Mrs
    Iteration (180000) - Loss: 45.16411257726586
    Iteration (180100) - Loss: 45.09628945146273
    Iteration (180200) - Loss: 45.46682707933247
    Iteration (180300) - Loss: 45.48260791697999
    Iteration (180400) - Loss: 45.86102722790104
    Iteration (180500) - Loss: 45.96695849835646
    Iteration (180600) - Loss: 45.87538191236694
    Iteration (180700) - Loss: 45.88196270226199
    Iteration (180800) - Loss: 45.92570338119846
    Iteration (180900) - Loss: 45.91796552420747
    ou wound a raking which
    a rightationy
    of the pot its. Ictibred a was sireving, fering when piany?"
    
    "But of a my of futtresen one chilt lust it to was any of
    the ten to seen, I verye explele Spraine wa
    Iteration (181000) - Loss: 45.660113495022706
    Iteration (181100) - Loss: 45.79714826334962
    Iteration (181200) - Loss: 45.68323562333679
    Iteration (181300) - Loss: 45.44880058609383
    Iteration (181400) - Loss: 45.309093979057295
    Iteration (181500) - Loss: 45.279026735906406
    Iteration (181600) - Loss: 45.490953803629445
    Iteration (181700) - Loss: 45.5083045838512
    Iteration (181800) - Loss: 45.55558402999569
    Iteration (181900) - Loss: 45.67773668439198
     antissiday
    at limate-b'gase, who bustr count
    crele used nom that dowe bath the ading as sis?"
    
    "Boing man all ertion, do very is Holmess is thent. A a bontche
    than sineL."
    
    "Jolmes, from in less to
    me
    Iteration (182000) - Loss: 45.975571372406044
    Iteration (182100) - Loss: 45.8611776971486
    Iteration (182200) - Loss: 45.664920319691525
    Iteration (182300) - Loss: 45.55096250899181
    Iteration (182400) - Loss: 45.520804226414555
    Iteration (182500) - Loss: 45.56018427707346
    Iteration (182600) - Loss: 45.455522217604525
    Iteration (182700) - Loss: 45.336253591976245
    Iteration (182800) - Loss: 45.48145630406117
    Iteration (182900) - Loss: 45.75159098456336
    onags to sury! Therrack that the haid holr-sa, which Holmes not at Cearion two
    poibred at pharle it brourtacoraintion dralent
    reef."
    
    un uptardmed some that me, shere as suppeemound
    reppious sain of ho
    Iteration (183000) - Loss: 45.73680313905583
    Iteration (183100) - Loss: 45.700050143378185
    Iteration (183200) - Loss: 45.65429294829181
    Iteration (183300) - Loss: 45.58920661851759
    Iteration (183400) - Loss: 45.50306731682736
    Iteration (183500) - Loss: 45.49176501383682
    Iteration (183600) - Loss: 45.28434764820816
    Iteration (183700) - Loss: 45.65378532279595
    Iteration (183800) - Loss: 45.52802458326003
    Iteration (183900) - Loss: 45.5374973088054
    nd whoches unversenfulnmang Mr minuatagliquy finghalres, wrevway of prosost, and lome at diverrate," sump in, wave have of itone a
    very
    but lowned. "Whath, my wearry a gucs, you we kemperss a ling to m
    Iteration (184000) - Loss: 45.42569389672096
    Iteration (184100) - Loss: 45.24958461560274
    Iteration (184200) - Loss: 44.925561978044215
    Iteration (184300) - Loss: 44.6387331066978
    Iteration (184400) - Loss: 44.559178153308174
    Iteration (184500) - Loss: 44.75031806404723
    Iteration (184600) - Loss: 44.99442832906411
    Iteration (184700) - Loss: 44.942435254990656
    Iteration (184800) - Loss: 44.97728248751648
    Iteration (184900) - Loss: 44.92117557823295
    rom
    Mrnom now, and when -neat
    but of sam eppenteres, to the man relide res to her oroulnserwave me wam a jo come lones behark-nanders. At for he veander. Thes pould the man is broutes my promayfatted f
    Iteration (185000) - Loss: 44.90146080632732
    Iteration (185100) - Loss: 44.78670337159154
    Iteration (185200) - Loss: 45.13282839297622
    Iteration (185300) - Loss: 45.24041852403072
    Iteration (185400) - Loss: 45.18785524760171
    Iteration (185500) - Loss: 45.21037992607043
    Iteration (185600) - Loss: 45.11774382054186
    Iteration (185700) - Loss: 45.08841728838142
    Iteration (185800) - Loss: 44.90936449070327
    Iteration (185900) - Loss: 45.07370919924839
    bagitid ay astace nea injos. Hom to in and atreculticacuing."
    
    "You droker allel. It man,
    youre. O"'
     How
    wad
    injumbely
    would it!" net him in his
    any. "I was mopenit to khich, it will them up of the fy
    Iteration (186000) - Loss: 44.873648808850554
    Iteration (186100) - Loss: 44.695325160056264
    Iteration (186200) - Loss: 44.61804204772858
    Iteration (186300) - Loss: 44.57382865560478
    Iteration (186400) - Loss: 44.62536377664876
    Iteration (186500) - Loss: 44.679813631879796
    Iteration (186600) - Loss: 44.58372795983486
    Iteration (186700) - Loss: 44.64078889964961
    Iteration (186800) - Loss: 44.68416638597035
    Iteration (186900) - Loss: 44.74566646978105
    eall lyak allis tepion ax latwoked you lose my."
    
    "I wa. I with wit
    not and cuit boe." I made time is with toctly littiyeensure was had lmalred hought in fust, and doed as jan's
    we. I conemply pards."
    
    Iteration (187000) - Loss: 44.70173828852461
    Iteration (187100) - Loss: 44.64833415661089
    Iteration (187200) - Loss: 44.75510010488699
    Iteration (187300) - Loss: 44.85947690293245
    Iteration (187400) - Loss: 44.788795931385174
    Iteration (187500) - Loss: 44.85181929684323
    Iteration (187600) - Loss: 44.99516700467736
    Iteration (187700) - Loss: 44.91865980663351
    Iteration (187800) - Loss: 45.007749926407506
    Iteration (187900) - Loss: 45.03882106649496
    ast?'
    
    "Wisked.
    It,
    aith,"'s onced it domberlice, nown on to the
    can of I, that had boted to your ohe it injopmswruch out livzoo. I cappopes samly
    lay to that it."
    
    "'Kanth onf lesten upon to cave made
    Iteration (188000) - Loss: 44.988517648503766
    Iteration (188100) - Loss: 45.0446372758462
    Iteration (188200) - Loss: 45.05485607814512
    Iteration (188300) - Loss: 45.38357642883553
    Iteration (188400) - Loss: 45.50508934340621
    Iteration (188500) - Loss: 45.65743131484945
    Iteration (188600) - Loss: 45.68825518625973
    Iteration (188700) - Loss: 45.732112224492724
    Iteration (188800) - Loss: 46.08820470472185
    Iteration (188900) - Loss: 46.093823437276534
    sh has who shee of the brink, theor,
    shall increck windess band.
    
    "Yell to from
    reianked what forry mothing nes, but a pone he domen, leck for I saught,
    Wased fas antter. In of that. We stamjooded the 
    Iteration (189000) - Loss: 46.26185392353935
    Iteration (189100) - Loss: 46.31365326583996
    Iteration (189200) - Loss: 46.23945818798044
    Iteration (189300) - Loss: 46.23519568598213
    Iteration (189400) - Loss: 46.35635495970706
    Iteration (189500) - Loss: 46.164416856050615
    Iteration (189600) - Loss: 46.0464297158963
    Iteration (189700) - Loss: 45.770040769831155
    Iteration (189800) - Loss: 45.454875174543325
    Iteration (189900) - Loss: 45.494844412245435
    ! Nother with twis beation."
    
    Sil, the rearteran, "8o about in
    fropse to-coiver you has it lave in ee livess. Helpo merers in in Paving he hove
    seatshor tralace
    thone an in stond him to thin. His
    sumar
    Iteration (190000) - Loss: 45.33481127168219
    Iteration (190100) - Loss: 45.18191985959455
    Iteration (190200) - Loss: 45.12477991281557
    Iteration (190300) - Loss: 45.13366137384442
    Iteration (190400) - Loss: 45.148822824780346
    Iteration (190500) - Loss: 45.18494442885393
    Iteration (190600) - Loss: 45.23621106783613
    Iteration (190700) - Loss: 45.208437036956745
    Iteration (190800) - Loss: 45.17551564555723
    Iteration (190900) - Loss: 45.32450688104552
     it mearse the ploker's punsing then of the cers at, and you las the
    sathill mest. Lindell tood the
    rackeary sepperppel, there of than theever Howe, a care than?"
    
    "But the cears doute my dopcut and sa
    Iteration (191000) - Loss: 45.332307035304844
    Iteration (191100) - Loss: 45.194629614911584
    Iteration (191200) - Loss: 45.16094393851031
    Iteration (191300) - Loss: 45.38577824764992
    Iteration (191400) - Loss: 45.390584577470825
    Iteration (191500) - Loss: 45.44604062472055
    Iteration (191600) - Loss: 45.484144582605985
    Iteration (191700) - Loss: 45.7055913653599
    Iteration (191800) - Loss: 45.683216125963774
    Iteration (191900) - Loss: 45.796577473933844
    h Mrssed
    Bot at the I his being!" H
    Los now to
    derackavionsult. Attensor Snow relimatly shair. Notle"'s
    hat About, End., hom. Haw have Men
    withly dry interes, so now not two! Hothough insome, Rustokees
    Iteration (192000) - Loss: 45.63752507908214
    Iteration (192100) - Loss: 45.66707501096874
    Iteration (192200) - Loss: 45.54693451063913
    Iteration (192300) - Loss: 45.40495955381238
    Iteration (192400) - Loss: 45.13133014940612
    Iteration (192500) - Loss: 45.304243949050424
    Iteration (192600) - Loss: 45.22199717844316
    Iteration (192700) - Loss: 45.232342933996826
    Iteration (192800) - Loss: 45.5929086468817
    Iteration (192900) - Loss: 45.625010153259716
    soom oully, stoped having is of a allan with the heising him. And a verull. Aldare a strest
    Caming
    whace bused gat one perisling uplouldang.
    "'Yous Hep underwel
    heard youth, my seiller ardon.'T we shac
    Iteration (193000) - Loss: 45.55856464170206
    Iteration (193100) - Loss: 45.46549959425564
    Iteration (193200) - Loss: 45.41685558228412
    Iteration (193300) - Loss: 45.15183592874859
    Iteration (193400) - Loss: 45.12763331301429
    Iteration (193500) - Loss: 45.165298495944555
    Iteration (193600) - Loss: 45.163911601866104
    Iteration (193700) - Loss: 45.03314370542498
    Iteration (193800) - Loss: 44.96146719777729
    Iteration (193900) - Loss: 44.9546601469966
     drimps only as she luching every popine
    sirfe sone gried."
    
    "Texcent in the was questely fill, to his deen was him even. I prime onss."
    
    "Do ams urseding. The gomar his cwing oo Abpeitter yoo some war
    Iteration (194000) - Loss: 44.76377384345176
    Iteration (194100) - Loss: 44.58663431174761
    Iteration (194200) - Loss: 44.619665960646145
    Iteration (194300) - Loss: 44.58382929528162
    Iteration (194400) - Loss: 44.829693095776186
    Iteration (194500) - Loss: 44.75666721918619
    Iteration (194600) - Loss: 44.93232524041658
    Iteration (194700) - Loss: 45.10811544409544
    Iteration (194800) - Loss: 45.081419954304195
    Iteration (194900) - Loss: 44.99693149216273
    . He spare."
    
    "Well brave to
    my hereancule at of ecfilocts--poftorely I sholemayy promarsceart that of."
    
    Where is frofl, which trestostion that I dust o6
    thock to slint squred anceen
    sad the wide, wer
    Iteration (195000) - Loss: 45.05903369746784
    Iteration (195100) - Loss: 45.041995863879336
    Iteration (195200) - Loss: 44.912701950670055
    Iteration (195300) - Loss: 44.79170805751132
    Iteration (195400) - Loss: 44.6829791559615
    Iteration (195500) - Loss: 44.76843803352811
    Iteration (195600) - Loss: 44.77391214918711
    Iteration (195700) - Loss: 44.75681151654922
    Iteration (195800) - Loss: 44.71619627494328
    Iteration (195900) - Loss: 44.65423379540027
    ather bettatter. I swarry afkting cotonch, I seen refadenvence on to ufon mast it as it, had an a nurtine. The difien of deise intered
    that bemarken frookent no
    mooully some."
    
    "He worting fallen
    you d
    Iteration (196000) - Loss: 44.5305050549396
    Iteration (196100) - Loss: 44.7305535431757
    Iteration (196200) - Loss: 44.741040665327205
    Iteration (196300) - Loss: 44.79691184388285
    Iteration (196400) - Loss: 45.0920274826902
    Iteration (196500) - Loss: 45.08744943328766
    Iteration (196600) - Loss: 45.52809488079228
    Iteration (196700) - Loss: 45.59365389414359
    Iteration (196800) - Loss: 45.59246665734828
    Iteration (196900) - Loss: 45.62770480257481
    sert an har like of a flove were orly over oide the mong tare to were becired table be trush will to had the bell from. Hak a veride that in at 1assidiend (hon form am it is I sectis.r
    had fach bevofe,
    Iteration (197000) - Loss: 45.55304848813133
    Iteration (197100) - Loss: 45.35539868349939
    Iteration (197200) - Loss: 45.275443133791555
    Iteration (197300) - Loss: 45.23345208176359
    Iteration (197400) - Loss: 45.15811085934485
    Iteration (197500) - Loss: 45.17242109173701
    Iteration (197600) - Loss: 45.19417912562136
    Iteration (197700) - Loss: 45.17015050507194
    Iteration (197800) - Loss: 45.01726342954705
    Iteration (197900) - Loss: 44.820738918632735
     the
    loom, quelunny simple in of smild he has nepse sishelich, whed bosin pent which me to
    the cain in a prableayice, kiry intorn at helf cearon, ardisf, ant Inly.
    Nonent, aw ied
    have busssnewlew, and 
    Iteration (198000) - Loss: 44.78551158454727
    Iteration (198100) - Loss: 44.89308844115425
    Iteration (198200) - Loss: 44.904041625627535
    Iteration (198300) - Loss: 44.94795886348303
    Iteration (198400) - Loss: 44.93659305636592
    Iteration (198500) - Loss: 44.863957719268996
    Iteration (198600) - Loss: 44.87112706070336
    Iteration (198700) - Loss: 44.763595986794
    Iteration (198800) - Loss: 44.65140654475476
    Iteration (198900) - Loss: 44.42507084151497
    "
    
    "Iy my paling incinge wordgugey of the doody you.'
    
    "Gmray I my &ron so foondfined dingalds of at he mapone you hat . untelt bul.
    He had no wide /hat your corsperes was antased hourice, a hact would
    Iteration (199000) - Loss: 44.411433364089284
    Iteration (199100) - Loss: 44.505850223697934
    Iteration (199200) - Loss: 44.50733784589577
    Iteration (199300) - Loss: 44.25210695938051
    Iteration (199400) - Loss: 44.16911028008303
    Iteration (199500) - Loss: 44.11060039507005
    Iteration (199600) - Loss: 44.015158006726594
    Iteration (199700) - Loss: 43.9444777518999
    Iteration (199800) - Loss: 44.006704714496934
    Iteration (199900) - Loss: 43.96852631167524
     the scowes pow mesire go laid haid hef youl swortile that with hure which of had that Her he very dishe; havio, ma than the danghen besingwant-oresed the tink
    was are conemen as
    whlwown whethaurione a
    Iteration (200000) - Loss: 43.7779163664087
    Iteration (200100) - Loss: 43.872279183082796
    Iteration (200200) - Loss: 44.17465021105508
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-8-fb71e54c11b9> in <module>()
         29         data_index = 0
         30 
    ---> 31     x, y = preprocessor.get_x_and_y(data_index)
         32     rnn.step(x, y, i)
         33     # update counters
    

    <ipython-input-3-0003e2a5912b> in get_x_and_y(self, index)
         45     def get_x_and_y(self, index):
         46         sequnce_length = configurations['SEQUENTIAL_LENGTH']
    ---> 47         enum_data = self.enumerated_data()
         48         x, y = [], []
         49         end_index = index + sequnce_length
    

    <ipython-input-3-0003e2a5912b> in enumerated_data(self)
         40         for char in list(self.data):
         41             id = dictionary[char]
    ---> 42             enumerated_data.append(id)
         43         return enumerated_data
         44 
    

    KeyboardInterrupt: 


**IMPORTANT NOTE:** The reason behind the **"KeyboardInterrupt"** error will be explained below!

 

 

#### Sampling Sequence

Here are some example generated texts:


```python
x, y = preprocessor.get_x_and_y(0)
```


```python
rnn.sample_sequence(rnn.hidden_layers[-1], x[0], 200)
```

    A lighed a marner upon of
    quine which do is have as lead you mad fard."
    
    "We him flit come lieth
    ythouse, me
    he. You haid is me
    it siett as howed in the
    dich
    to cloot her: If might a prees, fick a lat 
    


```python
rnn.sample_sequence(rnn.hidden_layers[-1], x[0], 200)
```

    And
    Holmes It one you breaniged me the carimy with a goien evidver sourciig, Ir."
    
    "Yes, frarrity offed of segr. As she his bey, shome
    will woave whered pattion, then anthrirging dirn.
    
    "I
    the plooking
    


```python
rnn.sample_sequence(rnn.hidden_layers[-1], x[10], 200)
```

    I may could ome deaeted shoo sived Bake it In
    sove a from hanw that that coal a plone
    streckece."
    
    "It lyanly in it two
    larf-sece to his sur the chalse to enow every pelarselad thin was ihence fall far
    


```python
rnn.sample_sequence(rnn.hidden_layers[-1], x[0], 200)
```

    As, your, whorg his flust for it it I
    addarse.'
    
    "Iy, which was I Holmess in the more strips hied firls very who tos with the rooeted has neAred.
    
    Will of for and, for she with neg licnised we. I am sa
    


```python
rnn.sample_sequence(rnn.hidden_layers[-1], x[0], 200)
```

    A could the macken coled infed the senceetile, I this
    for might I have the forwelreced copour wam to bedise,' said him therissed at to
    dore, Whet and to home, colmank that a upow masinet tinds bitter s
    


```python
rnn.sample_sequence(rnn.hidden_layers[-1], x[0], 200)
```

    Accouring to be ofraist
    om mith extell it whom?
    
    "'You have what, the enverge of Rupl to it or up. That have a viesal upon she like had note, If loode that at
    an for and couft you wis, ber that my his
    
    

 

 

**Comment:** My code works slower than some library implementations. I think it is because of its object-oriented architecture. But when the iteration increases, the quality of the generated texts are significantly increases. I planned to iterate 500K time to achieve a reasonable result. However, the code works slowly and I needed to submit the code before 23:59. So, I needed to stop iteration at 200K. That is why Keybord Interruption Error occured above. <br>

So, I believe that if I did not stop iterations, the quality of words will be much more than this. But still, most of the words are understandable.

 

 
