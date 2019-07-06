

```python
import numpy as np
import keras
from matplotlib import pyplot as plt
import matplotlib as mpl
```

    Using TensorFlow backend.
    


```python
from keras.models import Sequential
from keras.layers.core import Dense
from keras import layers
from keras import metrics
from keras import losses
from keras import optimizers
from keras import regularizers
from keras.utils.np_utils import to_categorical
```


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
```

### Helper Functions


```python
# plotting
        
def plot_classification(histories, c_type = 'binary'):
    acc_metric = 'binary_accuracy'
    if c_type == 'multiclass':
        acc_metric = 'acc'
    num_fold = len(histories)
    folds_train_accs   = []
    folds_valid_accs   = []
    folds_train_losses = []
    folds_valid_losses  = []
    for fold in range(num_fold):
        history = histories[fold]
        train_accs = history.history[acc_metric]
        valid_accs = history.history['val_{0}'.format(acc_metric)]
        train_losses = history.history['loss']
        valid_losses = history.history['val_loss']
        folds_train_accs.append(train_accs)
        folds_valid_accs.append(valid_accs)
        folds_train_losses.append(train_losses)
        folds_valid_losses.append(valid_losses)
    num_epochs = len(folds_train_accs[0])
    # print('Number of epochs: {0}'.format(num_epochs))
    # compute average values over folds
    avg_fold_train_accs_per_epoch   = [np.mean([x[i] for x in folds_train_accs]) for i in range(num_epochs)]
    avg_fold_train_losses_per_epoch = [np.mean([x[i] for x in folds_train_losses]) for i in range(num_epochs)]
    avg_fold_valid_accs_per_epoch   = [np.mean([x[i] for x in folds_valid_accs]) for i in range(num_epochs)]
    avg_fold_valid_losses_per_epoch = [np.mean([x[i] for x in folds_valid_losses]) for i in range(num_epochs)]
    plot_accuracy(avg_fold_train_accs_per_epoch, avg_fold_valid_accs_per_epoch, 'Average Train/Validation Accuracies per epoch')
    plot_loss(avg_fold_train_losses_per_epoch, avg_fold_valid_losses_per_epoch, 'Average Train/Validation Losses per epoch')
    #plot_accuracy(avg_fold_train_accs_per_epoch, 'Average Train Accuracies per epoch')
    #plot_accuracy(avg_fold_valid_accs_per_epoch, 'Average Validation Accuracies per epoch')
    #plot_loss(avg_fold_train_losses_per_epoch, 'Average Train Losses per epoch')
    #plot_loss(avg_fold_valid_losses_per_epoch, 'Average Validation Losses per epoch')
    
    
    

def plot_regression(histories, metric_name):
    num_fold = len(histories)
    folds_train_errs  = []
    folds_valid_errs  = []
    folds_train_losses = []
    folds_valid_losses  = []
    for fold in range(num_fold):
        history = histories[fold]
        train_errs = history.history[metric_name]
        valid_errs = history.history['val_{0}'.format(metric_name)]
        train_losses = history.history['loss']
        valid_losses = history.history['val_loss']
        folds_train_errs.append(train_errs)
        folds_valid_errs.append(valid_errs)
        folds_train_losses.append(train_losses)
        folds_valid_losses.append(valid_losses)
    num_epochs = len(folds_train_errs[0])
     # compute average values over folds
    avg_fold_train_errs_per_epoch   = [np.mean([x[i] for x in folds_train_errs]) for i in range(num_epochs)]
    avg_fold_train_losses_per_epoch = [np.mean([x[i] for x in folds_train_losses]) for i in range(num_epochs)]
    avg_fold_valid_errs_per_epoch   = [np.mean([x[i] for x in folds_valid_errs]) for i in range(num_epochs)]
    avg_fold_valid_losses_per_epoch = [np.mean([x[i] for x in folds_valid_losses]) for i in range(num_epochs)]
    plot_error(avg_fold_train_errs_per_epoch, avg_fold_valid_errs_per_epoch, 'Average Train/Validation Errors per epoch', metric_name)
    plot_loss(avg_fold_train_losses_per_epoch, avg_fold_valid_losses_per_epoch, 'Average Train/Validation Losses per epoch')
    
    #plot_error(avg_fold_train_errs_per_epoch, 'Average Train Accuracies per epoch', metric_name)
    #plot_error(avg_fold_valid_errs_per_epoch, 'Average Validation Accuracies per epoch', metric_name)
    #plot_loss(avg_fold_train_losses_per_epoch, 'Average Train Losses per epoch')
    #plot_loss(avg_fold_valid_losses_per_epoch, 'Average Validation Losses per epoch')
    

def plot_accuracy(train_accs, valid_accs, title):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, color='C0')
    ax.plot(train_accs, 'C1', label='train accuracy')
    ax.plot(valid_accs, 'C2', label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.legend()
    plt.show()

    
def plot_loss(train_losses, test_losses, title):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, color='C0')
    ax.plot(train_losses, 'C2', label='train loss')
    ax.plot(test_losses, 'C3', label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.legend()
    plt.show()
    
    
def plot_error(train_errors, validaiton_errors, title, error_type):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, color='C0')
    ax.plot(train_errors, 'C2', label='{0} on train set'.format(error_type))
    ax.plot(validaiton_errors, 'C3', label='{0} on validation set'.format(error_type))
    plt.xlabel('Epoch')
    plt.ylabel(error_type)
    ax.legend()
    plt.show()
```

 

 

## QUESTION-1

[Binary Classification] [40 pts] Use UCI’s sentiment
dataset (https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) to
perform binary classification to classify reviews into positive or
negative. Use k-fold cross validation and show loss/accuracy plots by
epoch.


```python
def get_data():
    sentences = []
    targets = []
    base_folder = 'sentiment_labelled_sentences'
    text_file_names = ['amazon_cells_labelled', 'yelp_labelled', 'imdb_labelled']
    for text_file_name in text_file_names:
        # read txt file
        with open('{0}/{1}.txt'.format(base_folder, text_file_name), 'r') as file:
            text = file.read()
        for instance in text.split('\n'):
            if len(instance.split('\t')) == 2:
                sentence = instance.split('\t')[0]
                target = instance.split('\t')[1]
                sentences.append(sentence)
                targets.append(target)
    return sentences, targets
```


```python
sentences, targets = get_data()
```


```python
print('Number of sentences: {0}'.format(len(sentences)))
```

    Number of sentences: 3000
    


```python
# vectorizer to create a vocabulary from the words
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_
```




    {'smeared': 5016,
     'further': 3136,
     'wild': 5797,
     'EVERYONE': 454,
     'Tied': 1367,
     'tickets': 5400,
     'forever': 3080,
     'LG': 800,
     'because': 1842,
     'voice': 5692,
     'worst': 5845,
     'steakhouse': 5137,
     'spends': 5088,
     'Soggy': 1249,
     'kidnapped': 3619,
     'lilt': 3719,
     'black': 1895,
     'pull': 4483,
     'ActiveSync': 98,
     'failed': 2932,
     'BROKE': 183,
     'line': 3722,
     'Integrated': 726,
     'Special': 1269,
     'rather': 4538,
     'tones': 5431,
     'stuffed': 5191,
     'fast': 2960,
     'ride': 4696,
     'Terrible': 1340,
     'build': 1997,
     'Metro': 910,
     'Sean': 1212,
     'Due': 442,
     'loud': 3766,
     'prefer': 4402,
     'dipping': 2586,
     'delivery': 2520,
     'More': 925,
     'seeingÂ': 4842,
     'edge': 2751,
     'Earpiece': 462,
     'fuzzy': 3138,
     'unbelievable': 5534,
     'throughout': 5391,
     'simmering': 4960,
     'ask': 1737,
     'themes': 5357,
     'AWESOME': 85,
     'cheeseburger': 2143,
     'deadpan': 2473,
     'shall': 4894,
     'Attack': 155,
     'likes': 3716,
     'Unfortunately': 1418,
     'GO': 579,
     'eventually': 2849,
     'holding': 3371,
     'arepas': 1719,
     'guy': 3258,
     'thinking': 5371,
     'Bell': 215,
     'legal': 3686,
     'we': 5750,
     'Enough': 487,
     'dissapointed': 2627,
     'suited': 5234,
     'leaf': 3674,
     'accurate': 1574,
     'might': 3903,
     'MGM': 867,
     'content': 2325,
     'exceeds': 2865,
     'Nevertheless': 959,
     'fliptop': 3048,
     'Falwell': 525,
     'developments': 2561,
     'Lot': 849,
     'banana': 1809,
     'Was': 1483,
     'kudos': 3636,
     'scientist': 4803,
     'overcooked': 4168,
     'horrible': 3394,
     'anyway': 1693,
     'HBO': 632,
     'grey': 3237,
     'biographical': 1884,
     'extensive': 2911,
     'ridiculous': 4697,
     'punched': 4487,
     'age': 1620,
     'sobering': 5033,
     'make': 3805,
     'Lino': 837,
     'atrocity': 1755,
     'Another': 133,
     'recharge': 4577,
     'smartphone': 5015,
     'as': 1735,
     'slid': 4997,
     'Angel': 125,
     'inch': 3475,
     'Non': 973,
     '25': 32,
     'earth': 2741,
     'performed': 4268,
     'distract': 2631,
     'Wienerschnitzel': 1509,
     'game': 3145,
     'OR': 986,
     'bells': 1864,
     'feet': 2977,
     'speedy': 5086,
     'brilliantly': 1974,
     'boxes': 1953,
     '40min': 42,
     'mortified': 3956,
     '375': 39,
     'regularly': 4608,
     'Reversible': 1144,
     'perhaps': 4270,
     'Fire': 540,
     '13': 5,
     'loudspeaker': 3770,
     'class': 2182,
     'national': 4007,
     'celluloid': 2096,
     'hoot': 3386,
     'owners': 4185,
     'ok': 4105,
     'basement': 1821,
     'spice': 5092,
     'iGo': 3430,
     'cliche': 2192,
     'wallet': 5713,
     'FOOLED': 516,
     'wear': 5753,
     'JX': 745,
     'handled': 3275,
     'humorous': 3420,
     'be': 1831,
     'know': 3633,
     'BARGAIN': 171,
     'usable': 5620,
     'Palance': 1035,
     'green': 3233,
     'app': 1699,
     'quick': 4514,
     'treo': 5486,
     'charismatic': 2125,
     'cutting': 2454,
     'lies': 3705,
     'product': 4442,
     'audience': 1766,
     'paint': 4203,
     'Someone': 1252,
     'heroine': 3342,
     'BITCHES': 177,
     'Valley': 1438,
     'fi': 2988,
     'Three': 1362,
     'Never': 958,
     'moz': 3976,
     'Ever': 498,
     'arrived': 1725,
     'touch': 5447,
     'checking': 2137,
     'describing': 2532,
     'bars': 1817,
     'Those': 1360,
     'products': 4444,
     'OK': 983,
     'excuses': 2882,
     'item': 3576,
     'Cant': 294,
     'tolerance': 5427,
     'trust': 5505,
     'Friends': 570,
     'FANTASTIC': 511,
     'avoided': 1778,
     'WHITE': 1467,
     'catchy': 2082,
     'Media': 904,
     'elegant': 2772,
     'haunt': 3303,
     'flick': 3043,
     'Storm': 1292,
     'into': 3556,
     'PS3': 1031,
     'Nargile': 952,
     'huge': 3411,
     'modern': 3932,
     'FREEZING': 520,
     'sand': 4765,
     'Philadelphia': 1056,
     'fairly': 2935,
     'performing': 4269,
     'ft': 3121,
     'Obviously': 990,
     'did': 2571,
     'Jenni': 759,
     'shipping': 4919,
     'wobbly': 5821,
     'To': 1373,
     'Art': 147,
     'syrupy': 5278,
     'educational': 2755,
     'evil': 2858,
     'constantly': 2312,
     'impeccable': 3460,
     'headsets': 3313,
     'scary': 4794,
     'destroy': 2552,
     'market': 3824,
     'CafÃ': 283,
     'pecan': 4253,
     'kept': 3611,
     'hairsplitting': 3266,
     'prettier': 4416,
     'travled': 5478,
     'gloves': 3191,
     'larger': 3651,
     'James': 751,
     'bulky': 2002,
     'Martin': 894,
     'flash': 3028,
     'zero': 5889,
     'extra': 2915,
     'bargain': 1814,
     'Initially': 723,
     'pretentious': 4414,
     'overnite': 4173,
     'New': 961,
     'bohemian': 1921,
     'cross': 2427,
     'add': 1597,
     'crowds': 2430,
     'fest': 2986,
     'keep': 3608,
     'Signal': 1235,
     'Juano': 773,
     'excellently': 2867,
     'Cailles': 284,
     'Paying': 1044,
     '1979': 18,
     'Valentine': 1437,
     'Crema': 367,
     'occurs': 4089,
     'Good': 607,
     'jabra': 3579,
     'perplexing': 4275,
     'lazy': 3671,
     'functional': 3130,
     'Try': 1400,
     '80': 59,
     'distant': 2628,
     'Also': 114,
     'meant': 3855,
     'You': 1539,
     'isn': 3571,
     'End': 485,
     'colleague': 2218,
     'power': 4392,
     'advertised': 1609,
     'sounds': 5069,
     'hopeless': 3389,
     'TARDIS': 1314,
     'Cibo': 325,
     'recommended': 4583,
     'handsfree': 3281,
     'unwelcome': 5605,
     'NYC': 950,
     'Olde': 996,
     'quantity': 4511,
     'slipping': 5005,
     'glasses': 3188,
     'dads': 2455,
     'schoolers': 4800,
     'yay': 5868,
     'minor': 3914,
     'there': 5362,
     'Walkman': 1479,
     'IN': 706,
     'replacement': 4650,
     'seafood': 4820,
     'Mediocre': 905,
     'politics': 4362,
     'keypad': 3614,
     'knock': 3631,
     'Jamie': 752,
     'fellow': 2981,
     '18th': 11,
     'sentiment': 4863,
     'parties': 4226,
     'Total': 1385,
     'goalies': 3195,
     'flawed': 3038,
     'fo': 3058,
     'quiet': 4517,
     'Shakespears': 1225,
     'flawlessly': 3040,
     'bus': 2009,
     'Dog': 430,
     'earbud': 2729,
     'Chicken': 316,
     'clock': 2201,
     'RAZR': 1110,
     'wish': 5813,
     'capacity': 2050,
     'stop': 5157,
     'loudest': 3768,
     'She': 1227,
     'empowerment': 2792,
     'size': 4985,
     'Howe': 695,
     'order': 4134,
     'muddled': 3981,
     'messes': 3894,
     'ribeye': 4693,
     'faultless': 2964,
     'sensitivities': 4860,
     'Texas': 1341,
     'Dustin': 445,
     'next': 4035,
     'r450': 4520,
     'keys': 3616,
     'directed': 2588,
     'gimmick': 3176,
     'ever': 2850,
     'correctly': 2364,
     'our': 4148,
     'extraordinary': 2917,
     'FX': 523,
     'bowl': 1951,
     'sack': 4747,
     'DEAD': 382,
     'pens': 4259,
     'mic': 3899,
     'REALLY': 1112,
     'cable': 2023,
     'profound': 4449,
     'wont': 5828,
     'weeks': 5763,
     'PLANE': 1026,
     'modest': 3933,
     'known': 3634,
     'pi': 4293,
     'Lots': 850,
     'gently': 3166,
     'Hoping': 687,
     'Poorly': 1083,
     'Excellent': 506,
     'lieutenant': 3706,
     'imagination': 3454,
     'disagree': 2598,
     'Tonight': 1378,
     'batteries': 1829,
     'happen': 3286,
     'verbal': 5654,
     'peaking': 4247,
     'guests': 3255,
     'views': 5672,
     'counter': 2372,
     'mainly': 3798,
     'pleasantly': 4331,
     'legendary': 3687,
     'direction': 2590,
     'engineered': 2807,
     'idiot': 3443,
     'promote': 4454,
     'High': 673,
     'eaten': 2747,
     'SETS': 1167,
     'Jonah': 770,
     'vacant': 5635,
     'linking': 3726,
     'retarded': 4677,
     'old': 4106,
     'Wife': 1510,
     'such': 5219,
     'He': 659,
     'exemplars': 2884,
     'LOVE': 803,
     'freedom': 3099,
     'does': 2644,
     'affordable': 1614,
     'EarGels': 460,
     'overly': 4171,
     'nearly': 4016,
     'From': 571,
     'Curve': 375,
     'feature': 2971,
     'crack': 2389,
     'Miner': 916,
     'snap': 5029,
     'portrayed': 4376,
     'used': 5623,
     'moment': 3938,
     'operates': 4124,
     'problems': 4431,
     'fantastic': 2951,
     'Stopped': 1291,
     'researched': 4659,
     'Worth': 1530,
     'People': 1046,
     'cow': 2385,
     'thinly': 5372,
     'Cool': 354,
     'hurry': 3424,
     'brutal': 1989,
     'Has': 650,
     'program': 4450,
     'continually': 2326,
     'ease': 2742,
     'miserable': 3918,
     'seasoning': 4828,
     'Large': 812,
     'sugary': 5230,
     'stratus': 5169,
     'NOBODY': 944,
     'employees': 2791,
     'FORWARD': 519,
     'grtting': 3250,
     'venturing': 5652,
     'shameful': 4897,
     'AZ': 88,
     'exactly': 2861,
     'loved': 3774,
     'Damian': 391,
     'Bertolucci': 222,
     'paced': 4191,
     'As': 149,
     'CA': 271,
     'frog': 3112,
     'promptly': 4457,
     'Daughter': 394,
     'live': 3735,
     'tell': 5327,
     'yourself': 5881,
     'bec': 1840,
     'only': 4116,
     'comparably': 2256,
     'unique': 5574,
     'Sci': 1205,
     'atmosphere': 1753,
     'Your': 1541,
     'yeah': 5869,
     'none': 4051,
     'at': 1750,
     'people': 4260,
     'list': 3727,
     'highly': 3354,
     'guest': 3254,
     'puree': 4497,
     'FLY': 514,
     'consistent': 2309,
     'jerky': 3586,
     'inexplicable': 3505,
     'meaning': 3852,
     'unintelligible': 5570,
     'charge': 2118,
     'Fox': 557,
     'jobs': 3591,
     'SIM': 1169,
     'drawings': 2683,
     'typical': 5524,
     'twirling': 5518,
     'timing': 5410,
     'couples': 2374,
     'No': 969,
     'treat': 5481,
     'random': 4528,
     'drama': 2678,
     'happiness': 3291,
     'Being': 213,
     'Drinks': 439,
     'plug': 4342,
     'Street': 1296,
     'max': 3842,
     'megapixels': 3866,
     'sold': 5038,
     'government': 3213,
     'communication': 2250,
     'worthy': 5849,
     'composed': 2272,
     'Turkey': 1404,
     'moods': 3951,
     'cry': 2435,
     'Hence': 668,
     'filmography': 3001,
     'Carrell': 302,
     'potato': 4386,
     'Makes': 883,
     'Lousy': 851,
     'EnV': 484,
     'sore': 5062,
     'drawback': 2681,
     'cash': 2072,
     'Perhaps': 1051,
     'others': 4145,
     'chance': 2108,
     'Bussell': 265,
     'Baseball': 200,
     'videos': 5668,
     'wrapped': 5857,
     'unintentionally': 5571,
     'Some': 1251,
     'American': 120,
     'helpful': 3333,
     'drenched': 2689,
     'martin': 3828,
     'decade': 2482,
     'save': 4780,
     'majority': 3804,
     'may': 3843,
     'momentum': 3940,
     'Things': 1353,
     'Baxendale': 205,
     'charm': 2126,
     'Work': 1525,
     'entertaining': 2817,
     'soldiers': 5039,
     'Plantronincs': 1071,
     'cant': 2048,
     'pastas': 4233,
     'Great': 616,
     'mid': 3901,
     'Technically': 1337,
     'peanut': 4248,
     'inexcusable': 3502,
     'laughable': 3664,
     'Tale': 1332,
     'strap': 5168,
     'terrible': 5338,
     'Bailey': 195,
     'inviting': 3562,
     'true': 5500,
     'Lassie': 814,
     'gifted': 3175,
     'tried': 5489,
     'eyes': 2921,
     'appreciate': 1712,
     'Absolutel': 92,
     'dude': 2713,
     'grace': 3215,
     'got': 3211,
     'THEY': 1320,
     'Zombiez': 1543,
     'akin': 1632,
     'shallow': 4895,
     'pants': 4216,
     'type': 5523,
     'support': 5247,
     'unaccompanied': 5531,
     'summer': 5238,
     'simple': 4961,
     'least': 3679,
     'imaginable': 3453,
     'Comfort': 340,
     'yelps': 5875,
     'Melville': 908,
     'satisfying': 4774,
     'Florida': 547,
     'taken': 5287,
     'dropped': 2706,
     'Titta': 1372,
     'are': 1716,
     'honestly': 3382,
     'Coffee': 334,
     'device': 2562,
     'seen': 4846,
     'taxidermists': 5311,
     'pleasure': 4336,
     'crowd': 2429,
     'whites': 5785,
     'chip': 2161,
     'goat': 3196,
     'Motorolas': 930,
     'sangria': 4768,
     'unconditional': 5538,
     'Charles': 312,
     'Ben': 219,
     'enjoyable': 2810,
     'fella': 2980,
     'FINALLY': 512,
     'played': 4325,
     'peach': 4245,
     'timely': 5407,
     'crepe': 2418,
     'peas': 4252,
     'Jessica': 762,
     'Go': 604,
     'produce': 4438,
     'Latin': 820,
     'appealing': 1702,
     'diabetic': 2565,
     'somewhere': 5052,
     'manages': 3817,
     'sketchy': 4987,
     'crocs': 2426,
     'purpose': 4499,
     'chef': 2147,
     '20th': 28,
     'qualified': 4508,
     'added': 1598,
     'backdrop': 1793,
     'mollusk': 3936,
     'club': 2205,
     'pace': 4190,
     'shouting': 4936,
     'metal': 3895,
     'opened': 4119,
     'idea': 3437,
     'Think': 1354,
     'running': 4742,
     'author': 1771,
     'producer': 4440,
     'Ample': 122,
     'folks': 3062,
     'Ponyo': 1081,
     'fisted': 3020,
     'snug': 5030,
     'peachy': 4246,
     'Las': 813,
     'equivalent': 2828,
     'PDA': 1019,
     'argued': 1720,
     'detachable': 2554,
     'puff': 4482,
     'Ebay': 466,
     'uses': 5628,
     'unpredictability': 5590,
     'colored': 2223,
     'owns': 4187,
     'infuriating': 3511,
     'Artless': 148,
     'Nevsky': 960,
     'Songs': 1253,
     'vomit': 5697,
     'Beware': 226,
     'video': 5667,
     'entire': 2820,
     'school': 4799,
     'submerged': 5208,
     'friends': 3108,
     'anguish': 1676,
     'expected': 2893,
     'Universal': 1419,
     'fits': 3022,
     'love': 3773,
     'certainly': 2103,
     'out': 4150,
     'EXPERIENCE': 456,
     'rips': 4709,
     'basic': 1822,
     '15g': 7,
     'shower': 4940,
     'always': 1653,
     'reccomendation': 4566,
     'bartenders': 1819,
     'camera': 2037,
     'Simply': 1238,
     'ineptly': 3501,
     'theatrical': 5352,
     'falling': 2939,
     'Frances': 559,
     'dreary': 2688,
     'nut': 4077,
     'SUPERB': 1181,
     'swivel': 5272,
     'Keep': 783,
     'potted': 4389,
     'recently': 4574,
     'Lately': 818,
     'occur': 4088,
     'explosion': 2906,
     'come': 2230,
     '510': 50,
     'puzzle': 4505,
     'without': 5817,
     'holds': 3372,
     'Tomorrow': 1377,
     'Att': 154,
     'rate': 4536,
     'ROAD': 1117,
     'locked': 3746,
     'third': 5373,
     'Don': 431,
     'America': 119,
     'identified': 3440,
     'unless': 5580,
     'been': 1847,
     'inexpensive': 3503,
     'Flynn': 549,
     'Errol': 491,
     'weren': 5770,
     'anything': 1691,
     'US': 1410,
     'insipid': 3519,
     'anytime': 1692,
     'passed': 4229,
     'community': 2252,
     'aired': 1629,
     'photograph': 4289,
     'realized': 4558,
     'usually': 5631,
     'owned': 4183,
     'mp3s': 3979,
     'sooner': 5058,
     'HAVE': 631,
     'connecting': 2298,
     'Waiting': 1477,
     'twists': 5520,
     'exteriors': 2913,
     'repeats': 4646,
     'detailing': 2556,
     'washing': 5732,
     'cheaply': 2132,
     'us': 5619,
     'leave': 3681,
     'confortable': 2293,
     'bland': 1899,
     'mistakes': 3926,
     'famed': 2943,
     'provide': 4468,
     'sucks': 5223,
     'lack': 3638,
     'Indian': 722,
     'focused': 3060,
     'sign': 4950,
     'choice': 2165,
     'melted': 3872,
     'realised': 4554,
     'below': 1866,
     'hitch': 3365,
     'Tracfone': 1388,
     'SPEAKERPHONE': 1172,
     'glove': 3190,
     'sometimes': 5050,
     'number': 4072,
     'develop': 2559,
     'intrigued': 3558,
     'LEGIT': 799,
     'see': 4839,
     'bit': 1889,
     'apart': 1695,
     'case': 2070,
     'seating': 4831,
     'normally': 4057,
     'amusing': 1668,
     'annoying': 1681,
     'soooooo': 5060,
     'cradle': 2392,
     'photo': 4288,
     'block': 1906,
     'IQ': 709,
     'Returned': 1142,
     'signals': 4952,
     'refuse': 4600,
     'renders': 4638,
     'Though': 1361,
     'procedure': 4433,
     'pair': 4205,
     'voted': 5699,
     'Duris': 444,
     'coconut': 2213,
     'delicate': 2508,
     'granted': 3220,
     'bat': 1824,
     'extended': 2910,
     'beeping': 1849,
     'smelled': 5018,
     'desperately': 2544,
     'resistant': 4661,
     'explanation': 2903,
     'interplay': 3554,
     'compelling': 2258,
     'lose': 3761,
     'shouldn': 4934,
     'believable': 1860,
     'haul': 3302,
     '1998': 22,
     'dry': 2710,
     'vivid': 5689,
     'Hayao': 657,
     'Transmitters': 1390,
     'Plus': 1078,
     'warts': 5728,
     'Editing': 473,
     'cuts': 2453,
     'Arrived': 146,
     'inconsistencies': 3482,
     'tribute': 5487,
     'waited': 5704,
     'section': 4835,
     'house': 3406,
     'Voodoo': 1452,
     'Europe': 494,
     'trying': 5507,
     'stated': 5129,
     'English': 486,
     'concert': 2283,
     'match': 3838,
     'volcano': 5694,
     'Doctor': 426,
     'stable': 5110,
     'FLAVOR': 513,
     'disgraceful': 2614,
     'weight': 5764,
     'non': 4050,
     'sight': 4949,
     'unusable': 5603,
     'layers': 3670,
     'Vegas': 1442,
     'integration': 3538,
     'sabotages': 4746,
     'costumes': 2367,
     'texture': 5343,
     'Up': 1422,
     'disappointing': 2602,
     'less': 3692,
     'animation': 1679,
     'BLACK': 178,
     'wise': 5812,
     'impossible': 3463,
     'looking': 3754,
     'perpared': 4274,
     'recognition': 4579,
     'Ebola': 467,
     'slim': 5003,
     'dish': 2618,
     'hatred': 3301,
     'horrid': 3395,
     'receipt': 4568,
     'leather': 3680,
     'relationship': 4615,
     'sturdy': 5196,
     'Internet': 728,
     'tight': 5402,
     'Amazon': 117,
     'Third': 1355,
     'funniest': 3134,
     'reenactments': 4591,
     'Echo': 469,
     'alongside': 1647,
     'Coming': 342,
     'instant': 3527,
     'actors': 1589,
     'focus': 3059,
     'E715': 449,
     'Bought': 247,
     'actually': 1593,
     'primary': 4426,
     'nationalities': 4008,
     'aged': 1621,
     'cancellation': 2043,
     'Mary': 895,
     'Plantronics': 1070,
     'poetry': 4351,
     'musician': 3992,
     'Belmondo': 218,
     'Casino': 305,
     'bring': 1975,
     'relocated': 4626,
     'reminds': 4633,
     'Emily': 483,
     'freaking': 3097,
     'hybrid': 3427,
     'sound': 5067,
     'WORTHWHILE': 1472,
     'MOTO': 868,
     'reporter': 4653,
     'greeted': 3235,
     'fat': 2962,
     'wagyu': 5702,
     'mouthful': 3967,
     'sliding': 5000,
     'StarTac': 1278,
     'Fall': 524,
     'spoiler': 5101,
     'Of': 991,
     'professional': 4445,
     'unnecessary': 5586,
     'Frankly': 562,
     'creative': 2412,
     'balanced': 1805,
     'Uncomfortable': 1416,
     'fulfills': 3124,
     'Better': 224,
     'Wirefly': 1516,
     'Plan': 1069,
     'chipolte': 2162,
     'carpaccio': 2061,
     'Wayne': 1491,
     'transfer': 5468,
     'thirty': 5374,
     'ed': 2750,
     'side': 4946,
     'finished': 3015,
     'vanilla': 5639,
     'May': 900,
     'Does': 428,
     'gallon': 3144,
     'Telly': 1339,
     'Reconciliation': 1133,
     'unfunny': 5566,
     'jealousy': 3583,
     'Painful': 1034,
     'amateurish': 1655,
     'things': 5369,
     'Monica': 924,
     'hockey': 3368,
     'tummy': 5510,
     'PIECE': 1025,
     'perfectly': 4265,
     'thereplacement': 5363,
     'magnetic': 3794,
     'vodka': 5691,
     'riveted': 4713,
     'murder': 3986,
     'Honestly': 685,
     'lightweight': 3713,
     'Kindle': 791,
     'Cute': 378,
     'Far': 528,
     'ripped': 4708,
     'ambiance': 1661,
     'Despite': 409,
     'PURCHASE': 1032,
     'dog': 2646,
     'stale': 5115,
     'golden': 3202,
     'Worst': 1529,
     'WAY': 1460,
     'upgrading': 5611,
     'cake': 2026,
     'intentions': 3544,
     'secure': 4836,
     'indictment': 3493,
     'batch': 1825,
     'Edinburgh': 472,
     'sheer': 4907,
     'suffering': 5227,
     'LOVED': 804,
     'crafted': 2394,
     '5lb': 53,
     'DROPPED': 389,
     'The': 1347,
     'carriers': 2063,
     'Air': 107,
     'fact': 2928,
     'VX': 1435,
     'nay': 4014,
     'foreign': 3079,
     'reception': 4575,
     'Premium': 1087,
     'hold': 3369,
     'dine': 2582,
     'drinking': 2698,
     'otherwise': 4146,
     'Only': 1003,
     'contain': 2320,
     'Salads': 1187,
     'mode': 3930,
     'ratings': 4540,
     'forced': 3076,
     'Sushi': 1311,
     'Bluetooth': 238,
     'all': 1637,
     'BMW': 180,
     'Counterfeit': 362,
     'standout': 5118,
     'paired': 4206,
     'Bellagio': 216,
     'carries': 2064,
     'soggy': 5037,
     'vey': 5664,
     'interesting': 3548,
     'pepper': 4261,
     'trysts': 5508,
     'Hackneyed': 639,
     'background': 1795,
     'Damn': 392,
     'vehicle': 5647,
     'memorized': 3877,
     'magnificent': 3795,
     'exclaim': 2878,
     'Disney': 421,
     'FS': 522,
     'pearls': 4250,
     'psychological': 4476,
     'science': 4802,
     'wall': 5712,
     'directing': 2589,
     'Purchase': 1104,
     'mary': 3830,
     'PCS': 1018,
     'company': 2255,
     'steaks': 5138,
     ...}




```python
data = vectorizer.transform(sentences).toarray()
```


```python
data.shape
```




    (3000, 5892)




```python
target = np.array(targets)
```


```python
target.shape
```




    (3000,)




```python
num_features = data.shape[1]
```


```python
# settings

seed = 7
np.random.seed(seed)

# training
activation_function = 'relu'
optimizer = optimizers.RMSprop(lr=0.001)
loss = losses.binary_crossentropy
metrics = [metrics.binary_accuracy]
batch_size = 50
num_epoch = 5

# validation (k-fold cross validation)
K = 10 
```


```python
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
```


```python
# Keras Network
def network(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(32, activation = activation_function, input_dim=num_features)) 
    model.add(layers.Dense(32, activation = activation_function)) 
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
```


```python
model = network(activation_function, num_features)
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 32)                188576    
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 189,665
    Trainable params: 189,665
    Non-trainable params: 0
    _________________________________________________________________
    


```python
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (data[test], target[test])
    history = model.fit(x = data[train], y = target[train], validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], target[test], verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 79.66666674613953%
    Fold-1 Test Accuracy: 80.3333334128062%
    Fold-2 Test Accuracy: 84.66666666666667%
    Fold-3 Test Accuracy: 83.99999992052713%
    Fold-4 Test Accuracy: 81.00000007947285%
    Fold-5 Test Accuracy: 83.33333333333334%
    Fold-6 Test Accuracy: 80.66666666666666%
    Fold-7 Test Accuracy: 80.3333334128062%
    Fold-8 Test Accuracy: 82.99999992052715%
    Fold-9 Test Accuracy: 80.66666674613953%
    Average Accuracy: 81.76666669050852  with standard deviation: 1.699999951848798
    

#### Training/Validation Accuracy and  Training/Validation Losses (Averaged values of K folds)


```python
plot_classification(histories)
```


![png](output_25_0.png)



![png](output_25_1.png)


I tried 80, 50, 20, 10 epochs. However, to avoid overfitting, I set number of epochs to 5 which give me a better performance among others. I also tried with more hidden units such as 64-64 or 64-32, but to reduce overfitting effect, I decided on 2 layered, 32-32 network. 

### Try a different activation function and report the difference


```python
# change activation function
activation_function = 'tanh'
```


```python
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
     # fit
    validation_data = (data[test], target[test])
    history = model.fit(x = data[train], y = target[train], validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], target[test], verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 82.0%
    Fold-1 Test Accuracy: 78.3333334128062%
    Fold-2 Test Accuracy: 84.00000007947285%
    Fold-3 Test Accuracy: 83.6666665871938%
    Fold-4 Test Accuracy: 81.0%
    Fold-5 Test Accuracy: 81.33333333333333%
    Fold-6 Test Accuracy: 80.33333325386047%
    Fold-7 Test Accuracy: 80.33333333333333%
    Fold-8 Test Accuracy: 81.33333333333333%
    Fold-9 Test Accuracy: 80.33333333333333%
    Average Accuracy: 81.26666666666668  with standard deviation: 1.5832456032390267
    

#### Training/Validation Accuracy and Training/Validation Losses (Averaged values of K folds)


```python
plot_classification(histories)
```


![png](output_31_0.png)



![png](output_31_1.png)


**COMMENT:** Using Tanh as an activation function, **decreased** the average accuracy by approximately **0.5%** in comparison to ReLU. ReLU is very popular activation function and in this experiment, proved its quality. I also try Leaky ReLU which is modified version of ReLU. However, average accuracy scores are approximately same. The accuracy values mentioned here, averaged over accuracy values obtained from each fold in K-fold cross-validation.  <br> <br>
Also when we look at the average validation loss graph, loss icreases significantly after epoch 2. It might be a signal of overfitting and meaning that ReLU is better for generalizing the network and avoiding overfitting.

### Try a different optimizer and report the difference


```python
# make activation function ReLU again
activation_function = 'relu'
# change the optimizer
optimizer = optimizers.Adam(lr=0.0001)
```


```python
model = network(activation_function, num_features)
```


```python
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
     # fit
    validation_data = (data[test], target[test])
    history = model.fit(x = data[train], y = target[train], validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], target[test], verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 69.99999992052715%
    Fold-1 Test Accuracy: 74.33333333333333%
    Fold-2 Test Accuracy: 81.33333333333333%
    Fold-3 Test Accuracy: 79.0%
    Fold-4 Test Accuracy: 76.3333334128062%
    Fold-5 Test Accuracy: 82.33333333333334%
    Fold-6 Test Accuracy: 79.33333333333333%
    Fold-7 Test Accuracy: 79.6666665871938%
    Fold-8 Test Accuracy: 78.66666658719382%
    Fold-9 Test Accuracy: 77.66666662693024%
    Average Accuracy: 77.86666664679845  with standard deviation: 3.406529686553992
    

#### Training/Validation Accuracy and Training/Validation Losses (Averaged values of K folds)


```python
plot_classification(histories)
```


![png](output_38_0.png)



![png](output_38_1.png)


**COMMENT:** The averaged **accuracy decreased** when we used **Adam** optimizer. However, it totaly depends on the learning rate used. I used 0.0001 which is lower than the one we used in RMSProb (0.001). Hence, it causes slow but more stable learning. The average accuracy decreased but if I increase the number of epochs it will definetly continue to learn. Also, **overfitting effect that observed on previous experiments clearly avoided**.

### What if I use Adam Optimizer with same learning rate with RMSProb?


```python
optimizer = optimizers.Adam(lr=0.001)
```


```python
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
     # fit
    validation_data = (data[test], target[test])
    history = model.fit(x = data[train], y = target[train], validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], target[test], verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 80.6666665871938%
    Fold-1 Test Accuracy: 80.00000007947285%
    Fold-2 Test Accuracy: 84.66666666666667%
    Fold-3 Test Accuracy: 83.6666665871938%
    Fold-4 Test Accuracy: 81.00000007947285%
    Fold-5 Test Accuracy: 83.66666666666667%
    Fold-6 Test Accuracy: 78.99999992052715%
    Fold-7 Test Accuracy: 81.3333332935969%
    Fold-8 Test Accuracy: 81.3333334128062%
    Fold-9 Test Accuracy: 81.33333325386047%
    Average Accuracy: 81.66666665474574  with standard deviation: 1.693123344213111
    


```python
plot_classification(histories)
```


![png](output_43_0.png)



![png](output_43_1.png)


**COMMENT:** The average accuracy increased and nearly same with RMSProb (81.6 and 81.7). However, it definetly **converged very fast** in comparison to RMSProb. On the other hand, accuracy and loss graphs show us signals of overfitting in this case.

 

  **NOTE:** Sometimes, the validation losses are incerasing with number of epochs. In general, we expect from a loss to decrease by number of epochs. In our case, I believe the increase in loss is caused by overfitting. The data has only 3K instances and it's a binary classification. So, maybe 1-2 epoch is enough for the model to learn the data.

### Let's try only 2 epochs


```python
num_epoch = 2
```


```python
# train
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
     # fit
    validation_data = (data[test], target[test])
    history = model.fit(x = data[train], y = target[train], validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], target[test], verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 80.33333325386047%
    Fold-1 Test Accuracy: 77.00000007947287%
    Fold-2 Test Accuracy: 82.0%
    Fold-3 Test Accuracy: 79.99999992052715%
    Fold-4 Test Accuracy: 83.66666666666667%
    Fold-5 Test Accuracy: 82.0%
    Fold-6 Test Accuracy: 80.33333333333333%
    Fold-7 Test Accuracy: 81.66666666666667%
    Fold-8 Test Accuracy: 80.6666665871938%
    Fold-9 Test Accuracy: 82.99999992052715%
    Average Accuracy: 81.06666664282481  with standard deviation: 1.7751369104133174
    


```python
plot_classification(histories)
```


![png](output_50_0.png)



![png](output_50_1.png)


**COMMENT:** The validation loss and accuracy did not change too much. So, this also shows that for this specific problem, using less number of epochs does not damage our scores seriously.

 

 

 

 

 

 

 

 

## QUESTION-2

[Multiclass Classification] [30 pts] Use Keras’ built-in
Reuters dataset (from keras.datasets import reuters) to classify 46
different topics. Use k-fold cross validation and show loss/accuracy
plots by epoch.


```python
from keras.datasets import reuters
```


```python
num_words = 10000
```


```python
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)
```


```python
x_train.shape
```




    (8982,)




```python
y_train.shape
```




    (8982,)




```python
x_test.shape
```




    (2246,)




```python
y_test.shape
```




    (2246,)



### Preprocessing

##### Vectorizing the inputs


```python
def vectorize(seq):
    vectorized = np.zeros((len(seq),num_words))
    for i,seq in enumerate(seq):
        vectorized[i,seq] = 1
    return vectorized
```


```python
x_train = vectorize(x_train)
x_test  = vectorize(x_test)
```


```python
x_train.shape
```




    (8982, 10000)




```python
y_train.shape
```




    (8982,)




```python
x_test.shape
```




    (2246, 10000)




```python
y_test.shape
```




    (2246,)



##### one-hot encoding for targets (categorical values)

**46** different *category* as target


```python
# I will apply this operation after kfold validation otherwise Kfold does not work.
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
```


```python
# y_train.shape
```


```python
# y_test.shape
```


```python
# append train and test data because we will be using KFold Cross Validation 
# which automatically separate as train and test sets
# concat data and target then apply kfold
data = np.vstack((x_train, x_test))
target = np.hstack((y_train, y_test))
```


```python
data.shape
```




    (11228, 10000)




```python
target.shape
```




    (11228,)




```python
# settings
seed = 7
np.random.seed(seed)

# training settings
activation_function = 'relu'
optimizer = optimizers.RMSprop(lr=0.001)
loss = losses.categorical_crossentropy
metrics = ['accuracy']
batch_size = 512
num_epoch = 10

# validation settings (k-fold cross validation)
K = 10 
```


```python
num_features = data.shape[1]
```


```python
# network
def multiclass_network(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(64, activation=activation_function, input_shape=(num_features,)))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(46, activation='softmax'))
    return model
```


```python
# train
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    train_target = to_categorical(target[train])
    test_target  = to_categorical(target[test])
    model = multiclass_network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
     # fit
    validation_data = (data[test], test_target)
    history = model.fit(x = data[train], y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], test_target, verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 81.65938862546562%
    Fold-1 Test Accuracy: 80.82311737975807%
    Fold-2 Test Accuracy: 79.3286218870655%
    Fold-3 Test Accuracy: 78.59680285250526%
    Fold-4 Test Accuracy: 81.76156583629893%
    Fold-5 Test Accuracy: 81.85053380782918%
    Fold-6 Test Accuracy: 79.69588547785081%
    Fold-7 Test Accuracy: 82.70270268122356%
    Fold-8 Test Accuracy: 82.20415534796538%
    Fold-9 Test Accuracy: 81.0909091125835%
    Average Accuracy: 80.97136830085458  with standard deviation: 1.2801679718884411
    


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_565 (Dense)            (None, 64)                640064    
    _________________________________________________________________
    dense_566 (Dense)            (None, 64)                4160      
    _________________________________________________________________
    dense_567 (Dense)            (None, 46)                2990      
    =================================================================
    Total params: 647,214
    Trainable params: 647,214
    Non-trainable params: 0
    _________________________________________________________________
    

#### Training/Validation Accuracy and Training/Validation Losses (Averaged values of K folds)


```python
plot_classification(histories, c_type='multiclass')
```


![png](output_91_0.png)



![png](output_91_1.png)


I used 10 as number of epochs because when I tried 20, 50, 80 epochs, overfitting was a problem in these settings. Also, using more layers and hiddent units caused too much overfitting effect.

### Change number of layers, report the difference


```python
# network  (a deeper one)
def multiclass_network(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(64, activation=activation_function, input_shape=(num_features,)))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(64, activation=activation_function))
    model.add(layers.Dense(46, activation='softmax'))
    return model
```


```python
# train
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    train_target = to_categorical(target[train])
    test_target  = to_categorical(target[test])
    model = multiclass_network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (data[test], test_target)
    history = model.fit(x = data[train], y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], test_target, verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 74.67248905173555%
    Fold-1 Test Accuracy: 74.95621717331079%
    Fold-2 Test Accuracy: 79.5053003744186%
    Fold-3 Test Accuracy: 77.26465365179479%
    Fold-4 Test Accuracy: 78.11387900355872%
    Fold-5 Test Accuracy: 79.1814946619217%
    Fold-6 Test Accuracy: 75.49194987856637%
    Fold-7 Test Accuracy: 80.81081075711293%
    Fold-8 Test Accuracy: 74.70641372541962%
    Fold-9 Test Accuracy: 79.63636363636364%
    Average Accuracy: 77.43395719142028  with standard deviation: 2.215058214822054
    


```python
plot_classification(histories, c_type='multiclass')
```


![png](output_96_0.png)



![png](output_96_1.png)


**COMMENT:** I added 3 more hidden layers and each of these layers has 64 neurons. In general, what we expect after adding more layers is an increase in validation accuracy. However, adding more layers may cause another problem called overfitting. I observed that the averaged accuracy is decreased (approximately by 4% percent) when I add more layers. I believe the problem is overfitting because adding more layer caaused fitting the training dataset so much that, the network's performance on validation set suffers. Moreover, training time has elongated in comparsion to the smaller network. Moreover, average validation loss started to increase after epoch 5 and validation accuracy could not go beyond 0.8. 

 

### Increase number of hidden units, report the difference.


```python
# convert the model into its previous from and increase number of hidden units
def multiclass_network(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(128, activation=activation_function, input_shape=(num_features,)))
    model.add(layers.Dense(256, activation=activation_function))
    model.add(layers.Dense(46, activation='softmax'))
    return model
```


```python
# train
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    train_target = to_categorical(target[train])
    test_target  = to_categorical(target[test])
    model = multiclass_network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (data[test], test_target)
    history = model.fit(x = data[train], y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], test_target, verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 81.57205241736366%
    Fold-1 Test Accuracy: 79.68476353092494%
    Fold-2 Test Accuracy: 81.71378093978963%
    Fold-3 Test Accuracy: 78.15275311893511%
    Fold-4 Test Accuracy: 81.13879003558719%
    Fold-5 Test Accuracy: 80.33807829181495%
    Fold-6 Test Accuracy: 80.41144897344927%
    Fold-7 Test Accuracy: 81.71171169023256%
    Fold-8 Test Accuracy: 81.48148152994055%
    Fold-9 Test Accuracy: 80.63636363636364%
    Average Accuracy: 80.68412241644015  with standard deviation: 1.065363187970485
    


```python
plot_classification(histories, c_type='multiclass')
```


![png](output_102_0.png)



![png](output_102_1.png)


**COMMENT** : The averaged accuracy slightly decreased when we add more neurons (hidden units) to the network. I added 64 to first layer and 192 neurons to second layer. In total, I added 256 more neurons to the network. Again, after some points the model no longer learns even if you add more layers, more neurons to the network.  Training time has elongated too much in comparsion to the network with less hidden units. The overfitting effect is clear in both graphs but obvious in average train/validation loss graph.

  

### Decrease number of hidden units, report the difference.


```python
# deccrease number of hidden units
def multiclass_network(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(16, activation=activation_function, input_shape=(num_features,)))
    model.add(layers.Dense(16, activation=activation_function))
    model.add(layers.Dense(46, activation='softmax'))
    return model
```


```python
# train
histories = []
score_values = []
fold_counter = 0
for train, test in kfold.split(data, target):
    train_target = to_categorical(target[train])
    test_target  = to_categorical(target[test])
    model = multiclass_network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
   # fit
    validation_data = (data[test], test_target)
    history = model.fit(x = data[train], y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    scores = model.evaluate(data[test], test_target, verbose=0)
    print('Fold-{0} Test Accuracy: {1}%'.format(fold_counter, scores[1] * 100))
    score_values.append(scores[1] * 100)
    histories.append(history)
    fold_counter += 1  
average_score = np.mean(score_values)
std = np.std(score_values)
print('Average Accuracy: {0}  with standard deviation: {1}'.format(average_score, std))
```

    Fold-0 Test Accuracy: 73.9737991318432%
    Fold-1 Test Accuracy: 73.6427320281595%
    Fold-2 Test Accuracy: 75.53003531462733%
    Fold-3 Test Accuracy: 72.91296626280722%
    Fold-4 Test Accuracy: 75.97864768683274%
    Fold-5 Test Accuracy: 75.26690391459074%
    Fold-6 Test Accuracy: 75.9391770699796%
    Fold-7 Test Accuracy: 76.93693688323906%
    Fold-8 Test Accuracy: 76.87443543255814%
    Fold-9 Test Accuracy: 75.27272729440169%
    Average Accuracy: 75.23283610190393  with standard deviation: 1.274379878276489
    


```python
plot_classification(histories, c_type='multiclass')
```


![png](output_108_0.png)



![png](output_108_1.png)


**COMMENT:** The accuracy is worse than the other networks. Clearly, this small network is not capable of completely learning the representation of the data. Average loss and validation curves are very near to eachother, so there is no overfitting effect here. Instead, we can call this effect as **underfitting**.

 

 

 

 

 

 

 

 

 

## Question-3

[Regression] [30 pts] Use Keras’ built-in Boston House
Pricing dataset (from keras.datasets import boston_housing) to
perform regression and predict house prices. Use k-fold cross
validation and show loss/MAE plots by epoch.


```python
from keras.datasets import boston_housing
```


```python
boston = boston_housing.load_data()
```


```python
(x_train, y_train), (x_test, y_test) = boston
```


```python
# settings

seed = 7
np.random.seed(seed)

# training
activation_function = 'relu'
optimizer = optimizers.Adam(lr=0.001)
loss = losses.MSE
metrics = ['mae']
batch_size = 10
num_epoch = 100

# validation (k-fold cross validation)
K = 10 
```


```python
num_features = x_train.shape[1]
```


```python
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
```


```python
# Keras Network
def network(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(64, activation = activation_function, input_dim=num_features))
    model.add(layers.Dense(64, activation = activation_function))
    model.add(layers.Dense(64, activation = activation_function))
    model.add(layers.Dense(64, activation = activation_function))
    model.add(layers.Dense(1))
    return model
```


```python
model = network(activation_function, num_features)
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1303 (Dense)           (None, 64)                896       
    _________________________________________________________________
    dense_1304 (Dense)           (None, 64)                4160      
    _________________________________________________________________
    dense_1305 (Dense)           (None, 64)                4160      
    _________________________________________________________________
    dense_1306 (Dense)           (None, 64)                4160      
    _________________________________________________________________
    dense_1307 (Dense)           (None, 1)                 65        
    =================================================================
    Total params: 13,441
    Trainable params: 13,441
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# concat data and target then apply kfold
data = np.vstack((x_train, x_test))
```


```python
data.shape
```




    (506, 13)




```python
target = np.hstack((y_train, y_test))
```


```python
target.shape
```




    (506,)




```python
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed, )
```

 

**NOTE:** StratifiedKFold is not usable in a continuous target. So, in this problem, I needed to implement it myself.

 


```python
# to be able to split into K, I needed to omit last 6
data_folds = np.split(data[:500], K)
target_folds = np.split(target[:500], K)
# add last 6 to the last fold
last_data_fold = np.vstack((data_folds[-1],data[500:]))
last_target_fold = np.hstack((target_folds[-1],target[500:]))
data_folds[-1] = last_data_fold
target_folds[-1] = last_target_fold
```


```python
def split_data(fold, data, target, K):
    test_data = data_folds[fold]
    test_target = target_folds[fold]
    if fold == 0:
        train_data_folds = data_folds[fold + 1 :]
        train_target_folds = target_folds[fold + 1 :]
    elif fold > 0 and fold < (K - 1):
        train_data_folds = data_folds[:fold]
        for data_fold in data_folds[fold + 1:]:
            train_data_folds.append(data_fold)
        train_target_folds = target_folds[:fold]
        for target_fold in target_folds[fold + 1:]:
            train_target_folds.append(target_fold)
    elif fold == (K - 1):
        train_data_folds = data_folds[:(K - 1)]
        train_target_folds = target_folds[:(K - 1)]
    else:
        print('Error!')
    train_data = np.zeros((len(data) - len(test_data), data.shape[1]))
    train_target = np.zeros((len(target) - len(test_target), ))
    index_counter = 0
    for i in range(len(train_data_folds)):
        train_fold = train_data_folds[i]
        num_instances = len(train_fold)
        train_data[index_counter : index_counter + num_instances] = train_fold
        index_counter += num_instances
    index_counter = 0
    for i in range(len(train_target_folds)):
        target_fold = train_target_folds[i]
        num_instances = len(target_fold)
        train_target[index_counter : index_counter + num_instances] = target_fold
        index_counter += num_instances
    return train_data, train_target, test_data, test_target
```


```python
# example usage
# train_data, train_target, test_data, test_target = split_data(fold = 9, data = data, target = target, K = 10)
```


```python
# train
histories = []
test_mse_scores = []
test_mae_scores = []
for i in range(K):    
    train_data, train_target, test_data, test_target = split_data(fold = i, data = data, target = target, K = K)
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (test_data, test_target)
    history = model.fit(x = train_data, y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    mse, mae = model.evaluate(test_data, test_target, verbose=0)
    print('Fold-{0} MSE: {1}  MAE: {2}'.format(i, mse, mae))
    test_mse_scores.append(mse)
    test_mae_scores.append(mae)
    histories.append(history)
```

    Fold-0 MSE: 17.202894287109373  MAE: 3.1536805534362795
    Fold-1 MSE: 7.95403018951416  MAE: 2.388444061279297
    Fold-2 MSE: 19.41813949584961  MAE: 3.0037548542022705
    Fold-3 MSE: 19.965714797973632  MAE: 3.4257075786590576
    Fold-4 MSE: 24.321148529052735  MAE: 3.6299257469177246
    Fold-5 MSE: 29.02158996582031  MAE: 3.0002075099945067
    Fold-6 MSE: 23.67058433532715  MAE: 3.4230592250823975
    Fold-7 MSE: 20.084044647216796  MAE: 3.3294317722320557
    Fold-8 MSE: 21.55253402709961  MAE: 3.3475891017913817
    Fold-9 MSE: 45.983392987932476  MAE: 4.468930176326206
    


```python
histories[0].history.keys()
```




    dict_keys(['loss', 'val_mean_absolute_error', 'val_loss', 'mean_absolute_error'])




```python
plot_regression(histories, 'mean_absolute_error')
```


![png](output_143_0.png)



![png](output_143_1.png)


#### Average Test MSE Score


```python
np.mean(test_mse_scores)
```




    22.917407326289585



#### Average Test MAE Score


```python
np.mean(test_mae_scores)
```




    3.3170730579921175



 So, after 100 epoch, it starts to overfit so, I stop the learning there.

### Compare the results when using no regularizer, L2 regularizer and Dropout as a regularization method.

##### L2 Regularizer


```python
regularizer = regularizers.l2(0.01)
```


```python
# Keras Network
def network_with_regularizer(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(64, kernel_regularizer = regularizer, activation = activation_function, input_dim=num_features))
    model.add(layers.Dense(64, kernel_regularizer = regularizer, activation = activation_function))
    model.add(layers.Dense(64, kernel_regularizer = regularizer, activation = activation_function))
    model.add(layers.Dense(64, kernel_regularizer = regularizer, activation = activation_function))
    model.add(layers.Dense(1))
    return model
```


```python
model = network_with_regularizer(activation_function, num_features)
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1358 (Dense)           (None, 64)                896       
    _________________________________________________________________
    dense_1359 (Dense)           (None, 64)                4160      
    _________________________________________________________________
    dense_1360 (Dense)           (None, 64)                4160      
    _________________________________________________________________
    dense_1361 (Dense)           (None, 64)                4160      
    _________________________________________________________________
    dense_1362 (Dense)           (None, 1)                 65        
    =================================================================
    Total params: 13,441
    Trainable params: 13,441
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# train
histories = []
test_mse_scores = []
test_mae_scores = []
for i in range(K):    
    train_data, train_target, test_data, test_target = split_data(fold = i, data = data, target = target, K = K)
    model = network_with_regularizer(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (test_data, test_target)
    history = model.fit(x = train_data, y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    mse, mae = model.evaluate(test_data, test_target, verbose=0)
    print('Fold-{0} MSE: {1}  MAE: {2}'.format(i, mse, mae))
    test_mse_scores.append(mse)
    test_mae_scores.append(mae)
    histories.append(history)
```

    Fold-0 MSE: 20.244517822265625  MAE: 3.163468894958496
    Fold-1 MSE: 11.836295471191406  MAE: 2.4984943962097166
    Fold-2 MSE: 14.458795356750489  MAE: 2.92088171005249
    Fold-3 MSE: 13.747651138305663  MAE: 2.566042308807373
    Fold-4 MSE: 17.316460800170898  MAE: 2.94060923576355
    Fold-5 MSE: 21.671018371582033  MAE: 2.783031520843506
    Fold-6 MSE: 23.984453392028808  MAE: 3.588962516784668
    Fold-7 MSE: 16.449061012268068  MAE: 3.0735275173187255
    Fold-8 MSE: 20.75134750366211  MAE: 3.294657516479492
    Fold-9 MSE: 39.876177651541575  MAE: 4.641913414001465
    


```python
plot_regression(histories, 'mean_absolute_error')
```


![png](output_156_0.png)



![png](output_156_1.png)


#### Average Test MSE Score


```python
np.mean(test_mse_scores)
```




    20.033577851976666



#### Average Test MAE Score


```python
np.mean(test_mae_scores)
```




    3.1471589031219485



**COMMENT:**

The averaged MAE error over folds decreased. (**previous : 3.31 and now: 3.14**). Also MSE error decreased. (22.91 - 20.03) The network, apart from the regulazier, is the same. This decrese is a good signal that using L2 regularazier **somehow reduced the effect of overfitting** even if the overfitting effect is not much in this example. Moreover, average loss and MAE curves of training and validation sets are closer to eachother which is an effect of reducing overfitting.

  

##### Dropout


```python
# Keras Network
def network_with_dropout(activation_function, num_features):
    model = Sequential()
    model.add(layers.Dense(64, activation = activation_function, input_dim=num_features))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation = activation_function))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation = activation_function))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation = activation_function))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    return model
```


```python
# train
histories = []
test_mse_scores = []
test_mae_scores = []
for i in range(K):    
    train_data, train_target, test_data, test_target = split_data(fold = i, data = data, target = target, K = K)
    model = network_with_dropout(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (test_data, test_target)
    history = model.fit(x = train_data, y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    mse, mae = model.evaluate(test_data, test_target, verbose=0)
    print('Fold-{0} MSE: {1}  MAE: {2}'.format(i, mse, mae))
    test_mse_scores.append(mse)
    test_mae_scores.append(mae)
    histories.append(history)
```

    Fold-0 MSE: 99.31305709838867  MAE: 7.856642227172852
    Fold-1 MSE: 110.65490356445312  MAE: 8.464013214111327
    Fold-2 MSE: 124.81808044433593  MAE: 9.038162269592284
    Fold-3 MSE: 98.32915496826172  MAE: 8.609693984985352
    Fold-4 MSE: 75.88119293212891  MAE: 7.378611106872558
    Fold-5 MSE: 73.12716033935547  MAE: 6.604808921813965
    Fold-6 MSE: 248.90896850585938  MAE: 13.282108192443848
    Fold-7 MSE: 123.04225799560547  MAE: 9.683528518676757
    Fold-8 MSE: 103.63031616210938  MAE: 8.363066177368164
    Fold-9 MSE: 163.6944318498884  MAE: 10.96738760811942
    


```python
plot_regression(histories, 'mean_absolute_error')
```


![png](output_167_0.png)



![png](output_167_1.png)


#### Average Test MSE Score


```python
np.mean(test_mse_scores)
```




    122.13995238603863



#### Average Test MAE Score


```python
np.mean(test_mae_scores)
```




    9.024802222115655



**COMMENT:**  The averaged MAE and MSE is much higher than the original network and L2 network. I think, it is because the network is not complex so that dropout affected learning seriously. In general, dropout is effective when we have a complex network that is being overfitted. Since, I needed to preserve the network architecture to compare effects of different experiments, the effect of dropout seems negative in this example. **Fluctuation/oscillation effect** in validation graphs, possibly because of dropout since we drop 30% of neurons randomly. Note that, I aslo try dropout with 0.1, 0.5, 0.7. What I observed is that, when dropout rate increases, the loss also increases. 

 

### Try a different loss function


```python
loss = losses.MSLE
```


```python
# train
histories = []
test_msle_scores = []
test_mae_scores = []
for i in range(K):    
    train_data, train_target, test_data, test_target = split_data(fold = i, data = data, target = target, K = K)
    model = network(activation_function, num_features)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit
    validation_data = (test_data, test_target)
    history = model.fit(x = train_data, y = train_target, validation_data = validation_data ,epochs = num_epoch, batch_size = batch_size, verbose = 0)
    msle, mae = model.evaluate(test_data, test_target, verbose=0)
    print('Fold-{0} MSLE: {1}  MAE: {2}'.format(i, msle, mae))
    test_msle_scores.append(msle)
    test_mae_scores.append(mae)
    histories.append(history)
```

    Fold-0 MSLE: 0.030994911305606367  MAE: 2.730000071525574
    Fold-1 MSLE: 9.456000862121583  MAE: 279.97599853515624
    Fold-2 MSLE: 9.752659759521485  MAE: 32.211522064208985
    Fold-3 MSLE: 9.526685562133789  MAE: 88.03454711914063
    Fold-4 MSLE: 0.03070126101374626  MAE: 3.14084228515625
    Fold-5 MSLE: 0.042200155556201935  MAE: 3.288249053955078
    Fold-6 MSLE: 0.03201048843562603  MAE: 3.7605451774597167
    Fold-7 MSLE: 9.716106491088867  MAE: 77.40607421875
    Fold-8 MSLE: 9.701545867919922  MAE: 349.4834716796875
    Fold-9 MSLE: 0.07658939542514938  MAE: 4.084607260567801
    


```python
histories[0].history.keys()
```




    dict_keys(['loss', 'val_mean_absolute_error', 'val_loss', 'mean_absolute_error'])




```python
plot_regression(histories, 'mean_absolute_error')
```


![png](output_178_0.png)



![png](output_178_1.png)


#### Average MSLE Score


```python
np.mean(test_msle_scores)
```




    4.836549475452198



#### Average MAE Score


```python
np.mean(test_mae_scores)
```




    84.41158574656079



**COMMENT:**

MSLE showed worse performance than MSE if we compare averaged MAE score. When we directly look at the values of loss functions, MSLE (4.83) seems better than MSE (22.9).  **MSLE is usually used when you do not want to penalize huge differences in the predicted and the actual values when both predicted and true values are huge numbers. So, the averaged loss is low in comparison to the MSE.** However, **the performance of MSE** is **much better** and we can understand this by looking averaged MAE error score which is much lower in MSE. 

 

 

 

 

 

 

 
