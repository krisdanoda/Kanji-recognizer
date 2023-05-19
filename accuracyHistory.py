import matplotlib.pyplot as plt

def kanji(kaniUnicode):
    return chr(int(kaniUnicode[2:], 16))

accuracy_history = tuple()


def save_accuracy(kanji, new_accuracy):

    global accuracy_history
    accuracy_history += (kanji, new_accuracy)
   
def display_accuracy():

    global accuracy_history
    accuracies = accuracy_history[1::2]  

    plt.plot(accuracies)
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Kanjis')
    plt.show()

def display_kanji_accuracy(kanjiSign):
    global accuracy_history
    
    sign = kanji(kanjiSign)

    kanjis = accuracy_history[::2]
    accuracies = accuracy_history[1::2]
        
    indexes = [i for i, k in enumerate(kanjis) if k == kanjiSign]
    
    if len(indexes) == 0:
        print(f"No ratings found for {kanji}.")
        return
    
    kanji_accuracies = [accuracies[i] for i in indexes]
    print("Plot for: ", sign)
    plt.plot(kanji_accuracies)
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Occurrences')
    plt.show()
