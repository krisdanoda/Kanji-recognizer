import matplotlib.pyplot as plt

def kanji(kaniUnicode):
    return chr(int(kaniUnicode[2:], 16))

accuracy_history = tuple()

def save_and_display_accuracy(kanji, new_accuracy):

    global accuracy_history
    accuracy_history += (kanji, new_accuracy)
    #kanjis = accuracy_history[::2]  
    accuracies = accuracy_history[1::2]  

    print(accuracy_history)
    plt.plot(accuracies)
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Kanjis')
    plt.show()

def display_kanji_accuracy(kanjiS):
    global accuracy_history
    
    sign = kanji(kanjiS)

    kanjis = accuracy_history[::2]
    accuracies = accuracy_history[1::2]
    
    print(accuracy_history)
    
    indices = [i for i, k in enumerate(kanjis) if k == kanjiS]
    
    if len(indices) == 0:
        print(f"No ratings found for {kanji}.")
        return
    
    kanji_accuracies = [accuracies[i] for i in indices]
    print("Plot for ", sign)
    plt.plot(kanji_accuracies)
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Occurrences')
    plt.show()
