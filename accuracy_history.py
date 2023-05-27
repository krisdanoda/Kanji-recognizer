import matplotlib.pyplot as plt
import csv
import helper_functions

def kanji(kaniUnicode):
    return chr(int(kaniUnicode[2:], 16))

accuracy_history = []
filename = 'accuracy_history.csv'

def save_accuracy(kanji, new_accuracy):
    global accuracy_history
    accuracy_history.append((kanji, new_accuracy))
   # print(accuracy_history)

def display_accuracy():
    global accuracy_history
    accuracies = [accuracy for kanji, accuracy in accuracy_history]



    plt.plot( accuracies, marker='o', color='#0229b8')
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Kanjis')
    plt.ylim(0, 103)
    plt.xlim(0)
    plt.show()

def display_specific_accuracy(kanjiSign):
    global accuracy_history
        
    indexes = [i for i, (k, _) in enumerate(accuracy_history) if k == kanjiSign]
    
    if len(indexes) == 0:
        print(f"No ratings found for {kanjiSign}.")
        return
    
    kanji_accuracies = [accuracy_history[index][1] for index in indexes]

    plt.plot(range(len(kanji_accuracies)), kanji_accuracies, marker='o', color='#0229b8')
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Occurrences')
    plt.ylim(0, 103)
    plt.xlim(0)
    plt.show()

def kanji_list():
    list_of_kanji = dict()
    
    for element in accuracy_history:
        list_of_kanji[helper_functions.to_kanji(element[0])] = element[0]
    return list_of_kanji

def save_history():
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(accuracy_history)
    file.close()
    print(f"Your history has been saved to {filename}.")

def load_history():
    global accuracy_history
    accuracy_history = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            accuracy_history.append((row[0], float(row[1])))
    file.close()

def clear_history():
    global accuracy_history
    accuracy_history = []
    file = open(filename, 'w')
    file.close()
    print(f"History in '{filename}' has been cleared.")