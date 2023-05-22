import matplotlib.pyplot as plt
import csv

def kanji(kaniUnicode):
    return chr(int(kaniUnicode[2:], 16))

accuracy_history = []
filename = 'accuracy_history.csv'


def save_accuracy(kanji, new_accuracy):
    global accuracy_history
    accuracy_history.append((kanji, new_accuracy))
    print(accuracy_history)

def display_accuracy():
    global accuracy_history
    accuracies = [accuracy for kanji, accuracy in accuracy_history]

    plt.plot(accuracies, marker='o', color='#0229b8')
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Kanjis')
    plt.ylim(0, 105)
    plt.xlim(0)
    plt.show()

def display_specific_accuracy(kanjiSign):
    global accuracy_history
    
    data = [(kanji, accuracy) for kanji, accuracy in accuracy_history]
        
    indexes = [i for i, (k, _) in enumerate(data) if k == kanjiSign]
    
    if len(indexes) == 0:
        print(f"No ratings found for {kanjiSign}.")
        return
    
    kanji_accuracies = [accuracy for kanji, accuracy in data if kanji in indexes]
    
    print("Plot for:", kanjiSign)
    plt.plot(kanji_accuracies)
    plt.title('Skill Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Occurrences')
    plt.show()

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