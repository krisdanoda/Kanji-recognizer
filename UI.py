import input_image
import accuracy_history
import helper_functions
from pathlib import Path
import ImageDrawing

def list_directory(path):
    imgs = []
    for index, img in enumerate(Path(path).iterdir(), start=1):
        print("{}. {}".format(index, img))
        imgs.append(img)
    return imgs


def pick_file():
    imgs = list_directory("input_images")
    while True:
        picked_kanji = input(
            "Choose which file you want to interperet. Confirm that you have placed your image correctly.")
        picked_kanji = int(picked_kanji)
        if picked_kanji > 0 and picked_kanji <= len(imgs) + 1:
            return picked_kanji-1


def translate_kanji_og():
    imgs = list_directory("input_images")
    print("Make sure you have placed your image in the input_images folder\n")

    while True:
        picked_kanji = input(
            "Choose which file you want to interpret. Confirm that you have placed your image correctly.")
        picked_kanji = int(picked_kanji)
        if picked_kanji > 0 and picked_kanji <= len(imgs) + 1:
            print("Your image is being recognized...\n")
            image_output_list = input_image.process_image(picked_kanji - 1)
            print(f"\nWe recognized your image as: {image_output_list[0]}\n")
            print("It translates to:")
            for item in image_output_list[1]:
                print(item)
            return


def translate_kanji():
    print("Make sure you have placed your image in the input_images folder\n")
    picked_kanji = pick_file()
    print("Your image is being recognized...\n")

    image_output_list = input_image.process_image(picked_kanji)

    print(f"\nWe recognized your image as: {image_output_list[0]}\n")
    print("It translates to:")
    for item in image_output_list[1]:
        print(item)
    return


def display_accuracy():
    print("Type which character you wish to see the history of")
    options = accuracy_history.kanji_list()
    print("----------------")
    for index, option in enumerate(options, start=1):
        print(f'{index} - {option}')
    print("----------------")
    sign_to_check = input('')
    try:
        sign_to_check = int(sign_to_check)
        selected_option = list(options.values())[sign_to_check - 1]
        print(f"You chose {list(options)[sign_to_check - 1]}")
        accuracy_history.display_specific_accuracy(selected_option)
    except Exception as e:
        print(e)
        print("Sign not found!")


def show_history_options():
    accuracy_history.kanji_list()


def skills_test():
    test_label, test_kanji = helper_functions.random_kanji()
    print("Can you draw this kanji: ", test_kanji)
    print("Click on this link to see a better rendition:", f'https://www.compart.com/en/unicode/{test_label}' "\n")
    print("It may take some time to open the canvas, please be patient\n")

    ImageDrawing.open_drawing_canvas(test_kanji)

    print("Your image is being recognized...\n")
    image_output_list = input_image.process_image(path="input_images/Canvas/", save_accuracy= True, kanji_unicode=test_label)

    print(f'You drew {image_output_list[0]} with an accuracy of {image_output_list[2]}%')


def skills_test_function():
    print("Would you like to pick a kanji character, leave it blank to use the default/n")
    kanji = input()
    if kanji == "":
        kanji = "U+91CE"

    while True:
        print("What would you like to do?:\n-Draw an image-> Type 1\n-Read Image -> Type 2\n")
        choice = input()
        choice = int(choice)
        if choice == 1:
            skills_test_read_drawing(kanji)
            break

        elif choice == 2:
            skills_test()

def skills_test_read_drawing(kanji):

    print("Can you draw this kanji: " +  kanji +"\n Please maximise the popup to draw you character")

    ImageDrawing.open_drawing_canvas(kanji)

    print("Your image is being recognized...\n")
    image_output_list = input_image.process_image(path="input_images/Canvas/", save_accuracy=True)

    print(f'You drew {image_output_list[0]} with an accuracy of {image_output_list[2]}%')


def clear_history():
    print("Are you sure you wish to clear your history?\n")
    print("If you are, type 'yes'")
    second_choice = input('')
    if (second_choice == "yes"):
        accuracy_history.clear_history()
        print("Your history has been cleared")
    else:
        print("Your history has NOT been cleared")
        second_choice = ''


def accuracy_plot():
    accuracy_history.display_accuracy()


no_shutdown = True
accuracy_history.load_history()
while no_shutdown:

    print("WELCOME TO KANJI RECOGNIZER :D´\n")
    print(
        "What would you like to do?:\n-Test your skills -> Type 1\n-Get translation of a kanji symbol -> Type 2\n-Show Full History -> Type 3\n-Show Specific History -> Type 4\n-Clear history -> Type 'clear'\n(Type C to shut down)\n")
    choice = input("Type your choice here!")

    if (choice == "1"):
        print("You have chosen to test your skills\n")
        skills_test()

    elif (choice == "2"):
        print("You have chosen to translate a kanji symbol\n")
        translate_kanji()

    elif (choice == "3"):
        print("Accuracy history: ")
        accuracy_plot()

    elif (choice == "4"):
        print("You have chosen to see your history of a specific Kanji\n")
        display_accuracy()

    elif (choice.lower() == "clear"):
        print("You have chosen to clear your history")
        clear_history()

    elif (choice in ["c", "C"]):

        accuracy_history.save_history()
        no_shutdown = False

    else:
        print("MAKE A CORRECT CHOICE")

print("SHUT DOWN")
