Kanji-Recognizer 

Description: 

We have made a Kanji recognition program. Kanji are Japanese characters with chinese origin and because there are 50k characters, it is impossible for the average person to learn all characters. 

We have therefore made a machine learning model using Keras-CNN. The dataset originally consists of 140,000 entries, but we reduced it significantly by cleaning the data and removing unwanted entries. By using the program, a user can draw a kanji and interpret the meaning, as well as the accuracy compared to the actual kanji. It keeps a log over attempts so the user can see a history, showing changes in accuracy over time.  


Technologies: 

-Keras CNN 

-Tkinter 

-Selenium 

-BeautifulSoup 

-OpenCV 


 
Installation Guide: 

These are the libraries you may need to install to run the program. 

-beautifulsoup4 

-seaborn 

-tk 

-tensorflow 

-sklearn 

-OpenCV 

-Matplotlib 

-Selenium 

-numpy	 

 
User guide: 

To run the program, you find the UI.py file (or JupUI.ipynb, if the .py file is troublesome) and run it.  

If you want to interpret any images, place the image in the inputImage directory. 

Then follow the instructions presented in the console (if running .ipynb file, the instructions will present themselves in the output section, and the input field will present itself at the top of your file window).  

If an error is encountered when option 1 in the main menu is selected, then you should run the program outside of the docker environment.  

 

Status: 

We have made a classification model, which recognizes kanji symbols. We have used various methods of data cleaning and web scraping to reduce the size, and clean the dataset provided to the model. We ended up with a model that performs quite well on our test data. On our own drawings the model performs well on simple kanji symbols but struggles a bit on more complex symbols. This model is used in a user interface where you can test your ability to draw specific kanji symbols, your “score” is then saved, and all your previous attempts can be displayed in the user interface. The user interface also provides an option to get a translation of a specific kanji symbol. 

  

Project Highlights: 

Classification model: We have made a sequential model that is a decent model for easy testing.  

Datacleaning: The data has been cleaned by removing kanji symbols with too few images, and by removing kanji symbols based on the number of contours in the image compared to the average for that symbol.  

Journal Folder: We have tested different versions of our classification model, to find the most reliable version.  

Drawing feature: If you decide to test your skills, a window pops up where you can draw the symbol presented. The drawing then gets saved and run through our classification model.  

Skill History: When getting accuracies of your drawings, it gets saved to your history. The history contains all the kanjis you have drawn plus the accuracy of each of them. The accuracy can be displayed as a graph. The graph can either display all your attempts or all the attempts of a specific kanji. 


