

## **Setup Instructions**

Follow these steps to get the Pollen Grain Classifier up and running on your local machine.

### **1\. Clone the Repository (or create files manually)**

First, create a project directory:  
mkdir pollen\_classifier\_flask  
cd pollen\_classifier\_flask

Then, manually create the app.py, train\_model.py, requirements.txt, and templates/index.html files within this directory, copying the content provided in the respective code blocks. Ensure you also create the dataset and templates subdirectories.  
Create requirements.txt:  
Create a file named requirements.txt in the pollen\_classifier\_flask directory with the following content:  
Flask  
Pillow  
tensorflow

### **2\. Install Dependencies**

You'll need Python 3 installed. Navigate to your project directory in the terminal and install the required Python packages using the requirements.txt file:  
pip install \-r requirements.txt

### **3\. Train and Save the CNN Model**

Before running the Flask application, you need to train and save your CNN model. This step runs the train\_model.py script. **Ensure your dataset folder is populated with your pollen grain images organized by class.**  
python train\_model.py

This script will define the CNN architecture, load your training data, train the model, and save it as pollen\_classifier\_model.h5.

### **4\. Run the Flask Application**

After the model has been trained and saved, you can start the Flask server:  
python app.py

The terminal will show output similar to: \* Running on http://127.0.0.1:5000/.

## **Usage**

1. **Open in Browser**: Open your web browser and navigate to http://127.0.0.1:5000/.  
2. **Upload Image**: Click the "Upload Pollen Image" button and select an image file (e.g., a .jpg or .png) of a pollen grain sample. An image preview will appear.  
3. **Identify Pollen Type**: Click the "Identify Pollen Type" button.  
4. **View Result**: The application will display the classified pollen type.

