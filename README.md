# 🧬 Breast Cancer Detection with Histopathology Images

<p>
  This project focuses on binary image classification—detecting Invasive Ductal Carcinoma (IDC) in breast tissue—using the Kaggle 
  <a href = "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images">Breast Histopathology Images dataset </a>.The dataset contains 277,524   high-resolution 50×50 RGB image patches, extracted from 162 patient whole-slide specimens
</p>

<p>
  Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide
</p>


<p>
  Invasive means the cancer has spread into surrounding breast tissues. Ductal means the cancer started in the milk ducts, the tubes that carry milk from the lobules to the nipple. Carcinoma refers to any cancer that begins in the skin or other tissues that cover the lining of internal organs, such as breast tissue.
</p>


<img src = 'https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/Breast_Cancer.png'>

<p>
Normal breast with invasive ductal carcinoma (IDC) in an enlarged cross-section of the duct

- Breast profile: A Ducts B Lobules C Dilated section of duct to hold milk D Nipple E Fat F Pectoralis major muscle G Chest wall/rib cage

- Enlargement: A Normal duct cell B Ductal cancer cells breaking through the basement membrane C Basement membrane
  
</p>

<p>
  Inasive ductal carcinoma (IDC) is - with ~ 80 % of cases - one of the most common types of breast cancer. It's malicious and able to form metastases which makes it especially dangerous. Often a biopsy is done to remove small tissue samples. Then a pathologist has to decide whether a patient has IDC, another type of breast cancer or is healthy. In addition sick cells need to be located to find out how advanced the disease is and which grade should be assigned. This has to be done manually and is a time consuming process. Furthermore the decision depends on the expertise of the pathologist and his or her equipment. Therefor deep learning could be of great help to automatically detect and locate tumor tissue cells and to speed up the process. In order to exploit the full potential one could build a pipeline using massive amounts of tissue image data of various hospitals that were evaluated by different experts. This way one would be able to overcome the dependence on the pathologist which would be especially useful in regions where no experts are available .
</p>

## Requirements

- **Python Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` , `keras` , `tensorflow`
- **Kaggle API Key** (for data downloading)

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Install Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Kaggle API, download the data  and follow the steps to load
   
    - Run this command on any code editor

      
   ```
     !pip install opendatasets
   ```

   
    - Once you install library then run below command  for loading data into your editor for that you need to pass your kaggle credentials
      
   ```
       import opendatasets as od
       url = "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data"
       od.download(url)
   ```

  


<h1>
  Understand the Data
</h1>
  
<h4>
  📊 Dataset Overview
</h4>

- Total images: 277,524

- IDC-negative: 198,738

- IDC-positive: 78,786 

- Image size: 50×50 pixels, RGB

- Labels: Extracted from image filenames (class0 = IDC-negative, class1 = IDC-positive)

<h1>
  Preprocessing
</h1>

```
# Non_IDC
N_IDC = []
# Positive_IDC
P_IDC = []

for img in breast_img:
  if img[-5] == "0":
    N_IDC.append(img)

  elif img[-5] == "1":
    P_IDC.append(img)


plt.figure(figsize = (15 , 15))

some_non = np.random.randint(0 , len(N_IDC) , 18)
some_can = np.random.randint(0 , len(P_IDC) , 18)

s = 0

for num in some_non:

  img = image.load_img((N_IDC[num]) , target_size = (100 , 100))
  img = image.img_to_array(img)

  plt.subplot(6,6 , 2*s+1)
  plt.axis("off")
  plt.title('no cancer')
  plt.imshow(img.astype('uint8'))

  s+=1

s = 1
for num in some_can:

  img = image.load_img((P_IDC[num]) ,  target_size= (100 , 100))
  img = image.img_to_array(img)

  plt.subplot(6,6 , 2*s)
  plt.axis("off")
  plt.title('IDC (+)')
  plt.imshow(img.astype('uint8'))
  s+=1

```

  <img src = "https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img1.png">



<p>
-  In breast-histopathology-images folder contain 2 types of images one is no-cancer images and cancer images
  
- Sometimes we can find artifacts or incomplete patches that have smaller size than 50x50 pixels.

- Patches with cancer look more violet and crowded than healthy ones. Is this really typical for cancer or is it more typical for ductal cells and tissue?

- Though some of the healthy patches are very violet colored too! Would be very interesting to hear what criteria are important for a pathologist. I assume that the wholes in the tissue belong to the mammary ducts where the milk can flow through.
- Total IDC (+)  - 78786
- Total No_cancer iamge - 198738
</p>

<h4>
  Now converting images into array ans resizing the image
</h4>

```
import cv2
import numpy as np
import random

# Initialize data holders
non_img_arr = []
can_img_arr = []

# Read, resize and label the images
for img in NewN_IDC:
    n_img = cv2.imread(img, cv2.IMREAD_COLOR)
    n_img_resized = cv2.resize(n_img, (50, 50))
    non_img_arr.append((n_img_resized, 0))

for img in P_IDC:
    c_img = cv2.imread(img, cv2.IMREAD_COLOR)
    c_img_resized = cv2.resize(c_img, (50, 50))
    can_img_arr.append((c_img_resized, 1))




num_samples = min(len(non_img_arr), len(can_img_arr))
breast_img_arr = non_img_arr[:num_samples] + can_img_arr[:num_samples]


random.shuffle(breast_img_arr)

# Separate features and labels
X = np.array([item[0] for item in breast_img_arr], dtype=np.float32)
y = np.array([item[1] for item in breast_img_arr], dtype=np.uint8)

print("X shape:", X.shape)
print("y shape:", y.shape)

```

<p>
How many patches do we have in total?

Our algorithm needs to decide whether an image patch contains IDC or not. Consequently not the whole patient tissue slice but the single patches have to be considered as input to our algorithm. How many of them do we have in total?

Ok, roughly 280000 images. To feed the algorithm with image patches it would be nice to store the path of each image. This way we can load batches of images only one by one without storing the pixel values of all images.

</p>


<h1>
  Train and Test Split
</h1>

<p>
  Since we know about our datasets now next step is split data into train and test part. Here i am using 70 % and 30% rule where 70% will be the train data and 30% will be test data
</p>

```
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.3)

```

```
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train , num_classes = 2)
y_test = to_categorical(y_test , num_classes = 2)

print("Training Data Shape :" , x_train.shape)
print("Testing Data Shape :" , x_test.shape)

```

<img src = https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img2.png>

<p>
  Done with loading,  Done with preprocessing , Done with spliting the data. .Now next is step is to build Convolutional Neural Network Model
</p>

 <h1>
   Model
 </h1> 

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.optimizers import Adam , SGD
from keras.metrics import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import itertools

```


```
early_stop = EarlyStopping(monitor = "val_loss" , patience = 5)

model = Sequential()

# Input size - (50 x 50)
model.add(Conv2D(32 , (3,3) , activation = "relu" , padding  = "same" , input_shape = (50 , 50 , 3)))
# output size - (50 x 50)
model.add(BatchNormalization())

# input size - (50 x 50)
model.add(Conv2D(32 , (3,3) , activation = "relu" , padding = "same"))
# output size - (50 x 50)
model.add(MaxPooling2D(2,2))
# output size - (25 x 25)
model.add(BatchNormalization())
# output size - (25 x 25)
model.add(Dropout(0.3))

# input size - (25 x 25)
model.add(Conv2D(64 , (3,3) , activation = "relu" , padding = "same"))
# output size - (25 x 25)
model.add(BatchNormalization())

# input size - (25 x 25)
model.add(Conv2D(64 , (3,3) , activation = "relu", padding = "same"))
# output size - (25 x 25)
model.add(BatchNormalization())
# output size - (25 x 25)
model.add(MaxPooling2D(2,2))
# output size - (12 x 12)
model.add(Dropout(0.3))

# input size - (12 x 12)
model.add(Conv2D(128 , (3,3) , activation = "relu", padding = "same"))
# output size - (12 x 12)

model.add(Flatten())
model.add(Dense(128 , activation = "relu" ,))
model.add(BatchNormalization())

model.add(Dense(64 , activation = "relu"))
model.add(BatchNormalization())

model.add(Dense(64 , activation = "relu"))
model.add(Dropout(0.3))

model.add(Dense(24 , activation = "relu"))

model.add(Dense(2 , activation="softmax"))


```

<p>Yes. We mede CNN Model , Now next and crucial step is check the accuracy and loss of our model on train and test data</p>

```
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

```

<img src = 'https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img3.png'>

<img src = 'https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img4.png'>


- Accuracy of our model on train and test data is increasing . But Slightly facing an overfitting problem.

- Loss of train and test is decreasing with epochs , But at one pint train and test loss is increase , But if we increase more epochs value then it can possible loss will decrease

<p>
  Now Let's check how our confusion matrix is look like
</p>

```
from sklearn.metrics import accuracy_score, confusion_matrix
#Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1) 
Y_true = np.argmax(y_test, axis=1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 


confusion_mtx_percent = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis] * 100 

f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_mtx_percent, annot=True, linewidths=0.01, cmap="BuPu", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Percentage)")
plt.show()

```

<img src = https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img5.png>

- 85% our model predict it's 0 and as per our data it is also 0 label
- 83% our model predict it's 1 and as per our data it is also 1 label
- 16% model predict patient doen't have a cancer but as per data patient has
- 14% model predict patient has a cancer but as per data it is not

<p>Accuracy of our model </p>

```
model.evaluate(x_test,y_test)

```
<img src = "https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img6.png">

 <h1>
   Test the model
 </h1> 


 <p>
   This is what crucial step , when we will going to be check how our model work on unseen data
 </p>


```
import matplotlib.pyplot as plt
def  img_plot(x_test , index = 0):
    
    plt.imshow(x_test[index].astype('uint8'))  # force uint8 for proper display
    plt.axis('off')
    plt.title('Test Image')
    plt.show()


index  = 100
img_plot(x_test , index)


```

```
index  = 100
img_plot(x_test , index)
input = x_test[index:index+1]
preds = model.predict(input)[0].argmax()
label = y_test[index].argmax()
print("Predicted value " , preds)
print("True value" , label)

```

<img src = https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img7.png>

<p>
  As per our data, image 100 has a Non-cancer patient image and our mdeol predict paerson doesn't have cancer
</p>

```

index  = 4000
img_plot(x_test , index)
input = x_test[index:index+1]
preds = model.predict(input)[0].argmax()
label = y_test[index].argmax()
print("Predicted value " , preds)
print("True value" , label)

```


<img src = https://github.com/Vinit-joshi2/Breast-Cancer/blob/main/images/img8.png>

<p>
  As per our data, image 4000 has a cancer patient image and our mdeol predict paerson has cancer
</p>

