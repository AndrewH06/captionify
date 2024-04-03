# Captionify: Generate emoji captions for any image

# 1. Exploratory Data Analysis

## 1.1 Exploratory Data Analysis
We first started by exploring our captions dataset. We created a graph to better understand the distribution of word count within each caption.

<img width="400" alt="Screenshot 2024-04-02 at 5 39 25 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/2b502610-d326-43f7-9038-7564fcd82eb0">

Figure 1.1: _Distribution of Caption Length graph_

From this, we understood that the average caption was around 10-15 words in length. The distribution is mostly normal with a slight left skew. This could be due to the fact that many captions will not be shorter than 5 words as images are hard to describe with so little words. After further analysis, we came to the conclusion that 3 emojis can represent a 10-15 word sentence.

Now that we understand our word distribution, we are interested in what the words are themselves. Because our dataset is so large, we must take a holistic approach to analyzing the words. We created Table 4.1 to understand how our unique words and most common words affect our data.

<img width="400" alt="Screenshot 2024-04-02 at 5 41 02 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/b4f3ba8f-5315-4011-acd5-ef5ccf009893">

Figure 1.2

About 40% of our vocabulary were unique words which make it difficult for a model to be trained on as it creates a high risk of overfitting. But, the top 50 most common words made up over half of the total words.
We created a graph of the top 50 most common words to determine if that was the most optimal route to take, seen in Figure 4.2.

<img width="400" alt="Screenshot 2024-04-02 at 5 41 34 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/7968a174-de9c-415d-b215-c4eaa464c837">

Figure 1.3: _Bar graph displaying the top 50 most common words in the caption text file_

# 2. Methodology

## 2.1 Dataset Description
The dataset utilized for this project is the Flickr8k Dataset, which we sourced from Kaggle. This dataset contains a folder with unique images and a text file with their corresponding captions. The image file contains 8,091 JPG images, each labeled by a unique image id. Within the text file, there are five distinct descriptions that all relate to the corresponding image. Multiple descriptions allow for variability and different perspectives on captions, which ensures important aspects and details are properly captured. This helps us understand and classify the many features in our CNN (3.2.1) might recognize.

## 2.2 Machine Learning Methods
### 2.2.1 Convolutional Neural Network: VGG16
Convolutional Neural Networks are effective tools for tasks such as image classification and object detection. We employed a CNN to identify features within an image to later create a brief description of what is happening in the image. 
CNNs are especially useful when conducting feature extraction for images, being able to identify detailed features like color and shapes. CNN has this ability which other machine learning techniques can not. In our case, it was important to extract these features in order to generate a caption that could be translated into emojis. 

<img src="https://miro.medium.com/v2/resize:fit:470/1*3-TqqkRQ4rWLOMX-gvkYwA.png" alt="Structure of VGG16 CNN Image" width="400"/>

Figure 2.1: _Structure of VGG16 CNN_

### 2.2.2 Recurrent Neural Network: LSTM
We used an LSTM network, a form of RNN that allowed us to develop our description generator. We fed our RNN the features that our CNN extracted as well as training captions. As a result, we were able to predict captions based on images.
The function of a RNN involves predicting the next word in a caption. We do this by training the model on thousands of captions, with multiple caption examples for each image. Our model compares training features and captions in order to tune a sentence prediction network that utilizes LSTMs.

<img src="https://colah.github.io/images/post-covers/lstm.png" alt="Structure of LSTM Image" width="400"/>

Figure 2.2: _Structure of LSTM_

### 2.2.3 Information Retrieval Machine Learning: TF-IDF Vectorization
Our project heavily relied on being able to understand which words from our description were the most important. This is to capture the image within 3 or fewer emojis. The challenge of computers not being able to process language and words led us to use Information Retrieval (IR) Machine Learning to determine which words were the most significant. This allowed us to assign numerical values to each word in the image description to help us pick the most significant words. We needed to determine the frequency of each word to find the lowest weighted, three words in the description to translate into emojis. To do so, we incorporated a TF-IDF vectorizer to help us define significant words in our captions. A TF-IDF vectorizer takes into account all the sentences in our dataset and converts each sentence into a matrix of TF-IDF features with a higher weight for the more relevant words. 

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTrqQM9jgfEB8v0fDBaSJRJKrgaX-I4ACWhJd8cFc2G5eTM6gRhLcFL1uU6KwtgWE3VMuE&usqp=CAU" alt="The equation for TF-IDF vectorization" width="400"/>

Figure 2.3: _The equation for TF-IDF vectorization_

However, in our case, we wanted more common words, with stronger and more relevant emoji translations. More specifically, we wanted to look for words that were not rare within our corpus, this meant wanting words with lower Inverse Document Frequency. To do this we looked for the three lowest weighted (lowest TF-IDF scores) words that had emoji translations. 

### 2.2.4 Final Model

<img width="705" alt="Screenshot 2024-04-02 at 5 51 19 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/dac9e891-ca26-40ff-bedf-ef297a406f1d">

Figure 2.4: _Final Model Structure_

`input_3` contained the caption data that passed through our LSTM to be trained with the extracted features from our VGG16 network contained in `input_2`

## 2.3 Non-Machine Learning Software
To translate image descriptions into emoji representations, we used two packages: "emoji" and "emoji-translate." Since we believe colors are an important aspect of an image, we employed the "emojize" function from the "emoji" package. Through this package, we transformed colors, red, orange, yellow, green, blue, purple, black, white, and brown, into heart emojis, correlating to its respective color.
Furthermore, using Translator from the “emoji-translate” package, we translated each word in the description if correlated with an emoji. Focusing on the words that corresponded to an emoji, we found the word frequency as outlined in section 3.2.2. This method led us to result in the lowest three emoji-translatable words of the description. These three words were translated back into emojis, producing our three emoji caption for our image

# 3. Experimental Results

Our goal of creating Captionify, the emoji generator, was successfully achieved with the implementations of many machine learning techniques.We were able to generate a three-emoji caption for images inputted into our program. With images that did not generate three emoji through our model, we created exceptions to allow for expectations. With images with only two emojis generated, we had the first and third emoji be the same. While for images with only one emoji generated, the emoji would be repeated two more times to create the 3-emoji caption. 
Overall, we were able to achieve our goal while using machine learning techniques learned throughout the quarter. We utilized Convolutional Neural Networks to identify key features in an image, while also utilizing TF-IDF Vectorizer to identify common words within our corpus. These machine learning techniques and additional methods were employed to generate three emoji captions for images.

However, there are some shortcomings in our output. For example, certain images will not generate accurate captions. Image 5.5 is a good example of how our model is not trained to recognize certain things. In this example, it captions the eagle as a dog, not at all encapsulating what happens in the image.

Image 5.1

<img width="387" alt="Screenshot 2024-04-02 at 5 44 32 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/d2c4fb52-530c-437a-bdb9-6828e8ff2b84">


_Caption: 🐶🖤🍃_

Image 5.2

<img width="228" alt="Screenshot 2024-04-02 at 5 43 53 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/5d78dc61-39cc-4221-b052-dbd114a427cb">

_Caption: 🐶🤎🛣️_


Image 5.3

<img width="386" alt="Screenshot 2024-04-02 at 5 44 50 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/46fceb1e-6669-4355-8a7e-5c72815d6410">

_Caption: 🏈​​❤️🏈_

Image 5.4

<img width="378" alt="Screenshot 2024-04-02 at 5 45 13 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/97f96861-4e94-4829-b40a-81b74558084c">

_Caption: 👨‍🦰🚲🌿_

Image 5.5

<img width="384" alt="Screenshot 2024-04-02 at 5 45 29 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/2bf0dd49-fb2e-4914-ad56-6feb3a536f66">

_Caption: 🐶🖤❤️_

Image 5.6

<img width="395" alt="Screenshot 2024-04-02 at 5 45 42 PM" src="https://github.com/AndrewH06/captionify/assets/54915593/08ffc129-d18e-4bc3-973e-6da718b114e8">

_Caption: 👨‍🦰🕴️🍾_
