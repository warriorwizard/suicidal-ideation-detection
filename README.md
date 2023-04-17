# suicidal-ideation-detection
Text Classification using LSTM and CNN
In this project, we built a text classification model using a combination of LSTM and CNN architectures to predict whether a given text contains suicidal message or not.

Dataset
The dataset used for this project contains texts of suicidal and non-suicidal chats, which are labeled as 0 and 1. The dataset is loaded using pandas and separated into the text and label columns. The text is preprocessed by tokenizing the text using the Tokenizer class from Keras and padding the sequences to have the same length.

Model Architecture
The text classification model uses a combination of LSTM and CNN architectures to analyze the textual data. The input layer of the model is an Embedding layer, which converts the text input into dense vectors of fixed size. This layer is followed by a combination of LSTM and CNN layers, which allow the model to learn both short-term and long-term dependencies in the text. Finally, the output layer of the model is a dense layer with a sigmoid activation function, which outputs a value between 0 and 1 that represents the probability that the input text contains a suicidal message.

Model Training and Evaluation
The model is trained on the preprocessed text data and labels using the fit method from Keras. The model is evaluated using the test data and labels using the evaluate method from Keras. The performance of the model is evaluated using the accuracy and the confusion matrix.

Flask App
We created a Flask app that allows the user to enter their own text for prediction. The app loads the trained model and tokenizer, preprocesses the user's input, and generates a prediction. The prediction result is displayed on the web page.

Conclusion
The text classification model using LSTM and CNN architectures was able to accurately predict whether a given text contains suicidal message or not. The Flask app provides an easy-to-use interface for users to enter their own text for prediction.



