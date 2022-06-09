![!automl-docker](banner.png)

# 🐳 automl-docker
With this tool, you can easily use your data to quickly create usable, scalabe maschine learning models. This CLI-based framework offers you the ability to automatically build ML models from training data into a servable Docker container. With this tool you can: 
- Easily create a maschine learning model.
- Create a docker container for the model.
- Connect to the container with an API.
- (Optionally) test your model using a UI.

##  Set-up & Installation
This repository uses various libraries, such as sklearn or our [embedders library](https://github.com/code-kern-ai/embedders). This guide also used the [clickbait_dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) to illustrate how you could use the automl-tool. The dataset is small, easy to use and already included in this repository. To use the automl-tool, you could of course use any other dataset that contains textual, labeled data. Our tool currently supports only the .csv format. 

First, you can clone this repository to your local computer. To this by typing:
```
$ git clone git@github.com:code-kern-ai/automl-docker.git
```

If you are new to GitHub, you'll find a nice guide to cloning a repository [here](https://github.com/git-guides/git-clone).

After you have cloned the repository, you simply need to install all the nessecary libraries with either pip or conda. This is very easy, as we have already collected all the necessary libraries for you. All you need to do is using one of the following commands (depending on wether you are using Anaconda):

```
$ pip install -r requirements.txt
$ conda install --file requirements.txt
```

## Getting started
Once the requirements are installed, you are ready to go! In the first step of the automl-tool, you are going to be using a CLI to load in your data, after which a maschine learning model will be created for you. To get going, start with the following command:

```
$ bash model
```

If your CLI does not have bash, you can also start by typing:
```
$ python3 ml/create_model.py
```

Once the script is started, you will be prompted to set a path to the data location on your system. Currently, only the .csv data format is usable in the tool. More data formats will follow soon!
On Mac and Linux, the path might look like this: 
```
home/user/data/training_data.csv
```

On windows, the path might look something like this:
```
C:\\Users\\yourname\\data\\training_data.csv
```

Next, you need to input the name of the columns where the training data and the labels are stored. In our example, you would use "headline" as the column for the training data and "clickbait" as the column for the training labels.
```
   headline    | clickbait
   example1    |     0
   example2    |     1
   example3    |     0
```

## Preprocessing the text data.
To make text data usable to maschines, it need to be preprocessed. To ensure state of the art maschine learning, we make use of large, pre-trained transformers pulled from [🤗 Hugging Face](https://huggingface.co/) to preprocess text data..
If you don't know what all that means: don't worry! All you need to do then is to choose between speed and accuracy by choosing one of our suggestions.
- distilbert-base-uncased -> Very accurate, state of the art method, but slow (especially on large datasets). [ENG]
- all-MiniLM-L6-v2 -> Faster, but still relatively accurate. [ENG]
- Custom model -> Input your own model from [🤗 Hugging Face](https://huggingface.co/).

By choosing "Custom model" you can always just use a different model from Hugging Face! After you have choosen your model, the text data will be processed.

## Building the maschine learning model 🚀
And that's it! Now it is time to grab a coffee, lean back and watch as your model is training on the data. You can see the training progress below.
```
76%|████████████████████████        | 7568/10000 [00:33<00:10, 229.00it/s]
```

After the training is done, we will automatically test you model and tell you, how well it is doing!
```
            Accuracy    Error     AUC
Positive    90.0%       90.0%     90.0%
Neutral     90.0%       90.0%     90.0%
Negative    90.0%       90.0%     90.0%

```

## Creating a container with Docker
Now, all the components are ready and it's time to bring them all together. Building the container is super easy! Make sure that Docker Desktop is running on your maschine. You can get it [here](https://www.docker.com/products/docker-desktop/). Next, run the following command:
```
$ bash container
```

Or, if you dont have bash in your CLI:
```
$ docker build -t automl-container-backend .
$ docker run -d -p 7531:7531 automl-container-backend

```

Building the container can take a couple of minutes. The perfect opportunity to grab yet another cup of coffee!

## Using the container 
After the container has been build, you are free to use it anywhere you want. To give you an example of what you could do with the container, we will be using it to connect to a graphical user interface that was build using the streamlit framework. If you have come this far, everything you'll need for that is already installed on your maschine! 

The user interface we build was made to get predictions on single sentences of texts. Our maschine learning model was trained on the [clickbait_dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) to be able to predict wether or not a headline of an article is clickbait or not. It's very small  So, depending on your maschine learning model, you need to modify the user interface. But don't worry: streamlit is very easy and fun to use and the [documentation](https://docs.streamlit.io/) is very helpful!

To start the streamlit application, simply use the following command:
```
$ streamlit run app/ui.py
```

And that's it! By default, the UI will automatically connect to the previouisly build container. You can test the UI on your local maschine, usually by going to the URL http://172.26.99.163:8501/ via the internet browser of your choice. 

## Roadmap
- [x] Build basic CLI to capture the data
- [x] Build mappings for language data (e.g. `EN` -> ask for `en_core_web_sm` AND recommend using `distilbert-base-uncased`)
- [x] Implement AutoML for classification (training, validation and storage of model)
- [ ] Implement AutoML for ner (training, validation and storage of model)
- [ ] Wrap instructions for build in a Dockerfile
- [ ] Add sample projects (clickbait dataset) and publish them in some posts
- [ ] Publish the repository and set up new roadmap

If you want to have something added, feel free to open an [issue](https://github.com/code-kern-ai/automl-docker/issues).

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

And please don't forget to leave a ⭐ if you like the work! 

## License
Distributed under the Apache 2.0 License. See LICENSE.txt for more information.

## Contact
This library is developed and maintained by [kern.ai](https://github.com/code-kern-ai). If you want to provide us with feedback or have some questions, don't hesitate to contact us. We're super happy to help. ✌️
