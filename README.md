![!automl-docker](banner.png)

# 🐳 automl-docker
With this tool, you can easily use your data to quickly create usable, scalabe maschine learning models. This CLI-based framework offers you the ability to automatically build ML models from training data into a servable Docker container. With this tool you can: 
- Easily create a maschine learning model.
- Create a docker container for the model.
- (Optionally) test your model using a streamlit interface.

##  Set-up & Installation
This repository uses various libraries, such as sklearn or our [embedders library](https://github.com/code-kern-ai/embedders).  

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

## Usage
Once the requirements are installed, you are ready to go! In the first step of the automl-tool, you are going to be using a CLI to load in your data, after which a maschine learning model will be created for you. To get going, start with the following command:

```
$ bash model
```
If your system does not have bash, you can also start by typing:
```
$ python3 ml/create_model.py
```
Once the script has started, you will be prompted to set a path to the data location on your system. Currently, only the .csv format is usable in the tool. More data formats will follow soon!
On windows, the path might look something like this:
```
C:\\Users\\yourname\\data\\training_data.csv
```
On Mac and Linux, the path might look like this: 
```
home/user/data/training_data.csv
```

Next, you need to input the name of the columns where the training data and the labels are stored.
```
training_data  |  labels
   example1    |     0
   example2    |     1
   example3    |     0
```

## Preprocessing the text data.
To make text data usable to maschines, it need to be preprocessed. To ensure state of the art maschine learning, we make use of large, pre-trained transformers pulled from [🤗 Hugging Face](https://huggingface.co/) to preprocess text data..
If you don't know what all that means: don't worry! All you need to do then is to choose between speed and accuracy by choosing one of our suggestions.
- distilbert-base-uncased -> Very accurate, state of the art method, but slow (especially on large datasets). [ENG]")
- all-MiniLM-L6-v2 -> Faster, but still relatively accurate. [ENG]")
- Custom model -> Input your own model from https://huggingface.co/.")

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

The Dockerfile to run this service has been stored to "training_data-distilbert-base-uncased.Dockerfile". You can find a detailed report of the performance under "training_data-distilbert-base-uncased.xlsx".
```
## Creating a container with Docker

Now, all the components are ready and it's time to bring them all together. Building the container is super easy! Make sure, that Docker Desktop is running on your maschine. You can get it [here](https://www.docker.com/products/docker-desktop/). Next, running the following command:
```
$ bash container
```
Or, if you dont have bash:
```
$ docker build -t automl-container-backend .
$ docker run -d -p 7531:7531 automl-container-backend

```
Building the container can take a couple of minutes. The perfect opportunity to grab yet another cup of coffee!

## (Optional) testing out the model using a user interface with streamlit


## Roadmap
- [ ] Build basic CLI to capture the data
- [ ] Build mappings for language data (e.g. `EN` -> ask for `en_core_web_sm` AND recommend using `distilbert-base-uncased`)
- [ ] Implement AutoML for classification (training, validation and storage of model)
- [ ] Implement AutoML for ner (training, validation and storage of model)
- [ ] Wrap instructions for build in a Dockerfile
- [ ] Add sample projects (twitter sentiment analysis, intent classification and some named entity recognition) and publish them in some posts
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
