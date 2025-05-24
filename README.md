# DeepExtract &#x0001F3A4;

## Overview

**DeepExtract** is a powerful and efficient tool designed to separate vocals and sounds from audio files, providing an enhanced experience for musicians, producers, and audio engineers. With DeepExtract, you can quickly and effectively isolate vocals or instruments from mixed audio tracks, facilitating tasks like remixing, karaoke preparation, or audio analysis.

We use [MDX Net](https://arxiv.org/abs/2111.12203) models for this.

## Installation Guide &#x0001F6E0;&#xFE0F;

Setting up **DeepExtract** is quick and straightforward! Simply follow the steps below to get started.

### Step 1: Clone the Repository

1. Clone this repository to your ComfyUI custom nodes folder. There is two way :

- A) Download this repository as a zip file and extract files in to `comfyui\custom_nodes\ComfyUI-DeepExtract` folder.

- B) Go to `comfyui\custom_nodes\` folder open a terminal window here and run `git clone https://github.com/set-soft/ComfyUI-DeepExtract` command.

### Step 2: Model download

You can use the auto-download feature. In this case you'll have just one model, it will be automatically downloaded the first time you run the node. The default node is the Kimberley Jensen Vocal 2 model, this is really good to extract the vocals.

If you want to play with more MDX Net models download them from internet and put them in the *audio/MDX* folder inside the ComfyUI *models* folder.

You can find various models at [HuggingFace](https://huggingface.co/seanghay/uvr_models/tree/main). Some models are designed to extract the vocals and others the instruments. You can get the instruments using a model for vocals, just connect the complement output. But for optimal results use a model to extract what you want.

### Step 3: Install dependencies

2. Go to `comfyui\custom_nodes\ComfyUI-DeepExtract` folder and open a terminal window and run `pip install -r requirements.txt` command. If you using windows you can double click `setup.bat` alternatively.

```bash
pip install -r requirements.txt
```

3. Wait patiently installation to finish.

4. Run the ComfyUI.

5. Double click anywhere in ComfyUI and search DeepExtract node by typing it or right click anywhere and select `Add Node > DeepExtract > VocalAndSoundSeparatorNode` node to using it.

<img src="https://github.com/set-soft/ComfyUI-DeepExtract/blob/main/public/images/node_location.png?raw=true" alt="nodel location" width="100%"/>

##### OR

<img src="https://github.com/set-soft/ComfyUI-DeepExtract/blob/main/public/images/node_search.png?raw=true" alt="nodel location" width="100%"/>

## Usage

### How to Use the DeepExtract Node

To utilize the **DeepExtract** node, simply connect your audio input to the **VocalAndSoundRemoverNode**. Adjust the parameters to tailor the output to your needs. The node will process the audio and return isolated vocal and background tracks for further manipulation.

### Example Workflow

1. **Load an Audio File:** Begin by loading your mixed audio file into ComfyUI.
2. **Add the Node:** Insert the **VocalAndSoundRemoverNode** into your workflow.
3. **Connect Inputs and Outputs:** Link your audio source to the node and specify where to send the separated tracks.
4. **Process the Audio:** Execute the workflow to separate the vocals and sounds effectively.

## Structure

<img src="https://github.com/set-soft/ComfyUI-DeepExtract/blob/main/public/images/node_structure.png?raw=true" alt="nodel location" width="100%"/>

### Node Layout

The **DeepExtract** node features an intuitive interface that allows for easy manipulation. The input section accepts mixed audio files, while the output section provides two distinct tracks: one for isolated vocals and another for the background sounds. This design facilitates seamless integration into your audio processing workflow.

### Parameter Overview

- **Input Sound:** This is where you connect the mixed audio file.
- **Main Output:** This output provides the isolated main track. The default is the vocals.
- **Complement Output:** This output delivers the remaining sound. The default are instruments.
- **model_filename:** When using *Default* the *Kim_Vocal_2.onnx* model is used. This model is downloaded if you don't have it installed. If you have other models in the *audio/MDX* folder you can select the model to use here.

These additions, along with your original text, will create a clearer understanding of how to use the **DeepExtract** tool effectively!

## Contributing

We welcome contributions from the community! If you'd like to enhance DeepExtract, please fork the repository and submit a pull request.

### Guidelines

1. Fork the project.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Submit a pull request.

### Author

&#x0001F464; **Abdullah Ozmantar**  
[GitHub Profile](https://github.com/comfyui-abdozmantar)

&#x0001F464; Modifications: **Salvador E. Tropea**  
[GitHub Profile](https://github.com/set-soft)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/set-soft/comfyui-deepextract/blob/main/LICENSE) file for details.

