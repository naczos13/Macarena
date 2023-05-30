# Macarena

#### Video Demo:
https://youtu.be/4eFbCqDW3n4


## Description

This project is written in Python and utilizes the Mediapipe module for pose detection. The main focus of the project is to implement two simple games.

First game is called "Macarena." The game involves connecting randomly chosen hand with randomly chosen body parts. The objective is to score as many points as possible within a 30-second time limit. As a result, players often find themselves dancing the Macarena to achieve higher scores :)

Second game is caled "CopyGame" where users can record their positions and later copy them. The game involves saving pose positions as JSON files, retrieving the saved positions, and replicating them in real-time.

## Features

- Pose detection using the Mediapipe module.
- Macarena game: Connect random hand with random body parts.
- Time limit: 30 seconds to score as many points as possible.
- Interactive and engaging gameplay experience.
- CopyGame: Record and copy pose positions
- Saving and retrieving pose positions as JSON files

## Installation

1. Clone the repository:
   ```
   https://github.com/naczos13/Macarena.git
   ```

2. Navigate to the project directory:
   ```
   cd Macarena
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

> **_NOTE:_** The mediapipe module does not work with the newest python version, therfore I encurage to use Python 3.8


## Usage

1. Ensure that your camera is connected to the system.

2. Run the main Python script:
   ```
   python main.py
   ```
4. The welcome screen will be displayd along side with menu

5. Wait for the camera input to be displayed.

6. Choose the game which you want to play by pressing the proper key, listen in menu

* For 'Macarena' 

     * Press 'm' to start the game 
     * Follow the on-screen instructions to play the Macarena game.
     * Dance and connect the random hand positions with the random body parts to score points.
     * The game will automatically end after the 30-second time limit.

* For 'CopyGame'

    * Ensure that you record enough pose snapshots by pressing 's'
    * Press 'c' to start the game
    * Match your body with the pose marked on the screen
    * Continue till the pose snapchot list is not empty
    * Have fun!

    To quit the game press 'q'

## Project Design Choices

In the development of this project, several design choices were made to enhance its functionality and flexibility. These design choices are outlined below:

### Class-based Camera Processing

The camera processing functionality has been implemented as a class. This design decision was made to facilitate potential future integration of the camera processing as a widget in a web application. By encapsulating the camera processing logic within a class, it becomes easier to reuse and extend this functionality in different contexts. Also I need 'global' variables that share the state of the curent app. This is easly done by 'self' in class.

### Multithreading for Independent Processes

Multithreading has been employed to handle two computationally independent processes. The first process involves capturing the camera frames, while the second process involves processing the frames to detect human poses and draw the necessary elements for an enhanced user experience. By using multithreading, these processes can run concurrently, improving the overall performance and responsiveness of the application.

### Storage of Pose Snapshots in JSON Format

Pose snapshots are stored in a JSON file as a list of key-value pairs. This design choice aligns with the implementation of landmarks used by the Mediapipe library. Storing the pose snapshots in JSON format provides a structured and easily interpretable way to save and retrieve the recorded pose positions. This facilitates the functionality of the CopyGame, where users can save and later replicate the stored poses.

By making these design choices, the project offers modularity, extensibility, and ease of integration into different environments or applications. It ensures a smooth user experience while providing the flexibility to incorporate additional features and improvements in the future.

## License

[MIT License](LICENSE)

## Contributing

Contributions to the project are always welcome. Here's how you can contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your enhancements/changes.
4. Commit your changes and push the branch.
5. Submit a pull request.

## Authors

- [Marcin Naczk](https://github.com/naczos13)

## Acknowledgements

- [Mediapipe](https://mediapipe.dev/) - Pose detection module.

## Contact

If you have any questions, suggestions, or feedback, please open the github issue.



Design choices.

I implement the camera processing as class becuase it could be helpful if I want to in the future implement it as as wiget in some web app.
I use the multithreading becuse I spotted two compute independent processes. One is capturing the camera frame. The second is processing frame, to detect the human pose and draw the necessary elments to make the game more user friendly.
I store the pose snapshots in json file as list of key value pair, becuse it simalar to the implementatio of the landmark used by mediapipe.
