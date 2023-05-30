# Macarena

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
