# Miniproject for BIOENG-456: Controlling behavior in animals and robots

Welcome to the Miniproject for BIOENG-456!

## Setup
Run `uv sync` to make sure you have all the dependencies installed
```bash
uv sync
```

## Usage

To explore the levels interactively with the keyboard, run the `run_interactive.py` script. Then you can use the WASD keys to control the fly (Q to stop and ESC to exit).

```bash
uv run miniproject/run_interactive.py --level <level> --seed <seed>
```

Replace `<level>` with the desired level number (0 to 4 for the 5 levels) and `<seed>` with the random seed for reproducibility.

If you want to see the fly's vision as well, add the `--render-fly-vision` argument.

> If you have an issue with the rendering (black screen - mainly seen on linux), you can try to add the argument `--dont-use-pygame-rendering` to fallback to opencv rendering instead of pygame. Note: first you will have to run `uv pip install pynput` to get the pynput library.

The `run_simulation.ipynb` notebook contains code that will be used to evaluate the controller.

## Creating a private copy while keeping track of the changes from the public repository
1. Clone this repository
```sh
git clone https://github.com/NeLy-EPFL/cobar-2026 cobar-miniproject-2026
cd cobar-miniproject-2026
```
2. Create a New Private Repository on GitHub:
- Go to GitHub and create a new private repository.
- Do not initialize it with a README, .gitignore, or any other files.
3. Set the New Private Repository as a Remote:
```sh
git remote rename origin upstream
git remote add origin https://github.com/<your_username>/cobar-miniproject-2026
```
4. Push the Cloned Repository to Your Private Repository:
```sh
git push -u origin main
```
5. We will notify you if there are important changes to the repository. To fetch updates from the public repository and merge them into your private repository, use the following commands:
```sh
git fetch upstream
git merge upstream/main
```
