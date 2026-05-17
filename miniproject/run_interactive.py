import argparse

import numpy as np
import pygame

from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController
from miniproject.interactive import GameState
from miniproject import MiniprojectSimulation

WINDOW_NAME = "COBAR 2026 Miniproject"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--keyboard-mode",
        choices=["hold", "sticky"],
        default="hold",
        help=(
            "Keyboard control mode: 'sticky' keeps last gait command until changed, "
            "'hold' only walks while movement keys are actively pressed."
        ),
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=4,
        help="The level of the simulation to run. Default is 0.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="The random seed for the simulation. Default is 0.",
    )
    parser.add_argument(
        "--dont-use-pygame-rendering",
        action=argparse.BooleanOptionalAction,
        help=(
            "If experiencing rendering issues, set this option to use opencv rendering instead of pygame."
            "Also requires installing the pynput library."
        ),
    )
    parser.add_argument(
        "--render-fly-vision",
        action=argparse.BooleanOptionalAction,
        help="Whether to also render what the fly sees from its perspective.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Game state and control setup
    game_state = GameState()

    sim = MiniprojectSimulation(
        level=args.level,
        seed=args.seed,
    )
    controller = TurningController(sim.timestep)

    pygame.init()

    print("Getting controllers")
    if args.dont_use_pygame_rendering:
        from miniproject.interactive.controls_pynput import KeyboardControlPynput
        import cv2

        controls = KeyboardControlPynput(game_state)
        cv2.namedWindow(
            WINDOW_NAME,
        )
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        def get_controller_gains():
            pygame.event.pump()
            gain_left, gain_right = controls.get_actions()
            return gain_left, gain_right

        def render(frame: np.ndarray):
            cv2.imshow(WINDOW_NAME, frame[..., ::-1])  # convert RGB to BGR for opencv
            cv2.waitKey(1)

    else:
        from miniproject.interactive import KeyboardControl

        controls = KeyboardControl(game_state)
        display_size = (1024, 1024 if args.render_fly_vision else 512)
        screen = pygame.display.set_mode(display_size)
        pygame.display.set_caption(WINDOW_NAME)

        def get_controller_gains():
            events = pygame.event.get()
            controls.process_events(events)
            keys_pressed = pygame.key.get_pressed()
            gain_left, gain_right = controls.get_actions(keys_pressed)
            return gain_left, gain_right

        def render(frame: np.ndarray):
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if frame_surface.get_size() != display_size:
                frame_surface = pygame.transform.smoothscale(
                    frame_surface, display_size
                )
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

    step = 0
    while not game_state.get_quit():
        gain_left, gain_right = get_controller_gains()

        if game_state.get_reset():
            pass  # not implemented yet
        if game_state.get_quit():
            break

        joint_angles, adhesion_signals = controller.step(
            np.array([gain_left, gain_right])
        )
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step()

        if sim.render_as_needed():
            # Render the latest RGB frame from flygym into the pygame window.
            frame = np.concatenate(
                [frames[-1] for frames in sim.renderer.frames.values()], axis=-2
            )
            if args.render_fly_vision:
                fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
                fly_vision = np.pad(
                    fly_vision,
                    (
                        [0] * 2,
                        [(frame.shape[1] - fly_vision.shape[1]) // 2] * 2,
                        [0] * 2,
                    ),
                )
                frame = np.vstack((fly_vision, frame))
            render(frame)
        step += 1

    controls.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
