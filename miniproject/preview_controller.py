import argparse

import numpy as np
import tqdm

from flygym.compose import ActuatorType
from miniproject import MiniprojectSimulation

from submission.controller import Controller

WINDOW_NAME = "COBAR 2026 Miniproject"
MAX_NUM_STEPS = 100_000


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
    parser.add_argument(
        "--dont-render",
        action=argparse.BooleanOptionalAction,
        help="Don't render anything, just run the simulation and print the result.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sim = MiniprojectSimulation(
        level=args.level,
        seed=args.seed,
    )
    controller = Controller(sim)

    if not args.dont_render:
        if args.dont_use_pygame_rendering:
            import cv2

            cv2.namedWindow(
                WINDOW_NAME,
            )
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

            def render(frame: np.ndarray):
                cv2.imshow(
                    WINDOW_NAME, frame[..., ::-1]
                )  # convert RGB to BGR for opencv
                cv2.waitKey(1)

            def check_quit():
                try:
                    if not cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE):
                        cv2.destroyAllWindows()
                        return True
                except:
                    cv2.destroyAllWindows()
                    return True
                return False
        else:
            import pygame

            pygame.init()

            display_size = (1024, 1024 if args.render_fly_vision else 512)
            screen = pygame.display.set_mode(display_size)
            pygame.display.set_caption(WINDOW_NAME)

            def render(frame: np.ndarray):
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                if frame_surface.get_size() != display_size:
                    frame_surface = pygame.transform.smoothscale(
                        frame_surface, display_size
                    )
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()

            def check_quit():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return True
                return False

    def got_to_food():
        banana_xy = sim.world.banana_xy
        fly_xy = np.array(sim.get_body_positions(sim.fly.name)[0][:2])
        dist = np.sqrt(np.sum((fly_xy - banana_xy) ** 2))
        return dist <= 3

    for step in tqdm.tqdm(range(MAX_NUM_STEPS)):
        if not args.dont_render and check_quit():
            print("Quit")
            break
        elif got_to_food():
            print(f"Got to goal in {step} timesteps.")
            break

        joint_angles, adhesion_signals = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step()

        if not args.dont_render and sim.render_as_needed():
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
    else:
        print("Took too long")


if __name__ == "__main__":
    main()
